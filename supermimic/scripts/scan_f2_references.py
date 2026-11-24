"""Nominate G1 motion windows for physical F2 verification."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from audit_forward_gait import G1_XML, foot_geometry

SPEEDS = (0.22, 0.35, 0.50)


def wrap(angle: np.ndarray) -> np.ndarray:
  return np.arctan2(np.sin(angle), np.cos(angle))


def rolling_mean(signal: np.ndarray, frames: int) -> np.ndarray:
  total = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
  return (total[frames:] - total[:-frames]) / frames


def nominate(
  path: Path,
  model: mujoco.MjModel,
  frames: int,
  count: int,
  directional_transitions: bool = False,
  quiet_arms: bool = False,
  forward_speeds: tuple[float, ...] = SPEEDS,
) -> dict[str, list[dict]]:
  with np.load(path) as data:
    q, p, quat = (data[key] for key in ("joint_pos", "root_pos_w", "root_quat_w"))
    joint_names = data["joint_names"].tolist()
  if len(q) < frames + 1:
    raise ValueError(f"{path} is too short for a {frames}-frame window.")

  rows = [
    {
      "pelvis_position_w_m": pos,
      "pelvis_quaternion_wxyz": rot,
      "posture": {"joint_pos_rad": joints},
    }
    for pos, rot, joints in zip(p, quat, q, strict=True)
  ]
  sites, soles, _ = foot_geometry(model, rows, joint_names)
  dt = 0.02
  w, x, y, z = quat.T
  yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
  left = np.column_stack((-forward[:, 1], forward[:, 0]))
  displacement = (p[frames:, :2] - p[:-frames, :2]) / (frames * dt)
  mean_forward = np.sum(displacement * forward[:-frames], axis=1)
  mean_left = np.sum(displacement * left[:-frames], axis=1)
  mean_yaw = wrap(yaw[frames:] - yaw[:-frames]) / (frames * dt)
  velocity_w = np.gradient(p[:, :2], dt, axis=0)
  speed = np.sum(velocity_w * forward, axis=1)
  tilt = np.rad2deg(2 * np.arctan2(np.hypot(x, y), np.hypot(w, z)))
  knees = np.rad2deg(
    q[:, [joint_names.index(f"{side}_knee_joint") for side in ("left", "right")]]
  )
  sole_speed = np.linalg.norm(np.gradient(sites[..., :2], dt, axis=0), axis=-1)
  # The source can have a uniform retargeting height offset. This contact estimate
  # nominates candidates only. The MuJoCo replay still measures contact independently.
  ground = np.clip(np.quantile(soles, 0.10), -0.05, 0.05)
  contact = (soles <= ground + 0.018) & (sole_speed < 0.26)
  events = ~contact[:-1] & contact[1:]
  touchdown_counts = np.stack(
    [np.rint(rolling_mean(events[:, side], frames) * frames) for side in (0, 1)],
    axis=1,
  )
  low_pace = rolling_mean(speed < 0.08, frames)[:-1]
  unsupported = rolling_mean(~contact.any(axis=1), frames)[:-1]
  high_knee = rolling_mean(np.maximum(knees - 38, 0).mean(axis=1), frames)[:-1]
  tilt_mean = rolling_mean(tilt, frames)[:-1]
  pose_cost = 0.75 * unsupported + 0.03 * high_knee + 0.014 * tilt_mean
  arm_cost = 0.0
  if quiet_arms:
    arm_names = [
      f"{side}_{joint}_joint"
      for side in ("left", "right")
      for joint in ("shoulder_pitch", "shoulder_roll", "elbow")
    ]
    arm_pose = q[:, [joint_names.index(name) for name in arm_names]]
    means = np.stack([rolling_mean(a, frames)[:-1] for a in arm_pose.T], axis=1)
    means2 = np.stack([rolling_mean(a * a, frames)[:-1] for a in arm_pose.T], axis=1)
    # Prefer quiet shoulders and elbows around a relaxed, lowered posture.
    excursion = np.sqrt(np.maximum(means2 - means**2, 0.0)).mean(axis=1)
    target = np.array((0.12, 0.14, 0.65, 0.12, -0.14, 0.65))
    posture = np.linalg.norm(means - target, axis=1)
    arm_cost = 5.0 * excursion + 0.45 * posture
  straight_cost = abs(mean_left) / 0.12 + abs(mean_yaw) / 0.50
  boundary = min(frames // 3, 30)
  index = np.arange(len(mean_forward))
  pitch_speed = rolling_mean(abs(speed), boundary)
  support = rolling_mean(contact.all(axis=1), boundary)
  start_speed, end_speed = pitch_speed[index], pitch_speed[index + frames - boundary]
  start_support, end_support = support[index], support[index + frames - boundary]
  scores = {
    f"{target:.2f}": straight_cost
    + pose_cost
    + arm_cost
    + abs(mean_forward - target) / 0.08
    + 1.2 * low_pace
    + 0.25 * (touchdown_counts.min(axis=1) < 2)
    for target in forward_speeds
  }
  scores["start"] = (
    straight_cost
    + pose_cost
    + start_speed / 0.035
    + (1 - start_support)
    + abs(end_speed - 0.28) / 0.10
  )
  scores["stop"] = (
    straight_cost
    + pose_cost
    + end_speed / 0.035
    + (1 - end_support)
    + abs(start_speed - 0.28) / 0.10
  )
  if directional_transitions:
    lateral = np.sum(velocity_w * left, axis=1)
    angular = np.gradient(yaw, dt)
    twist = np.column_stack((speed, lateral, angular))
    prefix_twist = np.stack(
      [rolling_mean(axis, boundary)[index] for axis in twist.T], axis=1
    )
    suffix_twist = np.stack(
      [rolling_mean(axis, boundary)[index + frames - boundary] for axis in twist.T],
      axis=1,
    )
    rest_speed = np.sqrt(speed**2 + lateral**2 + (0.15 * angular) ** 2)
    rest = rolling_mean(rest_speed, boundary)[index]
    for label, goal in (
      ("start_left", (0.25, 0.20, 0.0)),
      ("start_right", (0.25, -0.20, 0.0)),
      ("start_turn_left", (0.35, 0.0, 0.4)),
      ("start_turn_right", (0.35, 0.0, -0.4)),
    ):
      error = (suffix_twist - goal) / (0.10, 0.09, 0.24)
      scores[label] = (
        pose_cost + rest / 0.035 + (1 - start_support) + np.linalg.norm(error, axis=1)
      )
    for label, goal in (
      ("stop_left", (0.25, 0.20, 0.0)),
      ("stop_right", (0.25, -0.20, 0.0)),
    ):
      error = (prefix_twist - goal) / (0.10, 0.09, 0.24)
      terminal = rolling_mean(rest_speed, boundary)[index + frames - boundary]
      scores[label] = (
        pose_cost + terminal / 0.035 + (1 - end_support) + np.linalg.norm(error, axis=1)
      )
  result = {}
  for label, score in scores.items():
    ranked = []
    for frame in np.argsort(score):
      if any(abs(row["frame"] - frame) < frames for row in ranked):
        continue
      ranked.append(
        {
          "frame": int(frame),
          "cost": float(score[frame]),
          "vx": float(mean_forward[frame]),
          "vy": float(mean_left[frame]),
          "yaw_rate": float(mean_yaw[frame]),
          "fraction_under_008_mps": float(low_pace[frame]),
          "kinematic_touchdowns": touchdown_counts[frame].astype(int).tolist(),
          "prefix_abs_vx": float(start_speed[frame]),
          "suffix_abs_vx": float(end_speed[frame]),
        }
      )
      if directional_transitions:
        ranked[-1]["prefix_twist"] = prefix_twist[frame].tolist()
        ranked[-1]["suffix_twist"] = suffix_twist[frame].tolist()
      if len(ranked) == count:
        break
    result[label] = ranked
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motions", nargs="+", type=Path, required=True)
  parser.add_argument("--seconds", type=float, default=3.0)
  parser.add_argument("--count", type=int, default=2)
  parser.add_argument("--directional-transitions", action="store_true")
  parser.add_argument("--quiet-arms", action="store_true")
  parser.add_argument("--forward-speeds", type=float, nargs="+", default=SPEEDS)
  args = parser.parse_args()
  if not np.isfinite(args.seconds) or args.seconds < 0.2 or args.count < 1:
    parser.error(
      "Choose a window of at least 0.2 seconds and a positive candidate count."
    )
  if not all(np.isfinite(speed) and speed > 0 for speed in args.forward_speeds):
    parser.error("Forward scan speeds must be positive and finite.")
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  frames = round(args.seconds * 50)
  print(
    json.dumps(
      {
        path.stem: nominate(
          path,
          model,
          frames,
          args.count,
          args.directional_transitions,
          args.quiet_arms,
          tuple(args.forward_speeds),
        )
        for path in args.motions
      },
      indent=2,
      allow_nan=False,
    )
  )


if __name__ == "__main__":
  main()
