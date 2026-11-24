"""Measure complete walking cycles in local G1 eval_p0 posture traces."""

import argparse
import json
import re
from itertools import islice
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

SIDES = ("left", "right")


def foot_geometry(
  model: mujoco.MjModel, rows: list[dict], joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return foot sites, lowest collision points, and world-frame forward axes."""
  foot_geoms = [
    [
      i
      for i in range(model.ngeom)
      if re.fullmatch(rf"{side}_foot\d+_collision", model.geom(i).name)
    ]
    for side in SIDES
  ]
  if any(not ids for ids in foot_geoms):
    raise ValueError("Expected foot collision capsules in the G1 model.")
  if any(
    np.any(model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_CAPSULE) for ids in foot_geoms
  ):
    raise ValueError("Sole clearance requires capsule geometry.")
  site_ids = [model.site(f"{side}_foot").id for side in SIDES]
  qpos_ids = [model.joint(name).qposadr.item() for name in joint_names]
  root = model.joint(0).qposadr.item()
  data = mujoco.MjData(model)
  sites = np.empty((len(rows), 2, 3))
  soles = np.empty((len(rows), 2))
  forward = np.empty_like(sites)
  for i, row in enumerate(rows):
    data.qpos[root : root + 3] = row["pelvis_position_w_m"]
    data.qpos[root + 3 : root + 7] = row["pelvis_quaternion_wxyz"]
    data.qpos[qpos_ids] = row["posture"]["joint_pos_rad"]
    mujoco.mj_kinematics(model, data)
    sites[i] = data.site_xpos[site_ids]
    forward[i] = data.site_xmat[site_ids].reshape(2, 3, 3)[:, :, 0]
    for side, ids in enumerate(foot_geoms):
      # Capsule local z follows the line between its two spherical cap centers.
      axis_z = data.geom_xmat[ids, 8]
      radius, half_length = model.geom_size[ids, :2].T
      soles[i, side] = (
        data.geom_xpos[ids, 2] - abs(axis_z) * half_length - radius
      ).min()
  return sites, soles, forward


def bridge_single_ground_dropout(contact: np.ndarray, soles: np.ndarray) -> np.ndarray:
  """Bridge a one-frame dropout only if the collision sole is within 2 mm of flat ground."""
  result = contact.copy()
  result[1:-1] |= contact[:-2] & contact[2:] & (np.abs(soles[1:-1]) <= 0.002)
  return result


def metrics(
  model: mujoco.MjModel,
  rows: list[dict],
  names: list[str],
  dt: float,
  *,
  bridge_ground_contact: bool = False,
) -> dict:
  sites, soles, foot_forward = foot_geometry(model, rows, names)
  contact = np.asarray([r["foot_contact"] for r in rows], dtype=bool)
  if bridge_ground_contact:
    contact = bridge_single_ground_dropout(contact, soles)
  speed = np.asarray([r["actual_velocity"] for r in rows])
  foot_speed = np.asarray([r["posture"]["foot_speed_mps"] for r in rows])
  q = np.asarray([r["posture"]["joint_pos_rad"] for r in rows])
  w, x, y, z = np.asarray([r["pelvis_quaternion_wxyz"] for r in rows]).T
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  heading = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
  foot_gap = ((sites[:, 0, :2] - sites[:, 1, :2]) * heading).sum(-1)
  leftward = np.stack((-heading[:, 1], heading[:, 0]), axis=-1)
  lateral_foot_gap = ((sites[:, 0, :2] - sites[:, 1, :2]) * leftward).sum(-1)
  foot_yaw = np.rad2deg(
    np.arctan2(
      (foot_forward[..., :2] * leftward[:, None, :]).sum(-1),
      (foot_forward[..., :2] * heading[:, None, :]).sum(-1),
    )
  )
  pelvis = np.asarray([r["pelvis_position_w_m"] for r in rows])
  travel = pelvis[-1, :2] - pelvis[0, :2]
  if np.linalg.norm(travel) > 1e-5:
    path_left = np.array((-travel[1], travel[0])) / np.linalg.norm(travel)
  else:
    path_left = leftward[0]
  sway = (pelvis[:, :2] - pelvis[0, :2]) @ path_left
  smoothing = min(len(rows), max(1, round(0.12 / dt)))
  lateral_speed_smoothed = np.convolve(
    speed[:, 1], np.ones(smoothing) / smoothing, mode="valid"
  )
  touchdown_events: list[tuple[int, int]] = []
  per_foot = {}
  for side, name in enumerate(SIDES):
    active = contact[:, side]
    takeoff = np.flatnonzero(active[:-1] & ~active[1:]) + 1
    landing = np.flatnonzero(~active[:-1] & active[1:]) + 1
    swing_peak = []
    step_lengths = []
    swing_duration = []
    for start in takeoff:
      end_index = landing.searchsorted(start)
      if end_index == len(landing):
        continue
      end = landing[end_index]
      # Exclude single-frame contact interruptions and require persistent landing.
      if end - start < 3 or end + 2 >= len(active) or not active[end : end + 3].all():
        continue
      touchdown_events.append((int(end), side))
      swing_peak.append(float(soles[start:end, side].max()))
      step_lengths.append(float(foot_gap[end] * (1 - 2 * side)))
      swing_duration.append(float((end - start) * dt))
    # Removing two samples on both contact edges reduces roll and impact effects.
    interior = np.logical_and.reduce(
      [active[i : len(active) - 4 + i] for i in range(5)]
    )
    slip = foot_speed[2:-2, side][interior]
    knee = q[active, names.index(f"{name}_knee_joint")]
    per_foot[name] = {
      "complete_swings": len(swing_peak),
      "peak_sole_clearance_m": swing_peak,
      "step_length_at_touchdown_m": step_lengths,
      "swing_duration_s": swing_duration,
      "interior_contact_samples": len(slip),
      "interior_contact_site_speed_p90_mps": float(np.quantile(slip, 0.9))
      if len(slip)
      else None,
      "stance_knee_pitch_median_deg": float(np.rad2deg(np.median(knee)))
      if len(knee)
      else None,
      "stance_foot_yaw_median_deg": float(np.median(foot_yaw[active, side]))
      if active.any()
      else None,
    }
  touchdown_events.sort()
  alternating = [
    later - earlier
    for (earlier, a), (later, b) in zip(
      touchdown_events, touchdown_events[1:], strict=False
    )
    if a != b and later > earlier
  ]
  joints = {
    part: q[:, [names.index(f"{side}_{part}_joint") for side in SIDES]]
    for part in ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch")
  }
  leg_arm = np.corrcoef(
    np.diff(joints["hip_pitch"], axis=1)[:, 0],
    np.diff(joints["shoulder_pitch"], axis=1)[:, 0],
  )[0, 1]
  return {
    "samples": len(rows),
    "recorded_reference_frames": [
      rows[0]["reference_frame"],
      rows[-1]["reference_frame"],
    ],
    "mean_heading_velocity": speed.mean(0).tolist(),
    "pelvis_height_p05_p50_p95_m": np.quantile(
      pelvis[:, 2], (0.05, 0.5, 0.95)
    ).tolist(),
    "pelvis_lateral_speed_std_mps": float(speed[:, 1].std()),
    "pelvis_lateral_speed_smoothed_std_mps": float(lateral_speed_smoothed.std()),
    "pelvis_lateral_sway_p95_minus_p05_m": float(
      np.quantile(sway, 0.95) - np.quantile(sway, 0.05)
    ),
    "torso_tilt_p50_p90_deg": np.quantile(
      [r["posture"]["torso_tilt_deg"] for r in rows], (0.5, 0.9)
    ).tolist(),
    "foot_separation_lateral_median_m": float(np.median(lateral_foot_gap)),
    "hip_roll_median_deg": np.rad2deg(
      np.median(q[:, [names.index(f"{side}_hip_roll_joint") for side in SIDES]], axis=0)
    ).tolist(),
    "double_support_fraction": float(contact.all(axis=1).mean()),
    "no_support_fraction": float((~contact.any(axis=1)).mean()),
    "alternating_touchdown_interval_s": [n * dt for n in alternating],
    "excursion_p95_minus_p05_deg": {
      part: np.rad2deg(
        np.quantile(angle, 0.95, axis=0) - np.quantile(angle, 0.05, axis=0)
      ).tolist()
      for part, angle in joints.items()
    },
    "hip_shoulder_difference_corr": float(leg_arm) if np.isfinite(leg_arm) else None,
    "feet": per_foot,
  }


def render_comparison(
  output: Path, traces: list[tuple[str, list[dict], list[str]]], dt: float
) -> None:
  from PIL import Image, ImageDraw

  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[30, 30, 0.1],
    rgba=[0.18, 0.22, 0.28, 1.0],
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  camera = mujoco.MjvCamera()
  camera.distance = 2.35
  camera.elevation = -6.0
  root = model.joint(0).qposadr.item()
  columns = []
  for label, rows, names in traces:
    joints = [model.joint(name).qposadr.item() for name in names]
    start = np.asarray(rows[0]["pelvis_position_w_m"])[:2]
    travel = np.asarray(rows[-1]["pelvis_position_w_m"])[:2] - start
    heading = np.rad2deg(np.arctan2(travel[1], travel[0]))
    columns.append((label, rows, joints, start, travel, heading))

  width, height = 400, 246
  frames = []
  with mujoco.Renderer(model, height=height, width=width) as renderer:
    for step in range(0, min(len(rows) for _, rows, _ in traces), 2):
      frame = Image.new("RGB", (width * len(columns), height * 2))
      for col, (label, rows, joints, start, travel, heading) in enumerate(columns):
        row = rows[step]
        data.qpos[root : root + 3] = row["pelvis_position_w_m"]
        data.qpos[root + 3 : root + 7] = row["pelvis_quaternion_wxyz"]
        data.qpos[joints] = row["posture"]["joint_pos_rad"]
        mujoco.mj_forward(model, data)
        # Follow the mean path so the pelvis' lateral transfer remains visible.
        camera.lookat[:2] = start + travel * step / (len(rows) - 1)
        camera.lookat[2] = 0.67
        for panel, (view, offset) in enumerate((("side", 90), ("front", 180))):
          camera.azimuth = heading + offset
          renderer.update_scene(data, camera=camera)
          snapshot = Image.fromarray(renderer.render())
          draw = ImageDraw.Draw(snapshot)
          draw.rectangle((0, 0, width, 24), fill="#20262d")
          speed = row["actual_velocity"][0]
          draw.text((8, 6), f"{label} | {view} | {speed:.2f} m/s", fill="white")
          frame.paste(snapshot, (col * width, panel * height))
      frames.append(frame)
  frames[0].save(
    output, save_all=True, append_images=frames[1:], duration=round(2_000 * dt), loop=0
  )
  frames[len(frames) // 3].save(output.with_suffix(".png"))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--trace",
    nargs=4,
    action="append",
    required=True,
    metavar=("LABEL", "JSONL", "FIRST_ROW", "END_ROW"),
    help="Compare the zero-based row interval [FIRST_ROW, END_ROW) in each P0 trace.",
  )
  parser.add_argument("--replay-gif", type=Path, help="Render paired side and front views.")
  parser.add_argument(
    "--bridge-ground-contact", action="store_true",
    help="Diagnose one-frame foot contact dropouts near the flat ground plane.",
  )
  args = parser.parse_args()
  if args.replay_gif and args.replay_gif.suffix.lower() != ".gif":
    parser.error("The replay output must use the .gif extension.")
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  result = {}
  traces = []
  frame_intervals = []
  for label, filename, start, stop in args.trace:
    path = Path(filename)
    summary = json.loads(path.with_name("summary.json").read_text())
    first, end = int(start), int(stop)
    if first < 0 or end - first < 5:
      parser.error(f"{label}: specify a nonnegative trace window of at least five steps.")
    with path.open() as stream:
      rows = [json.loads(line) for line in islice(stream, first, end)]
    if label in result or len(rows) != end - first:
      parser.error(f"{label}: duplicate label or incomplete trace window.")
    names = summary["joint_position_order"]
    dt = rows[1]["state_time_s"] - rows[0]["state_time_s"]
    result[label] = metrics(
      model, rows, names, dt, bridge_ground_contact=args.bridge_ground_contact
    )
    if args.replay_gif:
      traces.append((label, rows, names))
      frame_intervals.append(dt)
  if args.replay_gif:
    if not np.allclose(frame_intervals, frame_intervals[0]):
      parser.error("Comparison replay requires traces with the same timestep.")
    render_comparison(args.replay_gif, traces, frame_intervals[0])
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
