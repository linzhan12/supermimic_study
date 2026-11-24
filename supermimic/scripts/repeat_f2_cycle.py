"""Build a straight walking reference from two compatible touchdown events."""

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp


def close_sequence(
  position: np.ndarray,
  velocity: np.ndarray,
  endpoint: np.ndarray,
  dt: float,
  blend: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Meet the next cycle's position and first derivative at the seam."""
  u = np.clip((np.arange(len(position)) - (len(position) - 1 - blend)) / blend, 0, 1)
  u = u[:, None]
  span = blend * dt
  dp = endpoint - position[-1]
  dv = velocity[0] - velocity[-1]
  position_delta = (3 * u**2 - 2 * u**3) * dp + span * (u**3 - u**2) * dv
  velocity_delta = (6 * u - 6 * u**2) / span * dp + (3 * u**2 - 2 * u) * dv
  return position + position_delta, velocity + velocity_delta


def repeat_cycle(
  source: Path, start: int, end: int, seconds: float, speed_scale: float = 1.0
) -> dict[str, np.ndarray | float | int | str]:
  with np.load(source, allow_pickle=False) as data:
    dt = float(data["dt"])
    if not 0 <= start < end < len(data["joint_pos"]) or not 35 <= end - start <= 140:
      raise ValueError("Select a full walking cycle from 35 to 140 control steps.")
    span = slice(start, end + 1)
    contact = data["foot_contact"][span]
    q, qd, p, v, quat = (
      data[key][span].astype(np.float64)
      for key in (
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_lin_vel_w",
        "root_quat_w",
      )
    )
    names = data["joint_names"]
  if not np.array_equal(contact[0], contact[-1]) or not contact[0].any():
    raise ValueError("Start and end must have the same nonempty support set.")
  q_gap = np.sqrt(np.mean((q[-1] - q[0]) ** 2))
  if q_gap > 0.18 or np.max(abs(q[-1] - q[0])) > 0.70:
    raise ValueError("The candidate endpoint pose change exceeds the seam limit.")
  rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
  endpoint_error = (rotation[0] * rotation[-1].inv()).as_rotvec()
  if np.linalg.norm(endpoint_error) > 0.24:
    raise ValueError("The candidate turns too far to be used as straight reference.")

  period = end - start
  blend = min(round(0.20 / dt), period // 3)
  q, qd = close_sequence(q, qd, q[0], dt, blend)
  travel = p[-1] - p[0]
  travel[2] = 0
  if np.linalg.norm(travel) < 0.1:
    raise ValueError("Expected a forward walking stride.")
  p, v = close_sequence(p, v, p[0] + travel, dt, blend)
  p[:, :2] -= p[0, :2].copy()

  # Close small orientation differences over an entire cycle. Keeping the
  # intra-cycle modulation avoids flattening the torso's gait phase motion.
  u = np.linspace(0, 1, period + 1)
  ease = u**3 * (10 - 15 * u + 6 * u**2)
  rotation = Rotation.from_rotvec(ease[:, None] * endpoint_error) * rotation
  forward = rotation.as_matrix()[:-1, :, 0].mean(axis=0)
  ref_yaw = np.arctan2(forward[1], forward[0])
  path_yaw = np.arctan2(travel[1], travel[0])
  rotation = Rotation.from_rotvec([0, 0, path_yaw - ref_yaw]) * rotation

  if speed_scale != 1:
    # Round the period on the control clock before deriving all velocity fields.
    scaled_period = round(period / speed_scale)
    source_time = np.arange(period + 1) * dt
    sample_time = np.linspace(0, period * dt, scaled_period + 1)
    rate = period / scaled_period
    joints = CubicHermiteSpline(source_time, q, qd)
    root = CubicHermiteSpline(source_time, p, v)
    q, qd = joints(sample_time), rate * joints.derivative()(sample_time)
    p, v = root(sample_time), rate * root.derivative()(sample_time)
    rotation = Slerp(source_time, rotation)(sample_time)
    period = scaled_period

  frame = np.arange(round(seconds / dt))
  index = frame % period
  position = p[index] + (frame // period)[:, None] * travel
  root_rotation = rotation[index]
  angular_velocity = np.zeros_like(position)
  increments = (root_rotation[1:] * root_rotation[:-1].inv()).as_rotvec() / dt
  angular_velocity[1:-1] = (increments[1:] + increments[:-1]) / 2
  angular_velocity[0], angular_velocity[-1] = increments[0], increments[-1]
  print(
    f"{source.stem}: {period} steps/stride, raw seam {q_gap:.3f} rad RMS, "
    f"{np.linalg.norm(travel) / (period * dt):.3f} m/s nominal"
  )
  return {
    "joint_names": names,
    "joint_pos": q[index].astype(np.float32),
    "joint_vel": qd[index].astype(np.float32),
    "root_pos_w": position.astype(np.float32),
    "root_quat_w": root_rotation.as_quat()[:, [3, 0, 1, 2]].astype(np.float32),
    "root_lin_vel_w": v[index].astype(np.float32),
    "root_ang_vel_w": angular_velocity.astype(np.float32),
    "period_steps": period,
    "source": str(source),
    "source_cycle": np.array([start, end]),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--start", required=True, type=int)
  parser.add_argument(
    "--end", required=True, type=int, help="Inclusive endpoint of one cycle."
  )
  parser.add_argument("--seconds", type=float, default=9)
  parser.add_argument("--speed-scale", type=float, default=1.0)
  args = parser.parse_args()
  if not np.isfinite(args.seconds) or args.seconds < 3:
    parser.error("Repeat at least three seconds for physical replay.")
  if not np.isfinite(args.speed_scale) or not 0.85 <= args.speed_scale <= 1.15:
    parser.error("Choose a local cycle speed scale in [0.85, 1.15].")
  data = repeat_cycle(args.source, args.start, args.end, args.seconds, args.speed_scale)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("xb") as stream:
    np.savez_compressed(stream, **data)


if __name__ == "__main__":
  main()
