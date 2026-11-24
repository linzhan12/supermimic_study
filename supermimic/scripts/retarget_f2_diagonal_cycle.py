"""Build diagonal full-body references by moving each foot during its swing."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from scipy.spatial.transform import Rotation

from supermimic.stance_bridge import StanceBridge


def swing_interval(height: np.ndarray) -> tuple[int, int]:
  airborne = height > height.min() + 0.012
  starts = np.flatnonzero(airborne & ~np.roll(airborne, 1))
  period = len(height)
  runs = []
  for start in starts:
    indices = (start + np.arange(period)) % period
    duration = int(np.flatnonzero(~airborne[indices])[0])
    runs.append((duration, int(start)))
  if not runs:
    raise ValueError("Each foot must have at least one complete nominal swing.")
  duration, start = max(runs)
  return (start + duration) % period, period - duration


def retarget(source: Path, lateral_speed: float, pelvis_drop: float) -> dict:
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  names = tuple(model.joint(i).name for i in range(1, model.njnt))
  with np.load(source, allow_pickle=False) as data:
    archive = dict(data)
  if tuple(archive["joint_names"].tolist()) != names:
    raise ValueError("The source must use the MuJoCo G1 joint order.")
  period = int(archive["period_steps"])
  dt = float(archive["dt"])
  q = archive["joint_pos"].astype(float)
  if not 1 < period < len(q) or not np.isclose(dt, 0.02):
    raise ValueError("Use a repeated 50 Hz G1 walking reference.")
  p = archive["root_pos_w"].astype(float)
  quat = archive["root_quat_w"].astype(float)
  rotations = Rotation.from_quat(quat[:period, [1, 2, 3, 0]])
  left = rotations.apply((0, 1, 0)).mean(0)
  left[2] = 0.0
  left /= np.linalg.norm(left)
  feet = archive["foot_position_w_m"].astype(float)
  side_shift = lateral_speed * dt * np.arange(len(q))
  goal_root = p + side_shift[:, None] * left
  goal_root[:, 2] -= pelvis_drop
  relative = np.zeros((period, 2))
  for side in range(2):
    touchdown, stance = swing_interval(feet[:period, side, 2])
    phase = (np.arange(period) - touchdown) % period
    swing = np.clip((phase - stance) / (period - stance), 0.0, 1.0)
    blend = swing**3 * (10 - 15 * swing + 6 * swing**2)
    # Each anchor is constant until toe-off and advances by one full-cycle step.
    relative[:, side] = lateral_speed * dt * (-phase + 0.5 * stance + period * blend)
  if (
    np.min(
      (feet[:period, 0] - feet[:period, 1]) @ left + relative[:, 0] - relative[:, 1]
    )
    < 0.08
  ):
    raise ValueError(
      "The projected footsteps would cross or be less than 0.08 m apart."
    )

  sites = torch.tensor([model.site(f"{side}_foot").id for side in ("left", "right")])
  legs = tuple(
    i
    for i, name in enumerate(names)
    if any(k in name for k in ("hip", "knee", "ankle"))
  )
  bridge = StanceBridge(
    model,
    torch.arange(7),
    torch.as_tensor(model.jnt_qposadr[1:].copy()),
    legs,
    sites,
    torch.as_tensor(q[0]),
    dt,
    0.4,
  )
  delta = np.zeros_like(q[:period])
  for i in range(period):
    joints = q[i].copy()
    bridge._update(joints, p[i], quat[i])
    orientation = bridge.data.site_xmat[sites.numpy()].reshape(2, 3, 3).copy()
    targets = feet[i] + (relative[i] + side_shift[i])[:, None] * left
    if not bridge._project(joints, goal_root[i], quat[i], targets, orientation):
      raise ValueError(f"Foot projection did not converge at source frame {i}.")
    delta[i] = joints - q[i]
  if np.max(np.abs(delta)) > 0.65:
    raise ValueError("The leg adjustment exceeds the tested reference range.")
  phase = np.arange(len(q)) % period
  correction_rate = (np.roll(delta, -1, axis=0) - np.roll(delta, 1, axis=0)) / (2 * dt)
  return {
    "joint_names": names,
    "joint_pos": (q + delta[phase]).astype(np.float32),
    "joint_vel": (archive["joint_vel"] + correction_rate[phase]).astype(np.float32),
    "root_pos_w": goal_root.astype(np.float32),
    "root_quat_w": archive["root_quat_w"],
    "root_lin_vel_w": (archive["root_lin_vel_w"] + lateral_speed * left).astype(
      np.float32
    ),
    "root_ang_vel_w": archive["root_ang_vel_w"],
    "period_steps": period,
    "source": str(source),
    "lateral_reference_speed": lateral_speed,
    "pelvis_drop_m": pelvis_drop,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--lateral-speed", type=float, required=True)
  parser.add_argument("--pelvis-drop", type=float, default=0.0)
  args = parser.parse_args()
  if not 0.0 < abs(args.lateral_speed) <= 0.25:
    parser.error("Choose a finite nonzero side speed of at most 0.25 m/s.")
  if not 0.0 <= args.pelvis_drop <= 0.03:
    parser.error("Choose a pelvis drop between zero and 0.03 m.")
  motion = retarget(args.source, args.lateral_speed, args.pelvis_drop)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("xb") as stream:
    np.savez_compressed(stream, **motion)


if __name__ == "__main__":
  main()
