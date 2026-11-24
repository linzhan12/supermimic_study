"""Resample public HOLD G1 HDF5 trajectories into 50 Hz SuperMimic references."""

import argparse
import hashlib
from pathlib import Path

import h5py
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from scipy.spatial.transform import Rotation, Slerp


def convert(source: Path, destination: Path, names: tuple[str, ...]) -> int:
  with h5py.File(source, "r") as raw:
    order = raw["robot_data/meta_data/quat_order"].asstr()[:].tolist()
    motion = raw["robot_data/trajectory_data"]
    root_names = motion["bodies/labels"].asstr()[:].tolist()
    joint_names = motion["joints/labels"].asstr()[:].tolist()
    if (
      root_names != ["base"]
      or sorted(order) != ["w", "x", "y", "z"]
      or sorted(joint_names) != sorted(names)
    ):
      raise ValueError(f"{source}: expected a HOLD G1 root and all 29 named joints.")
    joints = [joint_names.index(name) for name in names]
    xyzw = [order.index(axis) for axis in ("x", "y", "z", "w")]
    time = motion["time"][:]
    time -= time[0]
    pos, quat, linear = (
      motion[f"bodies/{field}"][:, 0] for field in ("pos_w", "quat_w", "lin_vel_w")
    )
    q, qd = (motion[f"joints/{field}"][:][:, joints] for field in ("pos", "vel"))
  if (
    len(time) < 2
    or np.any(np.diff(time) <= 0)
    or not all(np.isfinite(a).all() for a in (time, pos, quat, q, qd, linear))
    or not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-3)
  ):
    raise ValueError(f"{source}: invalid timestamps or motion values.")

  dt = 0.02
  sample_time = np.arange(int(time[-1] / dt) + 1) * dt
  if len(sample_time) < 46:
    raise ValueError(
      f"{source}: the resampled motion is shorter than one actor preview."
    )
  rotations = Slerp(time, Rotation.from_quat(quat[:, xyzw]))(sample_time)
  pose = rotations.as_quat()
  # Differentiate rotations in the world frame at the destination rate.
  angular = np.empty_like(pose[:, :3])
  angular[1:-1] = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2 * dt)
  angular[0] = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
  angular[-1] = (rotations[-1] * rotations[-2].inv()).as_rotvec() / dt
  # Make the stored rotation sign continuous across any Slerp hemisphere changes.
  flip = np.einsum("ij,ij->i", pose[:-1], pose[1:]) < 0
  pose[1:] *= np.cumprod(np.where(flip, -1.0, 1.0))[:, None]
  arrays = {}
  for key, value in (
    ("root_pos_w", pos),
    ("joint_pos", q),
    ("joint_vel", qd),
    ("root_lin_vel_w", linear),
  ):
    arrays[key] = np.stack(
      [np.interp(sample_time, time, column) for column in value.T], axis=1
    ).astype(np.float32)
  with destination.open("xb") as stream:
    np.savez_compressed(
      stream,
      **arrays,
      root_quat_w=pose[:, [3, 0, 1, 2]].astype(np.float32),
      root_ang_vel_w=angular.astype(np.float32),
      joint_names=names,
      source=source.name,
      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
  return len(sample_time)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("motions", nargs="+", type=Path)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  names = tuple(model.joint(i).name for i in range(1, model.njnt))
  for motion in args.motions:
    count = convert(motion, args.output / f"{motion.stem}.npz", names)
    print(f"{motion.name}: {count} frames at 50 Hz")


if __name__ == "__main__":
  main()
