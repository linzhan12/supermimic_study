"""Convert local HOLD G1 .h5 trajectories to 50 Hz SuperMimic references."""

import argparse
from pathlib import Path

import numpy as np

from convert_amp_g1_motion import g1_joint_names, resample_motion


def convert(source: Path, joint_names: list[str]) -> dict[str, np.ndarray]:
  import h5py

  with h5py.File(source, "r") as recording:
    data = recording["robot_data/trajectory_data"]
    time = data["time"][:]
    joint_labels = data["joints/labels"].asstr()[:].tolist()
    if len(joint_labels) != len(joint_names) or set(joint_labels) != set(joint_names):
      raise ValueError(f"{source}: joint names must match the G1 model.")
    order = [joint_labels.index(name) for name in joint_names]
    joints = data["joints/pos"][:, :][:, order]
    base = data["bodies/labels"].asstr()[:].tolist().index("base")
    pos = data["bodies/pos_w"][:, base]
    quat = data["bodies/quat_w"][:, base]
    axes = recording["robot_data/meta_data/quat_order"].asstr()[:].tolist()
    quat = quat[:, [axes.index(axis) for axis in "wxyz"]]
  if (
    len(time) < 3
    or not (len(time) == len(pos) == len(quat) == len(joints))
    or not np.isfinite(time).all()
    or not (np.diff(time) > 0).all()
  ):
    raise ValueError(f"{source}: invalid motion frame times.")
  return resample_motion(source, time - time[0], pos, quat, joints, joint_names)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument("inputs", type=Path, nargs="+")
  args = parser.parse_args()
  names = g1_joint_names()
  args.output.mkdir(parents=True, exist_ok=True)
  for source in args.inputs:
    output = args.output / f"{source.stem}.npz"
    fields = convert(source, names)
    with output.open("xb") as stream:
      np.savez_compressed(stream, **fields)
    print(f"{output}: {len(fields['joint_pos'])} frames at 50 Hz")


if __name__ == "__main__":
  main()
