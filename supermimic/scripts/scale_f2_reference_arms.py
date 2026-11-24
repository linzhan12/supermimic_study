"""Make an isolated reference bank with reduced shoulder and elbow excursion."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--scale", type=float, required=True)
  args = parser.parse_args()
  if not 0.0 <= args.scale <= 1.0:
    parser.error("Arm excursion scale must be within [0, 1].")
  paths = sorted(args.source.glob("*.npz"))
  if not paths:
    parser.error(f"No reference clips in {args.source}.")

  args.output.mkdir(parents=True, exist_ok=False)
  for path in paths:
    output = args.output / path.name
    with np.load(path, allow_pickle=False) as source:
      if source["mode"].item() != "walk" or "period_steps" not in source:
        copyfile(path, output)
        continue
      arrays = dict(source)
    period = int(arrays["period_steps"])
    if not 0 < period <= len(arrays["joint_pos"]):
      raise ValueError(f"{path}: an invalid period for scaling arm excursion.")
    joints = [
      index
      for index, name in enumerate(arrays["joint_names"])
      if "_shoulder_" in name or name.endswith("_elbow_joint")
    ]
    if len(joints) != 8:
      raise ValueError(f"{path}: expected six G1 shoulder joints and two elbow joints.")
    angles = arrays["joint_pos"]
    center = angles[:period, joints].mean(axis=0)
    angles[:, joints] = center + args.scale * (angles[:, joints] - center)
    arrays["joint_vel"][:, joints] *= args.scale
    arrays["arm_excursion_scale"] = np.asarray(args.scale)
    with output.open("xb") as stream:
      np.savez_compressed(stream, **arrays)


if __name__ == "__main__":
  main()
