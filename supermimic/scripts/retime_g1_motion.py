"""Retime a local 50 Hz SuperMimic clip for physical reference validation."""

import argparse
from pathlib import Path

import numpy as np

from convert_amp_g1_motion import resample_motion


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--speed-scale", type=float, required=True)
  parser.add_argument("--start-frame", type=int, default=0)
  parser.add_argument("--end-frame", type=int)
  args = parser.parse_args()
  if not np.isfinite(args.speed_scale) or args.speed_scale <= 0:
    parser.error("--speed-scale must be finite and positive.")
  with np.load(args.input, allow_pickle=False) as source:
    end = len(source["joint_pos"]) if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame <= end - 3 or end > len(source["joint_pos"]):
      parser.error("Set a valid zero-based half-open interval with at least three frames.")
    frames = slice(args.start_frame, end)
    time = np.arange(end - args.start_frame) / (50 * args.speed_scale)
    converted = resample_motion(
      args.input, time, source["root_pos_w"][frames], source["root_quat_w"][frames],
      source["joint_pos"][frames], source["joint_names"].tolist(),
    )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("xb") as stream:
    np.savez_compressed(stream, **converted)
  print(f"{args.output}: {len(converted['joint_pos'])} frames at 50 Hz")


if __name__ == "__main__":
  main()
