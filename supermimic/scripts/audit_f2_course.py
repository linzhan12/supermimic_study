"""Measure ground-plane drift over complete gait cycles in F2 command replays."""

import argparse
import json
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from supermimic.tasks.velocity_tracking import gait_cycle_hz


def course_metrics(rows: list[dict], dt: float) -> list[dict]:
  results = []
  for phase, group in groupby(rows, key=lambda row: row["phase"]):
    segment = list(group)
    target = segment[0]["target_velocity"]
    if target[0] <= 0 or target[1:] != [0.0, 0.0]:
      continue
    if any(row["target_velocity"] != target for row in segment):
      raise ValueError(f"{phase}: expected a constant command over the segment.")
    period = 1 / gait_cycle_hz(torch.tensor(target[0], dtype=torch.float64)).item()
    timestamps = np.arange(len(segment)) * dt
    previous_cycle_time = timestamps - period
    complete = previous_cycle_time >= 0
    if not complete.any():
      continue

    # Freeze the coordinate axes at the segment's initial pelvis heading.
    w, x, y, z = segment[0]["pelvis_quaternion_wxyz"]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pelvis = np.array([row["pelvis_position_w_m"][:2] for row in segment])
    side = (pelvis - pelvis[0]) @ np.array([-np.sin(yaw), np.cos(yaw)])
    previous_side = np.interp(previous_cycle_time[complete], timestamps, side)
    cycle_drift = (side[complete] - previous_side) / period
    half_second = segment[-max(1, round(0.5 / dt)) :]
    results.append({
      "phase": phase,
      "command_mps": target[0],
      "cycle_s": period,
      "last_half_second_body_lateral_mps": float(np.mean([
        row["actual_velocity"][1] for row in half_second
      ])),
      "whole_segment_lateral_displacement_m": float(side[-1]),
      "whole_segment_lateral_range_m": float(np.ptp(side)),
      "last_cycle_ground_lateral_mps": float(cycle_drift[-1]),
      "max_abs_cycle_ground_lateral_mps": float(np.abs(cycle_drift).max()),
    })
  return results


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("runs", type=Path, nargs="+")
  args = parser.parse_args()
  output = {}
  for run in args.runs:
    dt = json.loads((run / "summary.json").read_text())["control_dt_s"]
    with (run / "multispeed_profile_scripted_velocity.jsonl").open() as stream:
      rows = [json.loads(line) for line in stream]
    output[str(run)] = course_metrics(rows, dt)
  print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
