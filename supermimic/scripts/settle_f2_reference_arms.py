"""Blend a stop reference's arms toward a quiet walking posture."""

import argparse
from pathlib import Path

import numpy as np


def settle_arms(
  source: Path, target: Path, delay: float, duration: float
) -> dict[str, np.ndarray]:
  with np.load(source, allow_pickle=False) as stop_data:
    stop = dict(stop_data)
  with np.load(target, allow_pickle=False) as cycle:
    names = stop["joint_names"].tolist()
    if (
      names != cycle["joint_names"].tolist()
      or stop["mode"].item() != "stop"
      or cycle["mode"].item() != "walk"
      or not np.isclose(stop["dt"], cycle["dt"])
    ):
      raise ValueError(
        "Use a stop and a walking cycle with the same joint order and rate."
      )
    arms = [
      i
      for i, name in enumerate(names)
      if "_shoulder_" in name or name.endswith("_elbow_joint")
    ]
    period = int(cycle["period_steps"])
    if len(arms) != 8 or not 1 < period <= len(cycle["joint_pos"]):
      raise ValueError(
        "The target must contain a complete G1 shoulder and elbow cycle."
      )
    neutral = np.median(cycle["joint_pos"][:period, arms], axis=0)
  times = np.arange(len(stop["joint_pos"])) * float(stop["dt"])
  if not (0 <= delay < delay + duration <= times[-1]):
    raise ValueError("The arm transition must fit within the stop reference.")
  s = np.clip((times - delay) / duration, 0.0, 1.0)[:, None]
  blend = s**3 * (10 - 15 * s + 6 * s**2)
  rate = 30 * s**2 * (1 - s) ** 2 / duration
  q = stop["joint_pos"][:, arms]
  delta = neutral - q
  stop["joint_pos"][:, arms] = q + blend * delta
  stop["joint_vel"][:, arms] = (1 - blend) * stop["joint_vel"][:, arms] + rate * delta
  stop["arm_settle_target"] = np.asarray(str(target))
  stop["arm_settle_delay_s"] = np.asarray(delay)
  stop["arm_settle_duration_s"] = np.asarray(duration)
  return stop


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("stop", type=Path)
  parser.add_argument("cycle", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--delay-s", type=float, required=True)
  parser.add_argument("--duration-s", type=float, required=True)
  args = parser.parse_args()
  arrays = settle_arms(args.stop, args.cycle, args.delay_s, args.duration_s)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("xb") as stream:
    np.savez_compressed(stream, **arrays)


if __name__ == "__main__":
  main()
