"""Check the paired F2 velocity, stopping-posture and arm-swing criteria."""

import argparse
import json
import math
from filecmp import cmp
from pathlib import Path
from statistics import mean

MOVING_MAE_TOLERANCES = (0.03, 0.03, 0.05)
CASES = ("stand_start", "walk_start")


def load_trace(folder: Path, case: str, route: str) -> list[dict]:
  with (folder / f"{case}_{route}.jsonl").open() as stream:
    return [json.loads(line) for line in stream]


def check_f2(baseline: Path, candidate: Path, route: str = "planned") -> list[str]:
  ref, run = (
    json.loads((p / "summary.json").read_text()) for p in (baseline, candidate)
  )
  for key in ("motion_sha256", "seed", "control_dt_s", "device", "task_id"):
    if ref[key] != run[key]:
      raise ValueError(f"Evaluation settings differ: {key}")

  failures = []
  for case in CASES:
    original, result = (s["cases"][case] for s in (ref, run))
    if original["initial_full_state"] != result["initial_full_state"]:
      raise ValueError(f"Initial states differ: {case}")
    for mode, outcome in result["routes"].items():
      if not outcome["completed"]:
        failures.append(
          f"{case}/{mode} terminated after {outcome['observed_steps']} steps"
        )
    motion = f"{case}_motion.jsonl"
    if not cmp(baseline / motion, candidate / motion, shallow=False):
      failures.append(f"{case}/motion differs from F1")

    trace = load_trace(candidate, case, route)
    original_trace = load_trace(baseline, case, "scripted_velocity")
    commands = [(r["phase"], r["target_velocity"]) for r in trace]
    if commands != [
      (r["phase"], r["target_velocity"]) for r in original_trace[: len(trace)]
    ]:
      raise ValueError(f"Commands differ: {case}")
    old = original["routes"]["scripted_velocity"]["phases"]
    new = result["routes"][route]["phases"]
    for phase in old:
      if phase not in new:
        failures.append(f"{case}/{phase} missing")
        continue
      if phase == "stand" or phase.startswith("release"):
        continue
      for axis, prev, val, limit in zip(
        ("vx", "vy", "yaw"),
        old[phase]["velocity_mae"],
        new[phase]["velocity_mae"],
        MOVING_MAE_TOLERANCES,
        strict=True,
      ):
        if val - prev > limit:
          failures.append(f"{case}/{phase}/{axis} MAE +{val - prev:.5f} > {limit:.2f}")

  stand = load_trace(candidate, "stand_start", route)
  for phase, axis, sign, minimum in (
    ("forward", 0, 1, 0.20),
    ("left_turn", 2, 1, 0.15),
    ("right_turn", 2, -1, 0.15),
  ):
    speeds = [sign * r["actual_velocity"][axis] for r in stand if r["phase"] == phase]
    if not speeds or mean(speeds) < minimum:
      failures.append(f"{phase} absolute speed criterion")

  tail_steps = round(0.5 / run["control_dt_s"])
  release_speeds = [
    math.hypot(*r["actual_velocity"][:2])
    for r in load_trace(candidate, "walk_start", route)
    if r["phase"] == "release"
  ][-tail_steps:]
  if len(release_speeds) < tail_steps or mean(release_speeds) >= 0.10:
    failures.append("release absolute speed criterion")

  old_stop = ref["cases"]["walk_start"]["routes"]["scripted_velocity"]["phases"].get(
    "release", {}
  )
  new_stop = run["cases"]["walk_start"]["routes"][route]["phases"].get("release", {})
  for key in ("torso_tilt_mean_deg", "tail_0_5s_standing_joint_rms_rad"):
    if new_stop.get(key, math.inf) > 0.8 * old_stop[key]:
      failures.append(f"{key} did not decrease by 20%")

  steady = run["cases"]["steady_forward"]["routes"][route]
  if not steady["completed"]:
    return [*failures, "steady forward did not complete"]
  arm = steady["arm_swing"]
  excursions = arm["shoulder_pitch_excursion_deg"]
  if not all(8 <= angle <= 40 for angle in excursions):
    failures.append("one or more shoulder excursions outside 8–40 degrees")
  if min(excursions) < 0.5 * max(excursions):
    failures.append("smaller shoulder excursion below half of the larger one")
  if (
    arm["left_right_correlation"] > -0.25
    or arm["arm_hip_difference_correlation"] > -0.30
  ):
    failures.append("arm coordination correlation criteria")
  if arm["forward_mae_mps"] > 0.10:
    failures.append("steady forward MAE above 0.10 m/s")
  return failures


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("baseline", type=Path)
  parser.add_argument("candidate", type=Path)
  parser.add_argument("--candidate-route", default="planned")
  args = parser.parse_args()
  failures = check_f2(args.baseline, args.candidate, args.candidate_route)
  if failures:
    print("F2 criteria failed")
    for failure in failures:
      print(f"- {failure}")
    raise SystemExit(1)
  print("All paired F2 and steady arm-swing criteria passed.")
