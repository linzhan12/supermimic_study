"""Screen F2 closed-loop multi-speed gait before the paired-video review."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import mujoco
from audit_forward_gait import metrics
from check_f2 import check_f2, load_scripted
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

STEADY = (
  (0.22, "steady_forward_0p220"),
  (0.35, "steady_forward"),
  (0.50, "steady_forward_0p500"),
)


def check_gait(candidate: Path) -> tuple[list[str], dict]:
  summary = json.loads((candidate / "summary.json").read_text())
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  dt = summary["control_dt_s"]
  window = round(4.0 / dt)
  failures = []
  results = {}
  measured = []
  for target, case in STEADY:
    route = (
      summary["cases"].get(case, {}).get("routes", {}).get("scripted_velocity", {})
    )
    if not route.get("completed"):
      failures.append(f"{case} velocity replay incomplete")
      continue
    rows = load_scripted(candidate, case)[-window:]
    if len(rows) != window or not all(
      abs(row["target_velocity"][0] - target) < 1e-6
      and row["target_velocity"][1:] == [0.0, 0.0]
      for row in rows
    ):
      failures.append(
        f"{case} does not contain four steady seconds at {target:.2f} m/s"
      )
      continue
    stats = metrics(model, rows, summary["joint_position_order"], dt)
    results[case] = stats
    vx, vy, yaw = stats["mean_heading_velocity"]
    measured.append(vx)
    if abs(vx - target) > 0.06 or abs(vy) > 0.06 or abs(yaw) > 0.10:
      failures.append(f"{case} mean forward, lateral or yaw velocity outside tolerance")
    mae = mean(abs(row["actual_velocity"][0] - target) for row in rows)
    if mae > 0.10:
      failures.append(f"{case} forward MAE {mae:.3f} m/s above 0.10")
    if min(stats["excursion_p95_minus_p05_deg"]["knee"]) < 18:
      failures.append(f"{case} one or more knees move less than 18 degrees")
    feet = stats["feet"]
    counts = [foot["complete_swings"] for foot in feet.values()]
    if min(counts) < 2 or min(counts) < 0.6 * max(counts):
      failures.append(f"{case} bilateral swing counts too low or imbalanced: {counts}")
    if len(stats["alternating_touchdown_interval_s"]) < sum(counts) - 2:
      failures.append(f"{case} repeated touchdown on the same side")
    for side, foot in feet.items():
      if not foot["complete_swings"]:
        continue
      gap = median(foot["step_length_at_touchdown_m"])
      clearance = median(foot["peak_sole_clearance_m"])
      if gap < 0.26 * target:
        failures.append(
          f"{case}/{side} median forward landing gap {gap:.3f} m too short"
        )
      if clearance < 0.008:
        failures.append(
          f"{case}/{side} median swing clearance {clearance:.3f} m below 0.008"
        )
  if (
    len(measured) == len(STEADY)
    and min(b - a for a, b in zip(measured[:-1], measured[1:], strict=True)) < 0.07
  ):
    failures.append("steady forward levels differ by less than 0.07 m/s")

  profile = (
    summary["cases"]
    .get("multispeed_profile", {})
    .get("routes", {})
    .get("scripted_velocity", {})
  )
  if not profile.get("completed"):
    failures.append("multispeed profile did not complete")
  else:
    rows = load_scripted(candidate, "multispeed_profile")
    profile_results = {}
    for phase, target in (
      ("slow", 0.22),
      ("fast", 0.50),
      ("medium", 0.35),
      ("release", 0.0),
    ):
      tail = [row["actual_velocity"] for row in rows if row["phase"] == phase][
        -round(0.5 / dt) :
      ]
      vx, vy, yaw = map(mean, zip(*tail, strict=True))
      profile_results[phase] = (vx, vy, yaw)
      if abs(vx - target) > 0.10 or abs(vy) > 0.10 or abs(yaw) > 0.18:
        failures.append(f"multispeed {phase} does not settle to the commanded velocity")
    results["multispeed_profile"] = profile_results
  return failures, results


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("baseline", type=Path)
  parser.add_argument("candidate", type=Path)
  parser.add_argument("--output-json", type=Path)
  args = parser.parse_args()
  gait_failures, results = check_gait(args.candidate)
  failures = check_f2(args.baseline, args.candidate) + gait_failures
  if args.output_json:
    args.output_json.write_text(
      json.dumps({"failures": failures, "metrics": results}, indent=2, allow_nan=False)
      + "\n"
    )
  if failures:
    print("F2 screening failed")
    for failure in failures:
      print(f"- {failure}")
    raise SystemExit(1)
  print(
    "F2 numeric screening passed. Review paired full-body reference and policy videos."
  )


if __name__ == "__main__":
  main()
