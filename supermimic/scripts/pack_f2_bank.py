"""Package windows of checked F2 references for the online planner."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from audit_forward_gait import G1_XML, foot_geometry


def pack(
  model: mujoco.MjModel, output: Path, trace: Path, mode: str, first: int, stop: int
) -> None:
  summary = json.loads(trace.with_name("summary.json").read_text())
  frames = [json.loads(row) for row in trace.read_text().splitlines()]
  if not 0 <= first < stop <= len(frames) or stop - first < 46:
    raise ValueError(f"{trace}: supply at least 46 previously evaluated frames.")
  rows = frames[first:stop]
  if any(row["termination_reasons"] for row in rows):
    raise ValueError(f"{trace}: a selected frame terminated.")
  indices = np.asarray([row["reference_frame"] for row in rows])
  if not np.all(np.diff(indices) == 1):
    raise ValueError(f"{trace}: expected a contiguous fixed reference.")
  if mode not in ("stand", "start", "walk", "stop"):
    raise ValueError(f"Unknown reference mode {mode!r}.")
  names = summary["joint_position_order"]
  dt = summary["control_dt_s"]
  with np.load(summary["motion_file"], allow_pickle=False) as source:
    if names != source["joint_names"].tolist():
      raise ValueError(f"{trace}: reference joint order differs from the robot.")
    arrays = {
      name: source[name][indices]
      for name in (
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
      )
    }
  reference = [
    {
      "pelvis_position_w_m": p,
      "pelvis_quaternion_wxyz": quat,
      "posture": {"joint_pos_rad": q},
    }
    for p, quat, q in zip(
      arrays["root_pos_w"], arrays["root_quat_w"], arrays["joint_pos"], strict=True
    )
  ]
  feet, soles, _ = foot_geometry(model, reference, names)
  speed = np.linalg.norm(np.gradient(feet[:, :, :2], dt, axis=0), axis=-1)
  floor = np.clip(np.quantile(soles, 0.10), -0.05, 0.05)
  contact = (soles <= floor + 0.018) & (speed <= 0.26)
  actual = np.asarray([row["actual_velocity"] for row in rows])
  expected = np.asarray([row["reference_velocity_unclipped"] for row in rows])
  bias = (actual - expected).mean(0) if mode == "walk" else np.zeros(3)
  with output.open("xb") as stream:
    np.savez_compressed(
      stream,
      **arrays,
      joint_names=names,
      dt=dt,
      mode=mode,
      foot_position_w_m=feet,
      foot_contact=contact,
      mean_tracking_bias=np.clip(bias, (-0.12, -0.10, -0.16), (0.12, 0.10, 0.16)),
      source=summary["motion_file"],
      source_sha256=summary["motion_sha256"],
      actor_sha256=summary["actor_sha256"],
      source_frames=indices,
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--entry",
    nargs=5,
    metavar=("NAME", "MODE", "TRACE", "START_ROW", "END_ROW"),
    required=True,
    action="append",
  )
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  for name, mode, path, first, stop in args.entry:
    if Path(name).name != name:
      parser.error("Clip names must use plain filenames.")
    output = args.output / (name + ".npz")
    pack(model, output, Path(path), mode, int(first), int(stop))
    print(output)


if __name__ == "__main__":
  main()
