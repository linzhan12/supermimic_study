"""Convert a recorded simulated transition into a tracking reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def extract(trace: Path, start: int, end: int, output: Path) -> None:
  summary = json.loads(trace.with_name("summary.json").read_text())
  rows = [json.loads(line) for line in trace.read_text().splitlines()]
  if start < 0 or end > len(rows) or end - start < 46:
    raise ValueError("Select at least 46 recorded rows within one episode.")
  samples = rows[start:end]
  dt = float(summary["control_dt_s"])
  stamps = np.asarray([row["state_time_s"] for row in samples])
  if not np.allclose(np.diff(stamps), dt) or any(
    row["termination_reasons"] for row in samples
  ):
    raise ValueError("Select a contiguous transition without a termination.")
  names = summary["joint_position_order"]
  q = np.asarray([row["posture"]["joint_pos_rad"] for row in samples])
  p = np.asarray([row["pelvis_position_w_m"] for row in samples])
  quat = np.asarray([row["pelvis_quaternion_wxyz"] for row in samples])
  rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
  increments = (rot[1:] * rot[:-1].inv()).as_rotvec()
  angular = np.empty_like(p)
  angular[0] = increments[0] / dt
  angular[-1] = increments[-1] / dt
  angular[1:-1] = (increments[:-1] + increments[1:]) / (2 * dt)
  values = {
    "joint_pos": q,
    "joint_vel": np.gradient(q, dt, axis=0),
    "root_pos_w": p,
    "root_quat_w": quat,
    "root_lin_vel_w": np.gradient(p, dt, axis=0),
    "root_ang_vel_w": angular,
  }
  if q.shape != (len(samples), len(names)) or not all(
    np.isfinite(v).all() for v in values.values()
  ):
    raise ValueError("Invalid recorded whole-body state.")
  with trace.open("rb") as stream:
    source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("xb") as stream:
    np.savez_compressed(
      stream,
      **{key: value.astype(np.float32) for key, value in values.items()},
      joint_names=names,
      trace_sha256=source_hash,
      actor_sha256=summary["actor_sha256"],
      trace_rows=np.asarray([start, end]),
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("trace", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("start", type=int)
  parser.add_argument("end", type=int)
  args = parser.parse_args()
  extract(args.trace, args.start, args.end, args.output)


if __name__ == "__main__":
  main()
