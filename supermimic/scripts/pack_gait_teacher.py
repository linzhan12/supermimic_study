"""Save physical full-reference rollouts as SuperMimic training demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_motion(path: Path) -> dict[str, np.ndarray]:
  summary = json.loads(path.with_name("summary.json").read_text())
  case = path.stem.removesuffix("_motion")
  if (
    not path.stem.endswith("_motion")
    or not summary["cases"][case]["routes"]["motion"]["completed"]
  ):
    raise ValueError(f"{path}: expected a completed full-reference rollout.")
  with path.open() as stream:
    rows = [json.loads(line) for line in stream]
  dt = summary["control_dt_s"]
  if not np.isclose(dt, 1 / 50) or len(rows) < 3:
    raise ValueError(f"{path}: expected at least three samples at 50 Hz.")
  q = np.asarray([r["posture"]["joint_pos_rad"] for r in rows])
  pos = np.asarray([r["pelvis_position_w_m"] for r in rows])
  quat = np.asarray([r["pelvis_quaternion_wxyz"] for r in rows])
  rotation = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
  w, x, y, z = quat.T
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  forward, left, yaw_rate = np.asarray([r["actual_velocity"] for r in rows]).T
  linear = np.stack(
    (
      np.cos(yaw) * forward - np.sin(yaw) * left,
      np.sin(yaw) * forward + np.cos(yaw) * left,
      np.gradient(pos[:, 2], dt),
    ),
    axis=-1,
  )
  steps = np.arange(len(rows))
  before = np.maximum(steps - 1, 0)
  after = np.minimum(steps + 1, len(rows) - 1)
  angular = (rotation[after] * rotation[before].inv()).as_rotvec()
  angular /= ((after - before) * dt)[:, None]
  angular[:, 2] = yaw_rate
  contact_order = summary["foot_contact_order"]
  left_right = [
    contact_order.index(f"{side}_ankle_roll_link") for side in ("left", "right")
  ]
  fields = {
    "joint_pos": q,
    "joint_vel": np.gradient(q, dt, axis=0),
    "root_pos_w": pos,
    "root_quat_w": quat,
    "root_lin_vel_w": linear,
    "root_ang_vel_w": angular,
  }
  return {
    "joint_names": np.asarray(summary["joint_position_order"]),
    "action_names": np.asarray(summary["action_order"]),
    "executed_action": np.asarray([r["executed_action"] for r in rows], dtype=np.float32),
    "foot_contact_lr": np.asarray([r["foot_contact"] for r in rows], dtype=bool)[
      :, left_right
    ],
    **{key: value.astype(np.float32) for key, value in fields.items()},
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument("traces", type=Path, nargs="+")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  for path in args.traces:
    motion = load_motion(path)
    output = args.output / f"{path.parent.name}_{path.stem}.npz"
    with output.open("xb") as stream:
      np.savez_compressed(stream, **motion)
    print(f"{output}: {len(motion['joint_pos'])} samples")


if __name__ == "__main__":
  main()
