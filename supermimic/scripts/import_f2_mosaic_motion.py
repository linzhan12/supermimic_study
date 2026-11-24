"""Convert a MOSAIC G1 motion into the local MuJoCo reference layout."""

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from scipy.optimize import linear_sum_assignment

# IsaacLab column indices in G1 MuJoCo joint order.
# See NVlabs/GR00T-WholeBodyControl gear_sonic/envs/env_utils/joint_utils.py.
ISAAC_TO_MUJOCO = (
  0,
  3,
  6,
  9,
  13,
  17,
  1,
  4,
  7,
  10,
  14,
  18,
  2,
  5,
  8,
  11,
  15,
  19,
  21,
  23,
  25,
  27,
  12,
  16,
  20,
  22,
  24,
  26,
  28,
)


def convert(source: Path, destination: Path, model: mujoco.MjModel) -> float:
  with np.load(source, allow_pickle=False) as raw:
    if float(raw["fps"].item()) != 50:
      raise ValueError(f"{source}: only the 50 Hz MOSAIC export is supported.")
    q, qd = (raw[key] for key in ("joint_pos", "joint_vel"))
    body = raw["body_pos_w"]
    root_quat = raw["body_quat_w"][:, 0]
    count = len(q)
    if (
      count < 46
      or q.shape != (count, 29)
      or qd.shape != q.shape
      or body.shape != (count, model.nbody - 1, 3)
      or root_quat.shape != (count, 4)
    ):
      raise ValueError(f"{source}: unexpected G1 reference dimensions.")
    arrays = {
      "joint_pos": q[:, ISAAC_TO_MUJOCO],
      "joint_vel": qd[:, ISAAC_TO_MUJOCO],
      "root_pos_w": body[:, 0],
      "root_quat_w": root_quat,
      "root_lin_vel_w": raw["body_lin_vel_w"][:, 0],
      "root_ang_vel_w": raw["body_ang_vel_w"][:, 0],
    }
    if not all(np.isfinite(a).all() for a in arrays.values()):
      raise ValueError(f"{source}: non-finite motion values.")
    if not np.allclose(np.linalg.norm(root_quat, axis=-1), 1.0, atol=1e-3):
      raise ValueError(f"{source}: root quaternions are not normalized.")
    # Match the source's unnamed body positions to our MuJoCo bodies consistently
    # across several poses. Incorrect joint or quaternion order fails this test.
    data = mujoco.MjData(model)
    error = np.zeros((model.nbody - 1, model.nbody - 1))
    stamps = np.linspace(0, count - 1, 8, dtype=int)
    for frame in stamps:
      data.qpos[:3] = body[frame, 0]
      data.qpos[3:7] = root_quat[frame]
      data.qpos[7:] = arrays["joint_pos"][frame]
      mujoco.mj_kinematics(model, data)
      error += ((body[frame, :, None] - data.xpos[None, 1:]) ** 2).sum(-1)
    order = linear_sum_assignment(error)
    rms = float(np.sqrt(error[order].mean() / len(stamps)))
    if rms > 0.005:
      raise ValueError(
        f"{source}: MuJoCo body-position agreement failed ({rms:.4f} m)."
      )

  with destination.open("xb") as stream:
    np.savez_compressed(
      stream,
      **{name: value.astype(np.float32) for name, value in arrays.items()},
      joint_names=[model.joint(i).name for i in range(1, model.njnt)],
      source=source.name,
      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
  return rms


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("motions", nargs="+", type=Path)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  for motion in args.motions:
    rms = convert(motion, args.output / motion.name, model)
    print(f"{motion.name}: MuJoCo body-position RMS {rms:.3g} m")


if __name__ == "__main__":
  main()
