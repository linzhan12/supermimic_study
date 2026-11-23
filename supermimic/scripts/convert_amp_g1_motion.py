"""Convert local IsaacLab AMP G1 motion archives to 50 Hz SuperMimic references."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


def convert(source: Path, joint_names: list[str]) -> dict[str, np.ndarray]:
  with np.load(source, allow_pickle=False) as data:
    fps = float(data["fps"].item())
    positions = data["dof_positions"]
    names = data["dof_names"].tolist()
    if len(names) != len(joint_names) or set(names) != set(joint_names):
      raise ValueError(f"{source}: joint names must match the G1 model.")
    positions = positions[:, [names.index(name) for name in joint_names]]
    pelvis = data["body_names"].tolist().index("pelvis")
    root = data["body_positions"][:, pelvis]
    quat = data["body_rotations"][:, pelvis]
  if not np.isfinite(fps) or fps <= 0 or len(root) < 3 or len(positions) != len(root):
    raise ValueError(f"{source}: invalid frame count or sample rate.")
  if not np.isfinite(positions).all() or not np.isfinite(root).all():
    raise ValueError(f"{source}: nonfinite motion position.")
  if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-3):
    raise ValueError(f"{source}: invalid pelvis quaternions.")

  source_time = np.arange(len(root)) / fps
  time = np.arange(int(np.floor(source_time[-1] * 50 + 1e-8)) + 1) / 50
  if len(time) < 3:
    raise ValueError(f"{source}: less than three output frames.")
  root_spline = CubicSpline(source_time, root)
  joint_spline = CubicSpline(source_time, positions)
  rotation = Slerp(source_time, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(time)
  before = np.maximum(np.arange(len(time)) - 1, 0)
  after = np.minimum(np.arange(len(time)) + 1, len(time) - 1)
  # R_after @ inv(R_before) produces an angular displacement in world coordinates.
  angular = (rotation[after] * rotation[before].inv()).as_rotvec()
  angular /= (time[after] - time[before])[:, None]
  fields = {
    "joint_pos": joint_spline(time),
    "joint_vel": joint_spline(time, 1),
    "root_pos_w": root_spline(time),
    "root_quat_w": rotation.as_quat()[:, [3, 0, 1, 2]],
    "root_lin_vel_w": root_spline(time, 1),
    "root_ang_vel_w": angular,
  }
  return {
    "joint_names": np.asarray(joint_names),
    **{k: v.astype(np.float32) for k, v in fields.items()},
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument("input", type=Path, nargs="+")
  args = parser.parse_args()
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  names = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
  ]
  args.output.mkdir(parents=True, exist_ok=True)
  for source in args.input:
    output = args.output / source.name
    fields = convert(source, names)
    with output.open("xb") as stream:
      np.savez_compressed(stream, **fields)
    print(f"{output}: {len(fields['joint_pos'])} frames at 50 Hz")


if __name__ == "__main__":
  main()
