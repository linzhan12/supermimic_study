"""Reflect a dense G1 motion to add an independently tested opposite direction."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

REFLECTION = np.array((1, -1, 1))


def reflected_joint_map(names: list[str]) -> tuple[list[int], np.ndarray]:
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  indices, signs = [], []
  for name in names:
    if name.startswith("left_"):
      source = "right_" + name.removeprefix("left_")
    elif name.startswith("right_"):
      source = "left_" + name.removeprefix("right_")
    else:
      source = name
    source_id, target_id = model.joint(source).id, model.joint(name).id
    axial = -REFLECTION * data.xaxis[source_id]
    dot = axial @ data.xaxis[target_id]
    if not np.isclose(abs(dot), 1.0, atol=1e-3):
      raise ValueError(f"{source} and {name} are not reflected hinge axes.")
    indices.append(names.index(source))
    signs.append(1 if dot >= 0 else -1)
  return indices, np.asarray(signs)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("destination", type=Path)
  parser.add_argument("--first", type=int, default=0)
  parser.add_argument("--stop", type=int)
  args = parser.parse_args()
  with np.load(args.source, allow_pickle=False) as data:
    names = data["joint_names"].tolist()
    indices, signs = reflected_joint_map(names)
    stop = len(data["joint_pos"]) if args.stop is None else args.stop
    if not 0 <= args.first < stop <= len(data["joint_pos"]):
      parser.error("Choose a nonempty interval within the source motion.")
    window = slice(args.first, stop)
    position = data["root_pos_w"][window]
    origin = position[0].copy()
    reflected_pos = (position - origin) * REFLECTION + origin
    arrays = {
      "joint_names": names,
      "joint_pos": data["joint_pos"][window][:, indices] * signs,
      "joint_vel": data["joint_vel"][window][:, indices] * signs,
      "root_pos_w": reflected_pos,
      "root_quat_w": data["root_quat_w"][window] * (1, -1, 1, -1),
      "root_lin_vel_w": data["root_lin_vel_w"][window] * REFLECTION,
      "root_ang_vel_w": data["root_ang_vel_w"][window] * -REFLECTION,
    }
  with args.destination.open("xb") as stream:
    np.savez_compressed(
      stream,
      **{
        key: np.asarray(value, dtype=np.float32) if key != "joint_names" else value
        for key, value in arrays.items()
      },
    )


if __name__ == "__main__":
  main()
