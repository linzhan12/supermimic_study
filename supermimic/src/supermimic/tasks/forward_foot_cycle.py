"""Phase-specific flat-ground foot placement and clearance on the G1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers import RewardTermCfg
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from supermimic.tasks.velocity_tracking import (
  IndependentVelocityCommand,
  advance_gait_phase,
  forward_gait_gate,
  gait_cycle_hz,
  gait_phase_features,
  physical_root,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ForwardFootCycleReward:
  """Score alternating swing clearance and step geometry at the commanded cadence."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    robot = env.scene["robot"]
    model = env.sim.mj_model
    sides = ("left", "right")
    rows = [robot.body_names.index(f"{side}_ankle_roll_link") for side in sides]
    self.fk = ForwardKinematics(
      model,
      robot.indexing.body_ids,
      robot.indexing.joint_q_adr,
      torch.tensor(rows),
      device=env.device,
    )
    geoms = [
      [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and model.geom(g).name.rsplit("/", 1)[-1].startswith(f"{side}_foot")
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
      ]
      for side, body in zip(sides, robot.indexing.body_ids[rows].tolist(), strict=True)
    ]
    if not geoms[0] or len(geoms[0]) != len(geoms[1]):
      raise ValueError("Expected matching collision capsules on both G1 feet.")
    like = robot.data.default_joint_pos
    centers = like.new_tensor(model.geom_pos[geoms])
    local_rot = like.new_tensor(model.geom_quat[geoms])
    sizes = like.new_tensor(model.geom_size[geoms])
    axis = torch.zeros_like(centers)
    axis[..., 2] = sizes[..., 1]
    axis = quat_apply(local_rot, axis)
    self.capsule_ends = torch.cat((centers + axis, centers - axis), dim=1)
    self.capsule_radii = sizes[..., 0].repeat(1, 2)
    site_rows, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    self.site_offsets = like.new_tensor(
      model.site_pos[robot.indexing.site_ids[site_rows].cpu()]
    )
    command_term = env.command_manager.get_term("velocity")
    assert (
      isinstance(command_term, IndependentVelocityCommand)
      and command_term.gait_phase is not None
    )
    self.command_term = command_term

  def feet(
    self,
    height: torch.Tensor,
    quat: torch.Tensor,
    linear_w: torch.Tensor,
    angular_w: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros = torch.zeros_like(height)
    root = torch.stack((zeros, zeros, height), dim=-1)
    pos, rot, vel, angular = self.fk(root, quat, q, linear_w, angular_w, qd)
    ends = self.capsule_ends[None].expand(len(height), -1, -1, -1)
    offset = quat_apply(rot[:, :, None].expand(-1, -1, ends.shape[2], -1), ends)
    lowest = pos[..., 2] + (offset[..., 2] - self.capsule_radii).min(dim=-1).values
    sites = quat_apply(rot, self.site_offsets.expand_as(pos))
    return lowest, pos + sites, vel + torch.cross(angular, sites, dim=-1)

  def score(
    self,
    height: torch.Tensor,
    quat: torch.Tensor,
    linear_w: torch.Tensor,
    angular_w: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    command: torch.Tensor,
    phase: torch.Tensor,
  ) -> torch.Tensor:
    soles, sites, site_vel = self.feet(height, quat, linear_w, angular_w, q, qd)
    cosine, sine = gait_phase_features(phase).unbind(-1)
    swing = torch.stack((sine, -sine), dim=-1).clamp_min(0)
    speed = command[:, 0].clamp_min(0)
    desired_lift = (0.03 + 0.04 * speed[:, None]) * swing.square()
    height_error = ((soles - desired_lift) / 0.022).square().mean(-1)
    foot_gap = quat_apply_inverse(yaw_quat(quat), sites[:, 0] - sites[:, 1])[:, 0]
    desired_gap = -cosine * speed / (2 * gait_cycle_hz(speed))
    stride_error = ((foot_gap - desired_gap) / 0.10).square()
    support = (1 - swing).pow(4)
    slip_sq = site_vel[..., :2].square().sum(-1)
    support_error = (
      (support * slip_sq).sum(-1) / support.sum(-1).clamp_min(1e-4) / 0.25**2
    )
    return forward_gait_gate(command) * (
      0.55 / (1 + height_error) + 0.35 / (1 + stride_error) + 0.10 / (1 + support_error)
    )

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    height, quat, linear_w, angular_b = physical_root(env)
    robot = env.scene["robot"]
    command, phase = self.command_term.command, self.command_term.gait_phase
    assert phase is not None
    return self.score(
      height,
      quat,
      linear_w,
      quat_apply(quat, angular_b),
      robot.data.joint_pos,
      robot.data.joint_vel,
      command,
      advance_gait_phase(phase, command, env.step_dt),
    )
