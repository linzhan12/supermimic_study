"""Flat-ground posture shaping for the independent velocity task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import RewardTermCfg
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import quat_apply

from supermimic.tasks.velocity_tracking import physical_root

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def lower_body_score(
  joint_error: torch.Tensor,
  foot_position: torch.Tensor,
  foot_velocity: torch.Tensor,
  linear_w: torch.Tensor,
  yaw_rate: torch.Tensor,
  command: torch.Tensor,
) -> torch.Tensor:
  """Favor planted feet when moving and a settled neutral pose on release."""
  # Geometric support is available from the same state in MuJoCo and imagination.
  support = torch.sigmoid((0.045 - foot_position[..., 2]) / 0.01)
  foot_speed_sq = foot_velocity[..., :2].square().sum(-1)
  slip = (support * foot_speed_sq).sum(-1) / support.sum(-1).clamp_min(1e-4)
  moving = (1 - (1 - support).prod(-1)) * torch.exp(-slip / 0.20**2)

  stand = torch.exp(-joint_error.square().mean(-1) / 0.25**2)
  settled = torch.exp(
    -linear_w[..., :2].square().sum(-1) / 0.12**2 - yaw_rate.square() / 0.25**2
  )
  standing = stand * settled * support.prod(-1)
  return torch.where((command == 0).all(-1), standing, moving)


class VelocityPostureReward:
  """Score the same integrated G1 state in simulation and imagination."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    robot = env.scene["robot"]
    model = env.sim.mj_model
    device = env.device
    self.q_stand = robot.data.default_joint_pos[0].clone()
    self.waist = tuple(
      robot.joint_names.index(f"waist_{axis}_joint")
      for axis in ("yaw", "roll", "pitch")
    )
    self.arms = tuple(
      i
      for i, name in enumerate(robot.joint_names)
      if any(part in name for part in ("shoulder", "elbow", "wrist"))
    )
    bodies = [
      robot.body_names.index(name)
      for name in ("left_ankle_roll_link", "right_ankle_roll_link", "torso_link")
    ]
    self.fk = ForwardKinematics(
      model,
      robot.indexing.body_ids,
      robot.indexing.joint_q_adr,
      torch.tensor(bodies),
      device=device,
    )
    site_rows, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    site_ids = robot.indexing.site_ids[site_rows].cpu().numpy()
    self.foot_offsets = torch.as_tensor(
      model.site_pos[site_ids], device=device, dtype=self.q_stand.dtype
    )
    self.waist_width = self.q_stand.new_tensor((0.30, 0.22, 0.22))

  def score(
    self,
    height: torch.Tensor,
    quat_w: torch.Tensor,
    linear_w: torch.Tensor,
    angular_w: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    command: torch.Tensor,
  ) -> torch.Tensor:
    # Flat-ground support depends on root height and is invariant to horizontal origin.
    zeros = torch.zeros_like(height)
    root = torch.stack((zeros, zeros, height), dim=-1)
    pos, quat, vel, ang = self.fk(root, quat_w, q, linear_w, angular_w, qd)
    offsets = quat_apply(quat[:, :2], self.foot_offsets.expand_as(pos[:, :2]))
    foot_pos = pos[:, :2] + offsets
    foot_vel = vel[:, :2] + torch.cross(ang[:, :2], offsets, dim=-1)

    q_error = q - self.q_stand
    torso_tilt_sq = 4 * quat[:, 2, 1:3].square().sum(-1)
    waist = (q_error[:, self.waist] / self.waist_width).square().mean(-1)
    # Allow ordinary arm swing and only penalize excursions from the rest posture.
    arms = ((q_error[:, self.arms].abs() - 0.4).clamp_min(0) / 0.35).square().mean(-1)
    upper = (1 + torso_tilt_sq / 0.25**2 + waist + arms).reciprocal()
    lower = lower_body_score(
      q_error, foot_pos, foot_vel, linear_w, angular_w[:, 2], command
    )
    return 0.5 * (upper + lower)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    height, quat, linear, angular_b = physical_root(env)
    robot = env.scene["robot"]
    command = env.command_manager.get_command("velocity")
    return self.score(
      height,
      quat,
      linear,
      quat_apply(quat, angular_b),
      robot.data.joint_pos,
      robot.data.joint_vel,
      command,
    )
