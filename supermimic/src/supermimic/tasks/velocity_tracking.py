"""Independent flat-ground velocity commands and shared transition scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import time_out
from mjlab.managers import (
  CommandTerm,
  CommandTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


def heading_velocity(
  quat_w: torch.Tensor, linear_w: torch.Tensor, angular_w: torch.Tensor
) -> torch.Tensor:
  """Return forward and left speed in m/s and world-up yaw rate in rad/s."""
  planar = quat_apply_inverse(yaw_quat(quat_w), linear_w)[..., :2]
  return torch.cat((planar, angular_w[..., 2:3]), dim=-1)


def velocity_reward(quat_w, linear_w, angular_w, command):
  error = heading_velocity(quat_w, linear_w, angular_w) - command
  planar = torch.exp(-error[..., :2].square().sum(-1) / 0.25**2)
  yaw = torch.exp(-error[..., 2].square() / 0.5**2)
  return (planar + 0.5 * yaw) / 1.5


def fall_severity(height, quat_w):
  height_severity = 1.0 + (0.45 - height) / 0.2
  tilt_severity = 2 * quat_w[..., 1:3].square().sum(-1) / (1 - math.cos(math.pi / 3))
  return torch.maximum(height_severity, tilt_severity)


class VelocitySchedule:
  """Heading-frame velocity, integer steps left, and the sampled command category."""

  def __init__(
    self, dt: float, device: str, hold_s: tuple[float, float] = (1.0, 2.0),
    gait_fraction: float = 0.0,
  ):
    self.min_steps = max(1, math.ceil(hold_s[0] / dt))
    self.max_steps = math.floor(hold_s[1] / dt)
    if self.min_steps > self.max_steps:
      raise ValueError("Velocity hold interval must contain a whole control step.")
    self.lower = torch.tensor((-0.2, -0.25, -0.55), device=device)
    self.upper = torch.tensor((0.55, 0.25, 0.55), device=device)
    self.gait_fraction = gait_fraction
    # Categories are zero, forward-only, lateral-only, yaw-only, diagonal,
    # and walking with yaw. Reserve two of the seven draws for exact zero.
    self.masks = torch.tensor(
      ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)),
      device=device,
      dtype=torch.bool,
    )

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = self.lower.device
    category = (torch.randint(7, (count,), device=device) - 1).clamp_min(0)
    raw = torch.rand((count, 3), device=device) * (self.upper - self.lower) + self.lower
    command = torch.where(self.masks[category], raw, 0.0)
    if self.gait_fraction:
      walking = torch.rand(count, device=device) < self.gait_fraction
      speed = 0.19 + 0.36 * torch.rand(count, device=device)
      zero = torch.zeros_like(speed)
      command = torch.where(walking[:, None], torch.stack((speed, zero, zero), dim=-1), command)
      category = torch.where(walking, 1, category)
    steps = torch.randint(self.min_steps, self.max_steps + 1, (count,), device=device)
    return command, steps, category

  def advance(self, command, steps, category, elapsed=1):
    steps = steps - elapsed
    expired = steps <= 0
    next_command, next_steps, next_category = self.sample(len(command))
    return (
      torch.where(expired[:, None], next_command, command),
      torch.where(expired, next_steps, steps),
      torch.where(expired, next_category, category),
    )


class IndependentVelocityCommand(CommandTerm):
  def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.schedule = VelocitySchedule(
      env.step_dt, self.device, cfg.resampling_time_range, cfg.gait_fraction
    )
    self._command = torch.zeros((self.num_envs, 3), device=self.device)
    self.steps_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.category = torch.zeros_like(self.steps_left)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._command[env_ids], self.steps_left[env_ids], self.category[env_ids] = (
      self.schedule.sample(len(env_ids))
    )
    self.time_left[env_ids] = self.steps_left[env_ids] * self._env.step_dt

  def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None):
    ids = slice(None) if env_ids is None else env_ids
    elapsed = (torch.as_tensor(dt, device=self.device) > 0).long()
    self._command[ids], self.steps_left[ids], self.category[ids] = (
      self.schedule.advance(
        self._command[ids], self.steps_left[ids], self.category[ids], elapsed
      )
    )
    self.time_left[ids] = self.steps_left[ids] * self._env.step_dt

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class IndependentVelocityCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1.0, 2.0)
  gait_fraction: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> IndependentVelocityCommand:
    return IndependentVelocityCommand(self, env)


class ReferenceResetCommand(MotionCommand):
  """Use references for initial states, holding the reference frame between resets."""

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.metrics.clear()
    speed = self.motion.root_lin_vel_w[:, :2].norm(dim=-1)
    yaw_rate = self.motion.root_ang_vel_w[:, 2].abs()
    valid = torch.ones_like(speed, dtype=torch.bool)
    valid[self.motion.motion_ends - 1] = False
    self._standing_frames = (
      (valid & (speed < 0.08) & (yaw_rate < 0.15)).nonzero().flatten()
    )
    self._walking_frames = (valid & (speed > 0.2) & (speed < 0.85)).nonzero().flatten()
    if not len(self._standing_frames) or not len(self._walking_frames):
      raise ValueError("F1 requires reference clips with near-rest and walking frames.")

  def _uniform_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
    count = len(env_ids)
    stand = self._standing_frames[
      torch.randint(len(self._standing_frames), (count,), device=self.device)
    ]
    walk = self._walking_frames[
      torch.randint(len(self._walking_frames), (count,), device=self.device)
    ]
    return torch.where(torch.rand(count, device=self.device) < 0.25, stand, walk)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    self.update_relative_body_poses()

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class ReferenceResetCommandCfg(MotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> ReferenceResetCommand:
    return ReferenceResetCommand(self, env)


def physical_root(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, ...]:
  # Reward and termination managers run before the env refreshes derived kinematics.
  # Free-joint qpos/qvel already contain the integrated state at that point.
  robot = env.scene["robot"]
  pose = env.sim.data.qpos[:, robot.indexing.free_joint_q_adr]
  twist = env.sim.data.qvel[:, robot.indexing.free_joint_v_adr]
  return (
    pose[:, 2] - env.scene.env_origins[:, 2],
    pose[:, 3:7],
    twist[:, :3],
    twist[:, 3:6],
  )


def physical_velocity_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  _, quat_w, linear_w, angular_b = physical_root(env)
  command = env.command_manager.get_command("velocity")
  return velocity_reward(quat_w, linear_w, quat_apply(quat_w, angular_b), command)


def turn_tracking_reward(yaw_rate: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
  active = (command[..., 2].abs() / 0.30).clamp(0, 1)
  return active * torch.exp(-((yaw_rate - command[..., 2]) / 0.22).square())


def physical_turn_tracking_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  _, quat, _, angular_b = physical_root(env)
  command = env.command_manager.get_command("velocity")
  return turn_tracking_reward(quat_apply(quat, angular_b)[..., 2], command)


def physical_fall(env: ManagerBasedRlEnv) -> torch.Tensor:
  height, quat_w, _, _ = physical_root(env)
  return fall_severity(height, quat_w) > 1.0


def configure_velocity_env(cfg: ManagerBasedRlEnvCfg, gait_fraction: float = 0.0) -> None:
  reference = cfg.commands["motion"]
  assert isinstance(reference, MotionCommandCfg)
  reset_command = ReferenceResetCommandCfg(
    **{field.name: getattr(reference, field.name) for field in fields(reference)}
  )
  reset_command.sampling_mode = "uniform"
  reset_command.resampling_time_range = (1e9, 1e9)
  reset_command.pose_range = {}
  reset_command.velocity_range = {}
  reset_command.joint_position_range = (0.0, 0.0)
  cfg.commands["motion"] = reset_command
  cfg.commands["velocity"] = IndependentVelocityCommandCfg(gait_fraction=gait_fraction)
  cfg.episode_length_s = 10.0
  cfg.is_finite_horizon = False
  cfg.scale_rewards_by_dt = True
  cfg.rewards = {"velocity": RewardTermCfg(func=physical_velocity_reward, weight=1.0)}
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fall": TerminationTermCfg(func=physical_fall),
  }
