"""Speed-conditioned walking demonstrations and an optional velocity-policy adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import RewardTermCfg
from torch import nn

from supermimic.models import build_mlp
from supermimic.tasks.velocity_tracking import heading_velocity, physical_root

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionLoader


def forward_gait_gate(command: torch.Tensor) -> torch.Tensor:
  """Ramp in at walking speeds and leave turns, lateral moves and stops alone."""
  forward, lateral, yaw = command.unbind(-1)
  return (
    ((forward - 0.12) / 0.12).clamp(0, 1)
    * (1 - (lateral / 0.10).square()).clamp_min(0)
    * (1 - (yaw / 0.22).square()).clamp_min(0)
  )


def walking_samples(
  motion: MotionLoader, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
  """Find sustained forward frames using a centered filter confined to each clip."""
  half_smooth = max(1, round(0.24 / dt))
  context = max(1, round(0.14 / dt))
  rows = torch.arange(motion.time_step_total, device=motion.joint_pos.device)
  clip = torch.bucketize(rows, motion.motion_ends, right=True)
  starts = torch.cat((motion.motion_ends.new_zeros(1), motion.motion_ends[:-1]))
  offsets = torch.arange(-half_smooth, half_smooth + 1, device=rows.device)
  window = (rows[:, None] + offsets).clamp(
    starts[clip, None], motion.motion_ends[clip, None] - 1
  )
  direction = heading_velocity(
    motion.root_quat_w,
    motion.root_lin_vel_w[window].mean(1),
    motion.root_ang_vel_w[window].mean(1),
  )
  forward, lateral, yaw = direction.unbind(-1)
  valid = (
    (forward >= 0.16) & (forward < 0.66) & (lateral.abs() < 0.13) & (yaw.abs() < 0.22)
  )
  near = rows[:, None] + torch.arange(-context, context + 1, device=rows.device)
  in_clip = (near[:, 0] >= starts[clip]) & (near[:, -1] < motion.motion_ends[clip])
  retained = valid[near.clamp(0, len(rows) - 1)].all(-1) & in_clip
  return rows[retained], forward[retained]


class ForwardGaitPrior:
  """Reward the closest compatible pose and joint velocity across multiple clips."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    motion = env.command_manager.get_term("motion").motion
    rows, speed = walking_samples(motion, env.step_dt)
    self.columns = tuple(
      motion.fk.joint_names.index(f"{side}_{joint}_joint")
      for side in ("left", "right")
      for joint in ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "hip_roll")
    )
    self.pose_width = motion.joint_pos.new_tensor((0.18, 0.22, 0.14, 0.26, 0.12) * 2)
    self.velocity_width = motion.joint_pos.new_tensor((1.9, 2.5, 1.9, 1.8, 1.2) * 2)
    self.height_width = 0.05
    selected = []
    bins = (0.16, 0.26, 0.36, 0.46, 0.56, 0.66)
    counts = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
      pool = ((speed >= low) & (speed < high)).nonzero().flatten()
      counts.append(len(pool))
      if len(pool):
        sample = torch.linspace(
          0, len(pool) - 1, min(80, len(pool)), device=rows.device
        )
        selected.append(pool[sample.round().long()])
    if sum(n > 0 for n in counts) < 3:
      raise ValueError(
        f"Forward gait prior needs at least three populated speed bins, got {counts}."
      )
    print(f"[INFO] Forward gait reference frames by m/s {bins}: {counts}")
    selected = torch.cat(selected)
    q = motion.joint_pos[rows[selected]][:, self.columns]
    qd = motion.joint_vel[rows[selected]][:, self.columns]
    height = motion.root_pos_w[rows[selected], 2]
    sign = q.new_tensor((1, 1, 1, 1, -1) * 2)
    mirror = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    q = torch.cat((q, q[:, mirror] * sign))
    qd = torch.cat((qd, qd[:, mirror] * sign))
    height = height.repeat(2)
    self.speeds = speed[selected].repeat(2)
    self.samples = self._features(height, q, qd)
    self.sample_norm_sq = self.samples.square().sum(-1)

  def _features(
    self, height: torch.Tensor, q: torch.Tensor, qd: torch.Tensor
  ) -> torch.Tensor:
    # Per-joint averages keep the reward scale independent of representation size.
    return torch.cat(
      (
        q / self.pose_width / len(self.columns) ** 0.5,
        qd / self.velocity_width / len(self.columns) ** 0.5,
        (height[:, None] - 0.8) / self.height_width,
      ),
      dim=-1,
    )

  def score(
    self, height: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, command: torch.Tensor
  ) -> torch.Tensor:
    state = self._features(height, q[:, self.columns], qd[:, self.columns])
    distance = (
      state.square().sum(-1, keepdim=True)
      + self.sample_norm_sq
      - 2 * state @ self.samples.T
    )
    speed_cost = ((command[:, :1] - self.speeds) / 0.085).square()
    # A single nearby example preserves multimodal styles and opposite gait phases.
    cost = (distance.clamp_min(0) + speed_cost).min(-1).values
    return forward_gait_gate(command) / (1 + cost)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    height, _, _, _ = physical_root(env)
    robot = env.scene["robot"]
    return self.score(
      height,
      robot.data.joint_pos,
      robot.data.joint_vel,
      env.command_manager.get_command("velocity"),
    )


class ForwardGaitAdapter(nn.Module):
  """Bounded state-dependent joint corrections for the straight velocity route."""

  def __init__(
    self,
    obs_dim: int,
    action_names: tuple[str, ...],
    scale: torch.Tensor,
    limit_rad: float,
  ) -> None:
    super().__init__()
    joints = (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
      "shoulder_pitch",
      "shoulder_roll",
      "elbow",
    )
    indices = torch.tensor(
      [
        action_names.index(f"{side}_{joint}_joint")
        for side in ("left", "right")
        for joint in joints
      ],
      device=scale.device,
    )
    self.register_buffer("indices", indices, persistent=False)
    self.register_buffer(
      "limits", limit_rad / scale.reshape(-1)[indices], persistent=False
    )
    self.net = build_mlp(obs_dim + 3, (128, 128), output_dim=len(indices))
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(
    self, action: torch.Tensor, normalized_state: torch.Tensor, command: torch.Tensor
  ) -> torch.Tensor:
    delta = self.net(torch.cat((normalized_state, command), dim=-1)).tanh()
    delta = delta * self.limits * forward_gait_gate(command)[..., None]
    return action.index_add(-1, self.indices, delta)
