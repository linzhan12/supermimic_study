"""Speed-conditioned walking demonstrations and an optional velocity-policy adapter."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers import RewardTermCfg
from torch import nn

from supermimic.models import build_mlp
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.velocity_tracking import (
  IndependentVelocityCommand,
  advance_gait_phase,
  forward_gait_gate,
  gait_cycle_hz,
  gait_phase_features,
  heading_velocity,
  physical_root,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.models import MultiEncoderAE
  from supermimic.representations import StateRepresentation


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
    self.command_term = env.command_manager.get_term("velocity")
    assert isinstance(self.command_term, IndependentVelocityCommand)
    rows, speed = walking_samples(motion, env.step_dt)
    self.columns = tuple(
      motion.fk.joint_names.index(f"{side}_{joint}_joint")
      for side in ("left", "right")
      for joint in ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "hip_roll")
    )
    self.pose_width = motion.joint_pos.new_tensor((0.18, 0.22, 0.14, 0.26, 0.12) * 2)
    self.velocity_width = motion.joint_pos.new_tensor((1.9, 2.5, 1.9, 1.8, 1.2) * 2)
    self.height_width = 0.05
    phase = None
    if self.command_term.gait_phase is not None:
      left, right = self.columns[0], self.columns[5]
      leg = motion.joint_pos[rows, left] - motion.joint_pos[rows, right]
      rate = motion.joint_vel[rows, left] - motion.joint_vel[rows, right]
      scaled_rate = -rate / (math.tau * gait_cycle_hz(speed))
      angles = torch.atan2(scaled_rate, leg) / math.tau
      # Exclude frames whose legs are nearly static, including starts and finishes.
      cyclic = leg.square() + scaled_rate.square() >= 0.22**2
      rows, speed, phase = rows[cyclic], speed[cyclic], angles[cyclic]
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
    self.reference_rows = rows[selected]
    self.reference_speeds = speed[selected]
    self.reference_phase = phase[selected] if phase is not None else None
    self.phase_samples = None
    if self.reference_phase is not None:
      unit_phase = gait_phase_features(self.reference_phase)
      self.phase_samples = torch.cat((unit_phase, -unit_phase))
    q = motion.joint_pos[self.reference_rows][:, self.columns]
    qd = motion.joint_vel[self.reference_rows][:, self.columns]
    height = motion.root_pos_w[self.reference_rows, 2]
    sign = q.new_tensor((1, 1, 1, 1, -1) * 2)
    mirror = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    q = torch.cat((q, q[:, mirror] * sign))
    qd = torch.cat((qd, qd[:, mirror] * sign))
    height = height.repeat(2)
    self.speeds = self.reference_speeds.repeat(2)
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
    self,
    height: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    command: torch.Tensor,
    phase: torch.Tensor | None = None,
  ) -> torch.Tensor:
    state = self._features(height, q[:, self.columns], qd[:, self.columns])
    distance = (
      state.square().sum(-1, keepdim=True)
      + self.sample_norm_sq
      - 2 * state @ self.samples.T
    )
    speed_cost = ((command[:, :1] - self.speeds) / 0.085).square()
    if self.phase_samples is not None:
      assert phase is not None
      phase_similarity = gait_phase_features(phase) @ self.phase_samples.T
      distance = distance + 3 * (1 - phase_similarity.clamp(-1, 1))
    # A single nearby example preserves multimodal styles and opposite gait phases.
    cost = (distance.clamp_min(0) + speed_cost).min(-1).values
    return forward_gait_gate(command) / (1 + cost)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    height, _, _, _ = physical_root(env)
    robot = env.scene["robot"]
    command = self.command_term.command
    phase = self.command_term.gait_phase
    next_phase = (
      advance_gait_phase(phase, command, env.step_dt) if phase is not None else None
    )
    return self.score(
      height, robot.data.joint_pos, robot.data.joint_vel, command, next_phase
    )


class ForwardGaitAdapter(nn.Module):
  """Bounded state-dependent joint corrections for the straight velocity route."""

  def __init__(
    self,
    obs_dim: int,
    action_names: tuple[str, ...],
    scale: torch.Tensor,
    limit_rad: float,
    use_phase: bool = False,
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
    self.use_phase = use_phase
    self.net = build_mlp(
      obs_dim + 3 + 2 * use_phase, (128, 128), output_dim=len(indices)
    )
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)
    self.reflection: G1SagittalReflection | None = None

  def forward(
    self,
    action: torch.Tensor,
    normalized_state: torch.Tensor,
    command: torch.Tensor,
    phase: torch.Tensor | None = None,
  ) -> torch.Tensor:
    features = (normalized_state, command)
    if self.use_phase:
      assert phase is not None
      features += (gait_phase_features(phase),)
    delta = self.net(torch.cat(features, dim=-1)).tanh()
    delta = delta * self.limits * forward_gait_gate(command)[..., None]
    return action.index_add(-1, self.indices, delta)


class G1SagittalReflection(nn.Module):
  """Convert state and joint targets between mirrored G1 walking problems."""

  def __init__(
    self,
    joint_names: list[str],
    action_names: tuple[str, ...],
    scale: torch.Tensor,
    offset: torch.Tensor,
    blend: float,
  ) -> None:
    super().__init__()
    self.blend = blend
    for attr, names in (("joint", tuple(joint_names)), ("action", action_names)):
      counterparts = []
      for name in names:
        if name.startswith("left_"):
          partner = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
          partner = "left_" + name.removeprefix("right_")
        else:
          partner = name
        counterparts.append(names.index(partner))
      signs = scale.new_tensor(
        [-1 if "_roll_" in name or "_yaw_" in name else 1 for name in names]
      )
      self.register_buffer(
        f"{attr}_permutation",
        torch.tensor(counterparts, device=scale.device),
        persistent=False,
      )
      self.register_buffer(f"{attr}_sign", signs, persistent=False)
    self.register_buffer("scale", scale.reshape(-1).clone(), persistent=False)
    self.register_buffer("offset", offset.reshape(-1).clone(), persistent=False)
    self.register_buffer(
      "root_sign",
      scale.new_tensor((1, 1, -1, 1, 1, -1, 1, -1, 1, -1)),
      persistent=False,
    )
    self.register_buffer(
      "command_sign", scale.new_tensor((1, -1, -1)), persistent=False
    )

  def state(self, observation: torch.Tensor) -> torch.Tensor:
    root, q, qd = observation.split(
      (10, len(self.joint_sign), len(self.joint_sign)), dim=-1
    )
    return torch.cat(
      (
        root * self.root_sign,
        q.index_select(-1, self.joint_permutation) * self.joint_sign,
        qd.index_select(-1, self.joint_permutation) * self.joint_sign,
      ),
      dim=-1,
    )

  def action(self, action: torch.Tensor) -> torch.Tensor:
    position = action * self.scale + self.offset
    mirrored = position.index_select(-1, self.action_permutation) * self.action_sign
    return (mirrored - self.offset) / self.scale

  def weight(self, command: torch.Tensor) -> torch.Tensor:
    return 0.5 * self.blend * forward_gait_gate(command)[..., None]


def warm_start_gait_adapter(
  actor: MultiEncoderAE,
  representation: StateRepresentation,
  motion: MotionLoader,
  prior: ForwardGaitPrior,
  steps: int,
) -> None:
  """Imitate the frozen motion branch at compatible speed and phase before rollouts."""
  adapter = actor.forward_gait_adapter
  assert isinstance(adapter, ForwardGaitAdapter) and prior.reference_phase is not None
  motion_spec = next(spec for spec in actor.task_specs if spec.name == "motion")
  rows = prior.reference_rows
  with torch.no_grad():
    full_state = torch.cat(
      (
        motion.root_pos_w[rows],
        motion.root_quat_w[rows],
        motion.joint_pos[rows],
        motion.root_lin_vel_w[rows],
        motion.root_ang_vel_w[rows],
        motion.joint_vel[rows],
      ),
      dim=-1,
    )
    aux_state = representation.extract_aux_state(full_state)
    goal = motion_spec.extract_training_goal(motion, rows, aux_state)
    history = representation.represent(full_state)[:, None].expand(
      -1, actor.history_length, -1
    )
    speed = prior.reference_speeds.clamp_max(0.55)
    command = torch.stack(
      (speed, torch.zeros_like(speed), torch.zeros_like(speed)), dim=-1
    )
    actor.set_slot_mix({"motion": 1.0})
    teacher = actor(build_actor_obs_td({"motion": goal}, history))[:, adapter.indices]
  actor.set_slot_mix({"velocity": 1.0})
  optimizer = torch.optim.AdamW(adapter.parameters(), lr=8e-4)
  for step in range(steps):
    batch = torch.randint(len(rows), (128,), device=rows.device)
    obs = build_actor_obs_td({"velocity": command[batch]}, history[batch])
    obs["gait_phase"] = prior.reference_phase[batch]
    predicted = actor(obs)[:, adapter.indices]
    error = ((predicted - teacher[batch]) / adapter.limits).square().mean()
    optimizer.zero_grad(set_to_none=True)
    error.backward()
    optimizer.step()
    if step == 0 or step + 1 == steps:
      print(
        f"[INFO] Gait teacher step {step + 1}/{steps}: normalized action MSE {error.item():.5f}"
      )
  optimizer.zero_grad(set_to_none=True)
