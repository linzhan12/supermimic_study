"""Speed-conditioned walking demonstrations and an optional velocity-policy adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import RewardTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from torch import nn

from supermimic.models import build_mlp
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.velocity_tracking import (
  IndependentVelocityCommand,
  ReferenceResetCommand,
  advance_gait_phase,
  forward_gait_gate,
  gait_cycle_hz,
  gait_phase_features,
  heading_velocity,
  physical_root,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from supermimic.models import MultiEncoderAE
  from supermimic.representations import StateRepresentation


def walking_samples(
  motion: MotionLoader, dt: float, *, averaging_s: float = 0.48
) -> tuple[torch.Tensor, torch.Tensor]:
  """Find sustained forward frames using a centered filter confined to each clip."""
  half_smooth = max(1, round(0.5 * averaging_s / dt))
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
    use_physical_demos = isinstance(cfg, FullBodyGaitRewardCfg) and bool(
      cfg.reference_files
    )
    if use_physical_demos:
      motion = MotionLoader(list(cfg.reference_files), motion.fk, device=env.device)
    self.reference_motion = motion
    self.reference_files = cfg.reference_files if use_physical_demos else ()
    self.command_term = env.command_manager.get_term("velocity")
    assert isinstance(self.command_term, IndependentVelocityCommand)
    # Use a stride-length window for tracker rollouts so natural lateral transfer
    # does not turn an otherwise straight walk into a rejected frame.
    average_seconds = 1.48 if use_physical_demos else 0.48
    rows, speed = walking_samples(motion, env.step_dt, averaging_s=average_seconds)
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
    quat: torch.Tensor | None = None,
    linear_w: torch.Tensor | None = None,
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
    height, quat, linear_w, _ = physical_root(env)
    robot = env.scene["robot"]
    command = self.command_term.command
    phase = self.command_term.gait_phase
    next_phase = (
      advance_gait_phase(phase, command, env.step_dt) if phase is not None else None
    )
    return self.score(
      height,
      robot.data.joint_pos,
      robot.data.joint_vel,
      command,
      next_phase,
      quat,
      linear_w,
    )


def sagittal_joint_mapping(
  names: tuple[str, ...], like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  partners = []
  for name in names:
    if name.startswith("left_"):
      partner = "right_" + name.removeprefix("left_")
    elif name.startswith("right_"):
      partner = "left_" + name.removeprefix("right_")
    else:
      partner = name
    partners.append(names.index(partner))
  sign = like.new_tensor(
    [-1 if "_roll_" in name or "_yaw_" in name else 1 for name in names]
  )
  return torch.tensor(partners, device=like.device), sign


@dataclass(kw_only=True)
class FullBodyGaitRewardCfg(RewardTermCfg):
  reference_files: tuple[str, ...] = ()
  transfer_style: bool = False
  foot_orientation: bool = False
  foot_placement: bool = False
  reset_fraction: float = 0.0
  arm_weight: float = 0.0
  arm_reference_files: tuple[str, ...] = ()
  fast_stance_weight: float = 0.0
  fast_lateral_scale: float = 1.0


class FullBodyGaitPrior(ForwardGaitPrior):
  """Track speed- and phase-conditioned mocap, including weight transfer."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert self.phase_samples is not None
    motion = self.reference_motion
    rows = self.reference_rows
    names = tuple(motion.fk.joint_names)
    q = motion.joint_pos[rows]
    qd = motion.joint_vel[rows]
    permutation, sign = sagittal_joint_mapping(names, q)
    quat = motion.root_quat_w[rows]
    self.style_fk = None
    self.track_feet = isinstance(cfg, FullBodyGaitRewardCfg) and cfg.foot_orientation
    self.foot_offsets = None
    if isinstance(cfg, FullBodyGaitRewardCfg) and cfg.transfer_style:
      robot = env.scene["robot"]
      bodies = ("torso_link",)
      if self.track_feet:
        bodies += ("left_ankle_roll_link", "right_ankle_roll_link")
      body_indices = torch.tensor([robot.body_names.index(name) for name in bodies])
      self.style_fk = ForwardKinematics(
        env.sim.mj_model,
        robot.indexing.body_ids,
        robot.indexing.joint_q_adr,
        body_indices,
        device=env.device,
      )
      if cfg.foot_placement:
        site_rows, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        site_ids = robot.indexing.site_ids[site_rows].cpu()
        self.foot_offsets = q.new_tensor(env.sim.mj_model.site_pos[site_ids])
    transfer = self._transfer(
      motion.root_pos_w[rows, 2], quat, motion.root_lin_vel_w[rows], q
    )
    transfer_sign = (1, 1, -1, 1, -1, 1)
    if self.style_fk is not None:
      transfer_sign += (1, -1)
    split = len(transfer_sign)
    mirrored_transfer = transfer[:, :split] * q.new_tensor(transfer_sign)
    if self.track_feet:
      axes = transfer[:, split : split + 4].reshape(-1, 2, 2).flip(1)
      mirrored_axes = (axes * q.new_tensor((-1, 1))).flatten(1)
      mirrored_transfer = torch.cat((mirrored_transfer, mirrored_axes), dim=-1)
    if self.foot_offsets is not None:
      feet = transfer[:, -6:].reshape(-1, 2, 3).flip(1)
      mirrored_feet = (feet * q.new_tensor((1, -1, 1))).flatten(1)
      mirrored_transfer = torch.cat((mirrored_transfer, mirrored_feet), dim=-1)
    self.body_samples = torch.cat(
      (
        torch.cat((q, qd, transfer), dim=-1),
        torch.cat(
          (
            q[:, permutation] * sign,
            qd[:, permutation] * sign,
            mirrored_transfer,
          ),
          dim=-1,
        ),
      )
    )
    # Keep errors in physical units and emphasize hips and trunk, which control weight transfer.
    self.pose_widths = q.new_tensor(
      [
        0.11
        if "hip_roll" in name or name.startswith("waist_")
        else 0.17
        if "hip_" in name or "knee" in name or "ankle_" in name
        else 0.20
        if "shoulder_" in name or "elbow" in name
        else 0.32
        for name in names
      ]
    )
    self.rate_widths = 12 * self.pose_widths
    self.arm_weight = cfg.arm_weight if isinstance(cfg, FullBodyGaitRewardCfg) else 0.0
    self.arm_indices = (
      tuple(
        names.index(f"{side}_{joint}_joint")
        for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll", "elbow")
      )
      if self.arm_weight
      else ()
    )
    self.arm_samples = None
    if isinstance(cfg, FullBodyGaitRewardCfg) and cfg.arm_reference_files:
      arm_prior = ForwardGaitPrior(
        replace(cfg, reference_files=cfg.arm_reference_files), env
      )
      assert arm_prior.phase_samples is not None
      arm_rows = arm_prior.reference_rows
      arm_motion = arm_prior.reference_motion
      arm_q, arm_qd = arm_motion.joint_pos[arm_rows], arm_motion.joint_vel[arm_rows]
      arm_ids = permutation.new_tensor(self.arm_indices)
      mirror_ids = permutation[arm_ids]
      mirror_sign = sign[arm_ids]
      self.arm_samples = torch.cat(
        (
          torch.cat((arm_q[:, arm_ids], arm_qd[:, arm_ids]), dim=-1),
          torch.cat(
            (arm_q[:, mirror_ids] * mirror_sign, arm_qd[:, mirror_ids] * mirror_sign),
            dim=-1,
          ),
        )
      )
      self.arm_phase_samples = arm_prior.phase_samples
      self.arm_speeds = arm_prior.speeds
      self.arm_ids = arm_ids
    self.transfer_widths = q.new_tensor((0.023, 0.13, 0.15, 0.20, 0.10, 0.13))
    if self.style_fk is not None:
      self.transfer_widths = q.new_tensor(
        (0.023, 0.13, 0.15, 0.20, 0.070, 0.13, 0.075, 0.075)
      )
    self.fast_stance_weight = (
      cfg.fast_stance_weight if isinstance(cfg, FullBodyGaitRewardCfg) else 0.0
    )
    self.fast_lateral_scale = (
      cfg.fast_lateral_scale if isinstance(cfg, FullBodyGaitRewardCfg) else 1.0
    )
    if self.fast_stance_weight:
      self._prepare_stance_samples(names, q)
    if isinstance(cfg, FullBodyGaitRewardCfg) and cfg.reset_fraction:
      reset = env.command_manager.get_term("motion")
      assert isinstance(reset, ReferenceResetCommand)
      self.reset_command = reset
      self.reset_fraction = cfg.reset_fraction
      self.command_term.gait_reset = self._reset_from_demonstration

  def _prepare_stance_samples(self, names: tuple[str, ...], q: torch.Tensor) -> None:
    contacts = []
    ends = self.reference_motion.motion_ends.tolist()
    for path, begin, end in zip(self.reference_files, [0, *ends[:-1]], ends, strict=True):
      with np.load(path, allow_pickle=False) as demo:
        if "foot_contact_lr" not in demo:
          raise ValueError(f"{path}: repack the physical rollout to include foot contacts.")
        contact = demo["foot_contact_lr"]
        if contact.dtype != np.bool_ or contact.shape != (end - begin, 2):
          raise ValueError(f"{path}: invalid left/right foot contacts.")
        contacts.append(contact)
    contact = q.new_tensor(np.concatenate(contacts))[self.reference_rows]
    self.stance_contacts = torch.cat((contact, contact.flip(-1)))
    self.stance_knees = (names.index("left_knee_joint"), names.index("right_knee_joint"))
    values = self._stance_values(
      self.body_samples[:, : q.shape[-1]], self.body_samples[:, 2 * q.shape[-1] :]
    )
    self.stance_moments = (values * self.stance_contacts[..., None]).flatten(1)

  def _stance_values(self, q: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
    foot_start = len(self.transfer_widths)
    lateral_axis = transfer[:, foot_start : foot_start + 4 : 2]
    return torch.stack((q[:, self.stance_knees], lateral_axis), dim=-1)

  def _reset_from_demonstration(self, env_ids: torch.Tensor) -> None:
    command = self.command_term.command[env_ids]
    chance = torch.rand(len(env_ids), device=env_ids.device)
    straight = (command[:, 0] >= 0.16) & (command[:, 1:] == 0).all(-1)
    use_demo = straight & (chance < self.reset_fraction)
    ids = env_ids[use_demo]
    if not len(ids):
      return
    distance = (command[use_demo, :1] - self.reference_speeds) / 0.04
    selected = torch.multinomial(torch.softmax(-0.5 * distance.square(), dim=-1), 1)
    selected = selected.squeeze(-1)
    self.reset_command.reset_from_gait_demo(
      self.reference_motion, ids, self.reference_rows[selected]
    )
    assert self.reference_phase is not None and self.command_term.gait_phase is not None
    self.command_term.gait_phase[ids] = self.reference_phase[selected].remainder(1)

  def _transfer(
    self,
    height: torch.Tensor,
    quat: torch.Tensor,
    linear_w: torch.Tensor,
    q: torch.Tensor,
  ) -> torch.Tensor:
    zeros = torch.zeros_like(linear_w)
    heading = heading_velocity(quat, linear_w, zeros)
    world_up = quat.new_tensor((0.0, 0.0, 1.0)).expand_as(linear_w)
    up = quat_apply_inverse(quat, world_up)
    features = (height[:, None], up, heading[:, 1:2], linear_w[:, 2:3])
    if self.style_fk is not None:
      root_position = torch.cat((zeros[:, :2], height[:, None]), dim=-1)
      position, orientation, _, _ = self.style_fk(
        root_position, quat, q, zeros, zeros, torch.zeros_like(q)
      )
      torso_up = quat_apply_inverse(orientation[:, 0], world_up)
      features += (torso_up[:, :2],)
      if self.track_feet:
        foot_axis = quat.new_tensor((1.0, 0.0, 0.0)).expand(len(height), 2, 3)
        forward_w = quat_apply(orientation[:, 1:], foot_axis)
        heading_quat = yaw_quat(quat)[:, None].expand(-1, 2, -1)
        forward_heading = quat_apply_inverse(heading_quat, forward_w)
        features += (forward_heading[..., 1:].flatten(1),)
        if self.foot_offsets is not None:
          foot_offset_w = quat_apply(
            orientation[:, 1:], self.foot_offsets.expand_as(foot_axis)
          )
          site_w = position[:, 1:] + foot_offset_w
          site_heading = quat_apply_inverse(heading_quat, site_w)
          features += (site_heading.flatten(1),)
    return torch.cat(features, dim=-1)

  def score(
    self,
    height: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
    command: torch.Tensor,
    phase: torch.Tensor | None = None,
    quat: torch.Tensor | None = None,
    linear_w: torch.Tensor | None = None,
  ) -> torch.Tensor:
    assert (
      self.phase_samples is not None
      and phase is not None
      and quat is not None
      and linear_w is not None
    )
    phase_features = gait_phase_features(phase)
    similarity = phase_features @ self.phase_samples.T
    speed_error = (command[:, :1] - self.speeds) / 0.075
    # Choose the target from speed and cycle phase only. The policy cannot pick an easier pose.
    weights = torch.softmax(22 * similarity - 0.5 * speed_error.square(), dim=-1)
    reference_q, reference_qd, reference_transfer = (weights @ self.body_samples).split(
      (q.shape[-1], qd.shape[-1], self.body_samples.shape[-1] - 2 * q.shape[-1]), dim=-1
    )
    if self.arm_samples is not None:
      arm_similarity = phase_features @ self.arm_phase_samples.T
      arm_speed_error = (command[:, :1] - self.arm_speeds) / 0.075
      arm_weights = torch.softmax(
        22 * arm_similarity - 0.5 * arm_speed_error.square(), dim=-1
      )
      arm_q, arm_qd = (arm_weights @ self.arm_samples).chunk(2, dim=-1)
      reference_q = reference_q.index_copy(1, self.arm_ids, arm_q)
      reference_qd = reference_qd.index_copy(1, self.arm_ids, arm_qd)
    pose_error = ((q - reference_q) / self.pose_widths).square()
    rate_error = ((qd - reference_qd) / self.rate_widths).square()
    joint_error = 0.60 * pose_error.mean(-1) + 0.17 * rate_error.mean(-1)
    if self.arm_indices:
      arm_error = pose_error[:, self.arm_indices] + 0.25 * rate_error[:, self.arm_indices]
      joint_error = joint_error + self.arm_weight * arm_error.mean(-1)
    transfer = self._transfer(height, quat, linear_w, q)
    difference = transfer - reference_transfer
    fast = ((command[:, 0] - 0.35) / 0.15).clamp(0, 1)
    if self.fast_lateral_scale != 1.0:
      # Reduce the amplitude of the demonstrated lateral velocity at fast speeds.
      lateral = (
        difference[:, 4] + fast * (1 - self.fast_lateral_scale) * reference_transfer[:, 4]
      )
      difference = torch.cat(
        (difference[:, :4], lateral[:, None], difference[:, 5:]), dim=-1
      )
    n_transfer = len(self.transfer_widths)
    transfer_error = (
      (difference[:, :n_transfer] / self.transfer_widths).square().mean(-1)
    )
    if self.track_feet:
      foot_widths = difference.new_tensor((0.13, 0.20, 0.13, 0.20))
      foot_error = (
        (difference[:, n_transfer : n_transfer + 4] / foot_widths).square().mean(-1)
      )
      transfer_error = transfer_error + 0.8 * foot_error
    if self.foot_offsets is not None:
      placement_widths = difference.new_tensor((0.07, 0.05, 0.025) * 2)
      placement_error = (difference[:, -6:] / placement_widths).square().mean(-1)
      transfer_error = transfer_error + 1.3 * placement_error
    denominator = 1 + joint_error + 0.42 * transfer_error
    if self.fast_stance_weight:
      contact = weights @ self.stance_contacts
      support = contact[..., None].clamp_min(1e-5)
      target = (weights @ self.stance_moments).reshape(-1, 2, 2) / support
      stance_error = (self._stance_values(q, transfer) - target) / q.new_tensor((0.12, 0.09))
      error = (contact[..., None] * stance_error.square()).mean((1, 2))
      denominator = denominator + self.fast_stance_weight * fast * error
    return forward_gait_gate(command) / denominator


class ForwardGaitAdapter(nn.Module):
  """Bounded state-dependent joint corrections for the straight velocity route."""

  def __init__(
    self,
    obs_dim: int,
    action_names: tuple[str, ...],
    scale: torch.Tensor,
    limit_rad: float,
    use_phase: bool = False,
    full_body: bool = False,
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
    self.legacy_action_count = len(indices)
    if full_body:
      available = set(indices.tolist())
      added = indices.new_tensor(
        [i for i in range(len(action_names)) if i not in available]
      )
      indices = torch.cat((indices, added))
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

  def expand_checkpoint(self, state: dict[str, torch.Tensor], prefix: str) -> None:
    """Retain the learned limb corrections when adding torso and wrist outputs."""
    target = self.net[-1]
    layer = f"{prefix}net.{len(self.net) - 1}."
    for name in ("weight", "bias"):
      key = layer + name
      old = state[key]
      current = getattr(target, name)
      if old.shape[0] == self.legacy_action_count < current.shape[0]:
        expanded = current.detach().clone()
        expanded[: self.legacy_action_count] = old
        state[key] = expanded

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
      counterparts, signs = sagittal_joint_mapping(names, scale)
      self.register_buffer(
        f"{attr}_permutation",
        counterparts,
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


def recorded_gait_actions(
  prior: ForwardGaitPrior, action_names: tuple[str, ...]
) -> torch.Tensor | None:
  """Read teacher controls for the transition after each selected physical pose."""
  if not prior.reference_files or (
    len(prior.reference_files) == 1 and Path(prior.reference_files[0]).is_dir()
  ):
    return None
  motion = prior.reference_motion
  ends = motion.motion_ends.tolist()
  actions = []
  for path, start, end in zip(prior.reference_files, [0, *ends[:-1]], ends, strict=True):
    with np.load(path, allow_pickle=False) as demo:
      if "executed_action" not in demo:
        continue
      action = demo["executed_action"]
      if (
        "action_names" not in demo
        or tuple(demo["action_names"].tolist()) != action_names
        or action.shape != (end - start, len(action_names))
        or not np.isfinite(action).all()
      ):
        raise ValueError(f"{path}: invalid recorded action samples or joint order.")
      actions.append(action)
  if not actions:
    return None
  if len(actions) != len(ends):
    raise ValueError("Provide executed actions for every physical gait demonstration.")
  rows = prior.reference_rows
  # A trace stores the action that produced its row's state. For that state as
  # policy input, supervise with the action in the next row of the same clip.
  clip = torch.bucketize(rows, motion.motion_ends, right=True)
  if (rows + 1 >= motion.motion_ends[clip]).any():
    raise ValueError("Selected a terminal demonstration state with no next action.")
  controls = torch.as_tensor(
    np.concatenate(actions), device=rows.device, dtype=motion.joint_pos.dtype
  )
  return controls[rows + 1]


def warm_start_gait_adapter(
  actor: MultiEncoderAE,
  representation: StateRepresentation,
  prior: ForwardGaitPrior,
  steps: int,
  action_names: tuple[str, ...],
) -> None:
  """Imitate the frozen motion branch at compatible speed and phase before rollouts."""
  adapter = actor.forward_gait_adapter
  assert isinstance(adapter, ForwardGaitAdapter) and prior.reference_phase is not None
  motion = prior.reference_motion
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
    history = representation.represent(full_state)[:, None].expand(
      -1, actor.history_length, -1
    )
    speed = prior.reference_speeds.clamp_max(0.55)
    command = torch.stack(
      (speed, torch.zeros_like(speed), torch.zeros_like(speed)), dim=-1
    )
    teacher = recorded_gait_actions(prior, action_names)
    if teacher is None:
      aux_state = representation.extract_aux_state(full_state)
      goal = motion_spec.extract_training_goal(motion, rows, aux_state)
      actor.set_slot_mix({"motion": 1.0})
      teacher = actor(build_actor_obs_td({"motion": goal}, history))
    else:
      print("[INFO] Gait teacher uses actions recorded in physical reference rollouts.")
    teacher = teacher[:, adapter.indices]
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
