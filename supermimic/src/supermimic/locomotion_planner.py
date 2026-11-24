"""Rolling whole-body references for the existing SuperMimic motion tracker."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

from supermimic.representations import (
  GOAL_FRAME_STRIDE,
  NUM_GOAL_FRAMES,
  assemble_anchor_window,
)

Mode = Literal["stand", "start", "walk", "stop"]
PREVIEW_STEPS = (NUM_GOAL_FRAMES - 1) * GOAL_FRAME_STRIDE + 1


def heading_twist(
  quat: torch.Tensor, linear: torch.Tensor, angular: torch.Tensor
) -> torch.Tensor:
  planar = quat_apply_inverse(yaw_quat(quat), linear)[..., :2]
  return torch.cat((planar, angular[..., 2:3]), dim=-1)


@dataclass(frozen=True)
class ReferenceWindow:
  q: torch.Tensor
  qd: torch.Tensor
  p: torch.Tensor
  quat: torch.Tensor
  feet: torch.Tensor
  contact: torch.Tensor

  def take(self, index: torch.Tensor | slice) -> "ReferenceWindow":
    return ReferenceWindow(
      self.q[index],
      self.qd[index],
      self.p[index],
      self.quat[index],
      self.feet[index],
      self.contact[index],
    )

  def hold_after(self, start: int, length: int) -> "ReferenceWindow":
    """After the last sample, hold the pose and keep its velocity zero."""
    index = torch.arange(start, start + length, device=self.q.device)
    stopped = index >= len(self.q)
    window = self.take(index.clamp(max=len(self.q) - 1))
    return ReferenceWindow(
      window.q,
      window.qd.masked_fill(stopped[..., None], 0),
      window.p,
      window.quat,
      window.feet,
      window.contact,
    )

  def actor_goal(self, steps: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    """Use the original 10-frame layout with relative root orientation."""
    offsets = GOAL_FRAME_STRIDE * torch.arange(NUM_GOAL_FRAMES, device=self.q.device)
    index = steps[..., None] + offsets
    _, orientation = assemble_anchor_window(
      aux[..., :3], aux[..., 3:], self.p[index], self.quat[index]
    )
    content = torch.cat((self.q[index], self.qd[index]), dim=-1).flatten(start_dim=1)
    return torch.cat((content, orientation), dim=-1)


@dataclass(frozen=True)
class ReferenceClip:
  name: str
  mode: Mode
  frames: ReferenceWindow
  twist: torch.Tensor
  mean_tracking_bias: torch.Tensor
  period_steps: int = 0

  @classmethod
  def load(
    cls, path: Path, names: tuple[str, ...], device: str, dt: float
  ) -> "ReferenceClip":
    with np.load(path, allow_pickle=False) as data:
      if tuple(data["joint_names"].tolist()) != names or not np.isclose(data["dt"], dt):
        raise ValueError(f"{path}: unexpected G1 joint order or control interval.")
      mode = str(data["mode"].item())
      if mode not in ("stand", "start", "walk", "stop"):
        raise ValueError(f"{path}: invalid reference mode {mode!r}.")
      arrays = {
        field: torch.tensor(data[field], dtype=torch.float32, device=device)
        for field in (
          "joint_pos",
          "joint_vel",
          "root_pos_w",
          "root_quat_w",
          "foot_position_w_m",
          "root_lin_vel_w",
          "root_ang_vel_w",
          "mean_tracking_bias",
        )
      }
      contact = torch.tensor(data["foot_contact"], dtype=torch.bool, device=device)
      period = int(data["period_steps"]) if "period_steps" in data else 0
    frames = ReferenceWindow(
      arrays["joint_pos"],
      arrays["joint_vel"],
      arrays["root_pos_w"],
      arrays["root_quat_w"],
      arrays["foot_position_w_m"],
      contact,
    )
    count = len(frames.q)
    joints = len(names)
    if (
      count < PREVIEW_STEPS
      or frames.q.shape != (count, joints)
      or frames.qd.shape != (count, joints)
      or frames.p.shape != (count, 3)
      or frames.quat.shape != (count, 4)
      or frames.feet.shape != (count, 2, 3)
      or frames.contact.shape != (count, 2)
      or arrays["root_lin_vel_w"].shape != (count, 3)
      or arrays["root_ang_vel_w"].shape != (count, 3)
      or arrays["mean_tracking_bias"].shape != (3,)
      or not 0 <= period <= count - PREVIEW_STEPS
      or not all(torch.isfinite(a).all() for a in arrays.values())
    ):
      raise ValueError(f"{path}: invalid or incomplete motion reference.")
    twist = heading_twist(
      frames.quat, arrays["root_lin_vel_w"], arrays["root_ang_vel_w"]
    )
    return cls(path.stem, mode, frames, twist, arrays["mean_tracking_bias"], period)


@dataclass(frozen=True)
class _Candidate:
  clip: ReferenceClip
  frame: int
  cost: float
  pace_error: float


class LocomotionPlanner:
  """Choose a reference at compatible support states, then track a committed prefix.

  An optional score adds the learned tracker's predicted closed-loop error to the
  kinematic ranking. That score never replaces the geometric eligibility check.
  """

  def __init__(
    self,
    clips: list[ReferenceClip],
    initial: ReferenceWindow,
    dt: float,
    *,
    trackability_weight: float = 0.0,
    strict_contact: bool = True,
    prediction_steps: int = 10,
    pace_margin: float | None = None,
    stop_double_support_pose: float | None = None,
  ) -> None:
    if not any(clip.mode == "walk" for clip in clips):
      raise ValueError("The reference bank requires at least one walking clip.")
    self.clips = clips
    self.dt = dt
    self.replan_steps = max(1, round(0.10 / dt))
    self.blend_steps = max(3, round(0.16 / dt))
    self.preview_length = PREVIEW_STEPS + max(
      self.replan_steps + self.blend_steps + 2, prediction_steps
    )
    self.weight = trackability_weight
    self.strict_contact = strict_contact
    self.pace_margin = pace_margin
    self.stop_double_support_pose = stop_double_support_pose
    self.plan = initial.hold_after(0, self.preview_length)
    self.clip: ReferenceClip | None = None
    self.frame = 0
    self.offset = 0
    self.filtered_command = initial.p.new_zeros(3)
    self.mode: Mode = "stand"
    self.last_costs: list[dict] = []

  def preview(self) -> ReferenceWindow:
    return self.plan.hold_after(self.offset, PREVIEW_STEPS)

  def advance(self) -> None:
    self.offset += 1

  def update(
    self,
    command: torch.Tensor,
    observed_contact: torch.Tensor,
    score_prediction: Callable[[list[ReferenceWindow]], torch.Tensor] | None = None,
    *,
    stop_fallback: Callable[[], ReferenceWindow | None] | None = None,
  ) -> None:
    command_rate = command.new_tensor((0.7, 0.6, 1.8))
    self.filtered_command += (command - self.filtered_command).clamp(
      -command_rate * self.dt, command_rate * self.dt
    )
    if self.offset % self.replan_steps:
      return
    moving = bool(command.abs().max() > 0.025)
    source_frame = self.frame + self.offset
    active = self.clip
    if active is not None and self.offset < self.blend_steps:
      return
    if active is None and self.mode == "stop":
      if self.offset < len(self.plan.q) - 1:
        return
      self.mode = "stand"
    if not moving and self.mode == "stand":
      return
    # A start or stop has a task-relevant contact sequence. Finish it unless
    # the operator changes between moving and stopping, then replan immediately.
    if (
      active is not None
      and active.mode in ("start", "stop")
      and moving == (active.mode == "start")
      and source_frame < len(active.frames.q) - 2
    ):
      return
    if not moving and self.mode == "stop":
      self.mode = "stand"
      self.clip = None
      return
    if moving:
      mode: Mode = "start" if self.mode == "stand" else "walk"
    else:
      mode = "stop"
    candidates = self._candidates(mode, observed_contact)
    if not candidates and mode == "start":
      mode = "walk"
      candidates = self._candidates(mode, observed_contact)
    if not candidates:
      self.last_costs = []
      if mode == "stop" and stop_fallback is not None and bool(observed_contact.all()):
        recovery = stop_fallback()
        if recovery is not None:
          self.plan = recovery
          self.mode = "stop"
          self.clip = None
          self.frame = self.offset = 0
      return
    if mode == "walk" and self.pace_margin is not None:
      smallest = min(candidate.pace_error for candidate in candidates)
      candidates = [
        candidate
        for candidate in candidates
        if candidate.pace_error <= smallest + self.pace_margin
      ]
    candidates = sorted(candidates, key=lambda candidate: candidate.cost)[:4]
    windows = [self._join(candidate) for candidate in candidates]
    prediction = (
      score_prediction(windows).tolist()
      if score_prediction is not None and self.weight > 0
      else [0.0] * len(candidates)
    )
    total = [
      candidate.cost + self.weight * extra
      for candidate, extra in zip(candidates, prediction, strict=True)
    ]
    best = min(range(len(candidates)), key=total.__getitem__)
    self.last_costs = [
      {
        "clip": candidate.clip.name,
        "frame": candidate.frame,
        "kinematics": candidate.cost,
        "pace_error": candidate.pace_error,
        "world_model": extra,
        "selected": i == best,
      }
      for i, (candidate, extra) in enumerate(zip(candidates, prediction, strict=True))
    ]
    choice = candidates[best]
    if self._continues(choice):
      # This trajectory already contains the next source samples in the
      # same world frame. Restarting the blend would change the phase.
      return
    self.clip = choice.clip
    self.mode = choice.clip.mode
    self.frame = choice.frame
    self.plan = windows[best]
    self.offset = 0

  def _candidates(self, mode: Mode, observed_contact: torch.Tensor) -> list[_Candidate]:
    old = self.plan.hold_after(self.offset, self.blend_steps + 1)
    old_support = old.contact[0] if old.contact[0].any() else observed_contact
    continued = self.frame + self.offset
    choices = []
    for clip in self.clips:
      if clip.mode != mode:
        continue
      frames = clip.frames
      last_entry = len(frames.q) - PREVIEW_STEPS
      entries = min(last_entry + 1, clip.period_steps or last_entry + 1)
      indices = torch.arange(entries, device=frames.q.device)
      if mode == "start":
        indices = indices[: max(1, round(0.28 / self.dt))]
      if mode == "stop":
        indices = indices[: max(1, round(0.80 / self.dt))]
      if clip is self.clip and continued <= last_entry and mode == "walk":
        indices = torch.tensor([continued], device=frames.q.device)
      # Match ordered support and compare stance feet relative to the pelvis.
      new_contact = frames.contact[indices]
      common = new_contact & old_support
      contact_cost = (new_contact != old_support).float().mean(dim=-1)
      observed_cost = (new_contact != observed_contact).float().mean(dim=-1)
      old_feet = self._local_feet(old.p[:1], old.quat[:1], old.feet[:1])
      clip_feet = self._local_feet(
        frames.p[indices], frames.quat[indices], frames.feet[indices]
      )
      foot_cost = (((clip_feet - old_feet).square().sum(-1)) * common).sum(-1)
      pose_cost = (frames.q[indices] - old.q[0]).square().mean(-1)
      rate_cost = (frames.qd[indices] - old.qd[0]).square().mean(-1)
      if clip.period_steps:
        pace = clip.twist[: clip.period_steps].mean(dim=0, keepdim=True)
      else:
        pace_steps = max(2, round(0.72 / self.dt))
        at = (indices[:, None] + torch.arange(pace_steps, device=indices.device)).clamp(
          max=len(frames.q) - 1
        )
        pace = clip.twist[at].mean(dim=1)
      pace = pace + clip.mean_tracking_bias
      target = self.filtered_command
      if mode == "stop":
        target = target * 0.6
      pace_cost = (
        ((pace - target) / pace.new_tensor((0.15, 0.16, 0.6))).square().sum(-1)
      )
      score = 7 * pose_cost + 0.025 * rate_cost + 1.2 * contact_cost
      score += 0.3 * observed_cost + 14 * foot_cost + 0.65 * pace_cost
      if self.strict_contact:
        # A standing pose can differ in arm posture from the recorded start.
        # Permit that difference when both feet remain in support at the join.
        double_support_start = (mode == "start") & common.all(-1)
        pose_limit = torch.where(double_support_start, 0.32, 0.26)
        if mode == "stop" and self.stop_double_support_pose is not None:
          settled_support = common.all(-1) & observed_contact.all()
          pose_limit = torch.where(
            settled_support, self.stop_double_support_pose, pose_limit
          )
        compatible = (
          common.any(-1) & (pose_cost.sqrt() < pose_limit) & (foot_cost < 0.04)
        )
        score = score.masked_fill(~compatible, torch.inf)
      if clip is self.clip:
        score = score - 0.20 * (indices == continued)
      index = int(score.argmin())
      if torch.isfinite(score[index]):
        error = pace_cost.expand_as(score)[index].sqrt().item()
        choices.append(
          _Candidate(clip, int(indices[index]), float(score[index]), error)
        )
    return choices

  @staticmethod
  def _local_feet(
    p: torch.Tensor, quat: torch.Tensor, feet: torch.Tensor
  ) -> torch.Tensor:
    orientation = yaw_quat(quat)[:, None, :].expand(-1, 2, -1)
    return quat_apply_inverse(orientation, feet - p[:, None, :])

  def _continues(self, candidate: _Candidate) -> bool:
    return candidate.clip is self.clip and candidate.frame == self.frame + self.offset

  def _join(self, candidate: _Candidate) -> ReferenceWindow:
    if self._continues(candidate):
      return self.plan.hold_after(self.offset, self.preview_length)

    count = self.blend_steps + 1
    old = self.plan.hold_after(self.offset, count)
    remaining = len(candidate.clip.frames.q) - candidate.frame
    new = candidate.clip.frames.hold_after(
      candidate.frame, max(remaining, self.preview_length)
    )
    yaw_delta = quat_mul(yaw_quat(old.quat[:1]), quat_conjugate(yaw_quat(new.quat[:1])))
    rotation = yaw_delta.expand(len(new.q), -1)
    new_p = quat_apply(rotation, new.p)
    new_feet = quat_apply(rotation[:, None, :].expand(-1, 2, -1), new.feet)
    overlap = (old.contact[0] & new.contact[0]).nonzero().flatten()
    translation = old.p[0] - new_p[0]
    if len(overlap):
      translation = (old.feet[0, overlap] - new_feet[0, overlap]).mean(dim=0)
    # G1 reference height is part of the gait. Ground alignment only shifts XY.
    translation = translation * translation.new_tensor((1.0, 1.0, 0.0))
    new_p += translation
    new_feet += translation
    new_quat = quat_mul(rotation, new.quat)
    u = torch.linspace(0, 1, count, device=new.q.device)[:, None]
    blend = u.pow(3) * (10 - 15 * u + 6 * u.square())
    blend_rate = 30 * u.square() * (1 - u).square() / (self.blend_steps * self.dt)
    delta_q = new.q[:count] - old.q
    q = torch.cat((old.q + blend * delta_q, new.q[count:]))
    qd = torch.cat(
      (old.qd.lerp(new.qd[:count], blend) + blend_rate * delta_q, new.qd[count:])
    )
    p = torch.cat((old.p.lerp(new_p[:count], blend), new_p[count:]))
    local_quat = new_quat[:count]
    local_quat = torch.where(
      (old.quat * local_quat).sum(-1, keepdim=True) < 0, -local_quat, local_quat
    )
    quat = torch.cat(
      (
        torch.nn.functional.normalize(old.quat.lerp(local_quat, blend), dim=-1),
        new_quat[count:],
      )
    )
    feet = torch.cat(
      (old.feet.lerp(new_feet[:count], blend[..., None]), new_feet[count:])
    )
    contact = torch.cat(
      (torch.where(blend < 0.5, old.contact, new.contact[:count]), new.contact[count:])
    )
    return ReferenceWindow(q, qd, p, quat, feet, contact)
