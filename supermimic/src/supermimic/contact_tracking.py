"""Bounded contact-phase adjustment for the F2 motion-tracking route."""

import torch
from mjlab.utils.lab_api.math import quat_apply
from torch import nn

from supermimic.locomotion_planner import heading_twist
from supermimic.utils.gencoord_helpers import decompose_rep_state

JOINTS = tuple(
  f"{side}_{joint}_joint"
  for side in ("left", "right")
  for joint in ("hip_pitch", "hip_roll", "ankle_pitch", "ankle_roll")
)


class ContactTrackingAdapter(nn.Module):
  """Correct eight lower-body position targets while tracking one dense reference."""

  def __init__(
    self,
    joint_names: tuple[str, ...],
    action_names: tuple[str, ...],
    action_scale: torch.Tensor,
    limit_rad: float,
  ) -> None:
    super().__init__()
    if not 0.0 < limit_rad <= 0.12:
      raise ValueError(
        "The contact adapter needs a positive limit of at most 0.12 rad."
      )
    indices = torch.tensor([action_names.index(name) for name in JOINTS])
    scales = action_scale.flatten()[indices.to(action_scale.device)]
    if not bool(torch.isfinite(scales).all() & (scales.abs() > 1e-6).all()):
      raise ValueError("The adapted joints need finite nonzero action scales.")
    self.register_buffer(
      "joints",
      torch.tensor([joint_names.index(name) for name in JOINTS]),
      persistent=False,
    )
    self.register_buffer("actions", indices, persistent=False)
    self.register_buffer("scale", scales.detach().clone(), persistent=False)
    self.limit_rad = limit_rad
    self.head = nn.Sequential(
      nn.Linear(13 + 2 * len(JOINTS), 48), nn.Tanh(), nn.Linear(48, len(JOINTS))
    )
    nn.init.zeros_(self.head[-1].weight)
    nn.init.zeros_(self.head[-1].bias)

  def forward(
    self,
    action: torch.Tensor,
    state: torch.Tensor,
    aux: torch.Tensor,
    command: torch.Tensor,
    q_ref: torch.Tensor,
    qd_ref: torch.Tensor,
    measured_contact: torch.Tensor,
    reference_contact: torch.Tensor,
    gain: float = 1.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    _, gravity, v, w, q, _ = decompose_rep_state(state)
    rotations = aux[:, 3:]
    twist = heading_twist(rotations, quat_apply(rotations, v), quat_apply(rotations, w))
    speed_scale = state.new_tensor((0.35, 0.23, 0.65))
    features = torch.cat(
      (
        gravity,
        twist / speed_scale,
        (command - twist) / speed_scale,
        (q_ref[:, self.joints] - q[:, self.joints]) / 0.25,
        qd_ref[:, self.joints] / 2.0,
        measured_contact.to(state.dtype),
        reference_contact.to(state.dtype),
      ),
      dim=-1,
    )
    radians = gain * self.limit_rad * self.head(features.clamp(-5.0, 5.0)).tanh()
    return action.index_add(-1, self.actions, radians / self.scale), radians
