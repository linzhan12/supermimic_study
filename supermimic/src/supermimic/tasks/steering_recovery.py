"""State-conditioned velocity corrections for turns and lateral steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply
from torch import nn

from supermimic.models import build_mlp
from supermimic.tasks.velocity_tracking import heading_velocity, physical_root

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def steering_gate(command: torch.Tensor) -> torch.Tensor:
  turn = ((command[:, 2].abs() - 0.16) / 0.20).clamp(0, 1)
  lateral = ((command[:, 1].abs() - 0.08) / 0.10).clamp(0, 1)
  return torch.maximum(turn, lateral)


def steering_recovery_reward(quat, linear_w, angular_w, command):
  error = (heading_velocity(quat, linear_w, angular_w) - command) / command.new_tensor(
    (0.14, 0.14, 0.28)
  )
  return steering_gate(command) / (1 + error.square().sum(-1))


def physical_steering_recovery_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  _, quat, linear_w, angular_b = physical_root(env)
  return steering_recovery_reward(
    quat,
    linear_w,
    quat_apply(quat, angular_b),
    env.command_manager.get_command("velocity"),
  )


class SteeringRecoveryAdapter(nn.Module):
  """Learn leg and waist corrections for non-straight velocity commands."""

  def __init__(
    self,
    obs_dim: int,
    action_names: tuple[str, ...],
    action_scale: torch.Tensor,
    limit_rad: float,
  ) -> None:
    super().__init__()
    indices = torch.tensor(
      [
        i
        for i, name in enumerate(action_names)
        if name.startswith("waist_")
        or any(joint in name for joint in ("hip_", "knee_", "ankle_"))
      ],
      device=action_scale.device,
    )
    self.register_buffer("indices", indices, persistent=False)
    self.register_buffer(
      "limits", limit_rad / action_scale.reshape(-1)[indices], persistent=False
    )
    self.net = build_mlp(obs_dim + 3, (128, 128), output_dim=len(indices))
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(
    self, action: torch.Tensor, state: torch.Tensor, command: torch.Tensor
  ) -> torch.Tensor:
    delta = self.net(torch.cat((state, command), dim=-1)).tanh()
    return action.index_add(
      -1, self.indices, delta * self.limits * steering_gate(command)[:, None]
    )
