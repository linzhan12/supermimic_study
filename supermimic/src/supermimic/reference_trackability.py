"""Evaluate proposed references through a frozen SuperMimic world model."""

from collections.abc import Iterator
from itertools import chain

import torch
from mjlab.utils.lab_api.math import quat_conjugate, quat_mul, yaw_quat

from supermimic.locomotion_planner import PREVIEW_STEPS, ReferenceWindow, heading_twist
from supermimic.models import Dynamics, MultiEncoderAE
from supermimic.representations import StateRepresentation
from supermimic.tasks.task_spec import build_actor_obs_td


def projected_twist(
  previous: torch.Tensor, current: torch.Tensor, dt: float
) -> torch.Tensor:
  linear = (current[:, :3] - previous[:, :3]) / dt
  delta = yaw_quat(quat_mul(current[:, 3:], quat_conjugate(previous[:, 3:])))
  radians = 2 * torch.atan2(delta[:, 3], delta[:, 0])
  angular = torch.zeros_like(linear)
  angular[:, 2] = torch.atan2(radians.sin(), radians.cos()) / dt
  return heading_twist(previous[:, 3:], linear, angular)


def tracking_steps(
  windows: list[ReferenceWindow],
  actor: MultiEncoderAE,
  dynamics: Dynamics,
  representation: StateRepresentation,
  state: torch.Tensor,
  aux: torch.Tensor,
  history: torch.Tensor,
  context: torch.Tensor,
  steps: int,
  *,
  dt: float,
  yaw_rate: float,
  heading_strength: float,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
  """Yield predicted root poses, joint states and actions in control order."""
  count = len(windows)
  if (
    not windows or not 1 <= steps <= min(len(w.q) for w in windows) - PREVIEW_STEPS + 1
  ):
    raise ValueError("The world-model rollout must fit every reference preview.")
  state = state.expand(count, -1)
  aux = aux.expand(count, -1)
  history = history.expand(count, -1, -1)
  context = context.expand(count, -1)
  for index in range(steps):
    t = torch.tensor([index], device=state.device)
    goals = torch.cat(
      [
        window.actor_goal(
          t, aux[i : i + 1], yaw_rate=yaw_rate, heading_strength=heading_strength, dt=dt
        )
        for i, window in enumerate(windows)
      ]
    )
    action = actor(build_actor_obs_td({"motion": goals}, history))
    aux, state = dynamics(state, action, aux, context)
    joints = representation.joint_state(state)
    yield aux, joints, action
    sensor = representation.compute_actor_sensor(
      state, context, aux_state=aux, joint_state=joints, last_action=action
    )
    history = torch.cat((history[:, 1:], sensor.unsqueeze(1)), dim=1)


@torch.no_grad()
def predict_tracking_cost(
  windows: list[ReferenceWindow],
  actor: MultiEncoderAE,
  dynamics: Dynamics,
  representation: StateRepresentation,
  state: torch.Tensor,
  aux: torch.Tensor,
  history: torch.Tensor,
  context: torch.Tensor,
  steps: int,
  *,
  command: torch.Tensor | None = None,
  command_weight: float = 0.0,
  dt: float = 0.02,
  heading_strength: float = 0.0,
) -> torch.Tensor:
  """Roll out the same motion actor that will execute the proposed reference.

  The caller provides one measured initial state. Candidates run in parallel
  from that state without advancing the physical environment. Error compares
  positions to the matching next reference frame, including pelvis displacement.
  """
  count = len(windows)
  yaw_rate = float(command[2]) if heading_strength and command is not None else 0.0
  rollout = tracking_steps(
    windows,
    actor,
    dynamics,
    representation,
    state,
    aux,
    history,
    context,
    steps,
    dt=dt,
    yaw_rate=yaw_rate,
    heading_strength=heading_strength,
  )
  # The generator checks for a complete preview before the first model step.
  first = next(rollout)
  joint_targets = torch.stack([w.q[1 : steps + 1] for w in windows])
  root_targets = torch.stack([w.p[1 : steps + 1] for w in windows])
  origin = torch.stack([w.p[0] for w in windows])
  real_origin = aux[:, :2]
  cost = state.new_zeros(count)
  use_command = command is not None and command_weight > 0.0
  command_start = steps // 3
  twist_sum = state.new_zeros((count, 3))
  previous = aux.expand(count, -1)
  for index, (aux, joints, _) in enumerate(chain((first,), rollout)):
    if use_command and index >= command_start:
      twist_sum += projected_twist(previous, aux, dt)
    previous = aux
    q = joints[..., : joint_targets.shape[-1]]
    q_error = ((q - joint_targets[:, index]) / 0.25).square().mean(-1)
    expected_xy = root_targets[:, index, :2] - origin[:, :2]
    planar = ((aux[:, :2] - real_origin - expected_xy) / 0.10).square().sum(-1)
    height = ((aux[:, 2] - root_targets[:, index, 2]) / 0.08).square()
    # The squared horizontal part of a quaternion bounds the gravity tilt
    # independent of yaw and of which quaternion hemisphere is used.
    tilt = (2 * aux[:, 4:6].norm(dim=-1) - 0.35).clamp_min(0).square()
    cost += q_error + 0.5 * planar + 0.5 * height + 8 * tilt
  cost = cost / steps
  if use_command:
    assert command is not None
    delta = twist_sum / (steps - command_start) - command
    scaled = delta / delta.new_tensor((0.18, 0.16, 0.50))
    cost += command_weight * torch.nn.functional.smooth_l1_loss(
      scaled, torch.zeros_like(scaled), reduction="none"
    ).sum(dim=-1)
  return cost.nan_to_num(nan=1000.0, posinf=1000.0, neginf=1000.0)
