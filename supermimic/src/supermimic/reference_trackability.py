"""Evaluate proposed references through a frozen SuperMimic world model."""

import torch

from supermimic.locomotion_planner import PREVIEW_STEPS, ReferenceWindow
from supermimic.models import Dynamics, MultiEncoderAE
from supermimic.representations import StateRepresentation
from supermimic.tasks.task_spec import build_actor_obs_td


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
) -> torch.Tensor:
  """Roll out the same motion actor that will execute the proposed reference.

  The caller provides one measured initial state. Candidates run in parallel
  from that state without advancing the physical environment. Error compares
  positions to the matching next reference frame, including pelvis displacement.
  """
  count = len(windows)
  if (
    not windows or not 1 <= steps <= min(len(w.q) for w in windows) - PREVIEW_STEPS + 1
  ):
    raise ValueError("The world-model rollout must fit every reference preview.")
  state = state.expand(count, -1)
  aux = aux.expand(count, -1)
  history = history.expand(count, -1, -1)
  context = context.expand(count, -1)
  joint_targets = torch.stack([w.q[1 : steps + 1] for w in windows])
  root_targets = torch.stack([w.p[1 : steps + 1] for w in windows])
  origin = torch.stack([w.p[0] for w in windows])
  real_origin = aux[:, :2]
  cost = state.new_zeros(count)
  for index in range(steps):
    t = torch.tensor([index], device=state.device)
    goals = torch.cat(
      [window.actor_goal(t, aux[i : i + 1]) for i, window in enumerate(windows)]
    )
    action = actor(build_actor_obs_td({"motion": goals}, history))
    aux, state = dynamics(state, action, aux, context)
    joints = representation.joint_state(state)
    q = joints[..., : joint_targets.shape[-1]]
    q_error = ((q - joint_targets[:, index]) / 0.25).square().mean(-1)
    expected_xy = root_targets[:, index, :2] - origin[:, :2]
    planar = ((aux[:, :2] - real_origin - expected_xy) / 0.10).square().sum(-1)
    height = ((aux[:, 2] - root_targets[:, index, 2]) / 0.08).square()
    # The squared horizontal part of a quaternion bounds the gravity tilt
    # independent of yaw and of which quaternion hemisphere is used.
    tilt = (2 * aux[:, 4:6].norm(dim=-1) - 0.35).clamp_min(0).square()
    cost += q_error + 0.5 * planar + 0.5 * height + 8 * tilt
    sensor = representation.compute_actor_sensor(
      state, context, aux_state=aux, joint_state=joints, last_action=action
    )
    history = torch.cat((history[:, 1:], sensor.unsqueeze(1)), dim=1)
  return (cost / steps).nan_to_num(nan=1000.0, posinf=1000.0, neginf=1000.0)
