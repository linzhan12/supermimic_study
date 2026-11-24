from unittest.mock import Mock

import pytest
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from tensordict import TensorDict

from supermimic.envs.velocity_world_model_env import VelocityWorldModelEnv
from supermimic.tasks.velocity_tracking import (
  IndependentVelocityCommand,
  VelocitySchedule,
  advance_gait_phase,
  forward_precision_reward,
  heading_velocity,
  velocity_reward,
)
from tests.conftest import make_mock_command_env, make_mock_representation


def test_heading_velocity_uses_yaw_and_world_vertical_rate():
  quat = quat_from_euler_xyz(
    torch.tensor([0.4]), torch.tensor([-0.2]), torch.tensor([torch.pi / 2])
  )
  linear = torch.tensor([[0.3, 0.2, 0.8]])
  angular = torch.tensor([[0.4, -0.7, 0.5]])
  torch.testing.assert_close(
    heading_velocity(quat, linear, angular), torch.tensor([[0.2, -0.3, 0.5]])
  )


def test_command_timer_keeps_zero_until_last_transition():
  schedule = VelocitySchedule(0.02, "cpu")
  replacement = torch.tensor([[0.35, 0.0, 0.4]])
  schedule.sample = Mock(
    return_value=(replacement, torch.tensor([50]), torch.tensor([5]))
  )
  zero = torch.zeros_like(replacement)
  initial = (zero, torch.tensor([2]), torch.tensor([0]))
  held = schedule.advance(*initial, elapsed=0)
  torch.testing.assert_close(held, initial)
  held = schedule.advance(*held)
  torch.testing.assert_close(held, (zero, torch.tensor([1]), torch.tensor([0])))
  torch.testing.assert_close(schedule.advance(*held), schedule.sample.return_value)


@pytest.mark.parametrize("with_phase", (False, True))
def test_imagination_scores_old_command_before_refresh(with_phase):
  schedule = VelocitySchedule(0.02, "cpu")
  schedule.sample = Mock(
    return_value=(
      torch.tensor([[0.5, 0.0, 0.0]]),
      torch.tensor([80]),
      torch.tensor([1]),
    )
  )
  env = make_mock_command_env()
  env.cfg.rewards = {}
  rep = make_mock_representation(rep_state_dim=68)
  initial_phase = torch.tensor([0.9]) if with_phase else None
  command = Mock(
    spec=IndependentVelocityCommand, schedule=schedule, gait_phase=initial_phase
  )
  motion = env.unwrapped.command_manager.get_term.return_value
  env.unwrapped.command_manager.get_term.side_effect = {
    "velocity": command,
    "motion": motion,
  }.__getitem__
  env.unwrapped.step_dt = 0.02
  pose = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]])
  dynamics = Mock(side_effect=lambda state, action, aux, ctx: (aux, state))
  wm_env = VelocityWorldModelEnv(dynamics, rep, env, 1, Mock(), 1, 0.05)
  rep.extract_aux_state.side_effect = lambda _: pose
  rep.represent.side_effect = lambda state: torch.zeros((state.shape[0], 68))
  batch = TensorDict(
    {
      "state": torch.zeros((1, 71)),
      "prev_action": torch.zeros((1, 29)),
      "sim_ctx": torch.zeros((1, 1)),
      "velocity_command": torch.zeros((1, 3)),
      "velocity_hold_steps": torch.ones(1, dtype=torch.long),
      "velocity_command_category": torch.zeros(1, dtype=torch.long),
    },
    batch_size=[1],
  )
  if with_phase:
    batch["gait_phase"] = initial_phase
  obs = wm_env.reset(batch)
  torch.testing.assert_close(obs["goal_velocity"], batch["velocity_command"])
  if with_phase:
    torch.testing.assert_close(obs["gait_phase"], initial_phase)
  obs, reward, done, _ = wm_env.step(batch["prev_action"])
  torch.testing.assert_close(reward, torch.tensor([0.02]))
  assert done.item() < 1e-6
  torch.testing.assert_close(obs["goal_velocity"], schedule.sample.return_value[0])
  if with_phase:
    torch.testing.assert_close(obs["gait_phase"], torch.tensor([0.25]))


def test_walking_clock_advances_continuously_across_speed_changes():
  phase = torch.tensor([0.99, 0.25, 0.80])
  command = torch.tensor([[0.22, 0.0, 0.0], [0.50, 0.0, 0.0], [0.35, 0.0, 0.4]])
  new_phase = advance_gait_phase(phase, command, 0.02)
  torch.testing.assert_close(
    new_phase, torch.tensor([0.00352, 0.268, 0.25]), atol=1e-7, rtol=0
  )


def test_gait_reward_allows_weight_transfer_and_keeps_steering_errors():
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
  linear = torch.tensor([[0.35, 0.12, 0.0]]).repeat(3, 1)
  angular = torch.zeros_like(linear)
  commands = torch.tensor([[0.35, 0.0, 0.0], [0.35, 0.2, 0.0], [0.35, 0.0, 0.4]])
  original = velocity_reward(quat, linear, angular, commands)
  gait = velocity_reward(quat, linear, angular, commands, allow_gait_lateral=True)
  assert gait[0] == 1 and original[0] < gait[0]
  torch.testing.assert_close(gait[1:], original[1:], atol=0, rtol=0)


def test_forward_precision_scores_forward_and_yaw_during_weight_transfer():
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
  linear = torch.tensor([[0.35, 0.13, 0.0], [0.45, 0.13, 0.0], [0.35, 0.13, 0.0]])
  angular = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.15]])
  command = torch.tensor([[0.35, 0.0, 0.0]]).repeat(3, 1)
  original = forward_precision_reward(quat, linear, angular, command)
  walking = forward_precision_reward(quat, linear, angular, command, allow_gait_lateral=True)
  assert walking[0] == 1 and original[0] < walking[0]
  assert (walking[1:] < walking[0]).all()
