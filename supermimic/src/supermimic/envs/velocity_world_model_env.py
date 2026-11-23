"""Differentiable rollout of the independent velocity task."""

import torch
from mjlab.utils.lab_api.math import quat_apply
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.velocity_tracking import (
  IndependentVelocityCommand,
  fall_severity,
  velocity_reward,
)
from supermimic.utils.gencoord_helpers import decompose_rep_state


class VelocityWorldModelEnv(WorldModelEnv):
  def _configure_scoring(self, env) -> None:
    term = env.unwrapped.command_manager.get_term("velocity")
    assert isinstance(term, IndependentVelocityCommand)
    self.schedule = term.schedule
    self.dt = env.unwrapped.step_dt

  def _on_reset(self, batch: TensorDict) -> None:
    self.command = batch["velocity_command"]
    self.steps_left = batch["velocity_hold_steps"]
    self.category = batch["velocity_command_category"]
    self._reset_sensors(batch)

  def _score_transition(
    self, next_aux_state, next_rep_state, next_joint_state, action, last_action
  ) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, linear_b, angular_b, _, _ = decompose_rep_state(next_rep_state)
    quat = next_aux_state[..., 3:7]
    reward = velocity_reward(
      quat, quat_apply(quat, linear_b), quat_apply(quat, angular_b), self.command
    )
    severity = fall_severity(next_aux_state[..., 2], quat)
    done = torch.sigmoid((severity - 1.0) / self._termination_softness)
    return done, reward * self.dt

  def _compute_commands(self) -> None:
    self.command, self.steps_left, self.category = self.schedule.advance(
      self.command, self.steps_left, self.category
    )

  def get_observations(self) -> TensorDict:
    hold_seconds = self.steps_left[:, None] * self.dt
    critic = torch.cat(
      (self.rep_state, self.command, hold_seconds, self.sim_ctx), dim=-1
    )
    return build_actor_obs_td(
      {"velocity": self.command}, self._sensor_history(), critic
    )
