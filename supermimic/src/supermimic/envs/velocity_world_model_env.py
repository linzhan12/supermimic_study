"""Differentiable rollout of the independent velocity task."""

import torch
from mjlab.utils.lab_api.math import quat_apply
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks.forward_foot_cycle import ForwardFootCycleReward
from supermimic.tasks.forward_gait import ForwardGaitPrior
from supermimic.tasks.steering_recovery import steering_recovery_reward
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.tasks.velocity_posture import (
  VelocityArmSwingReward,
  VelocityPostureReward,
)
from supermimic.tasks.velocity_tracking import (
  IndependentVelocityCommand,
  advance_gait_phase,
  fall_severity,
  forward_precision_reward,
  gait_phase_features,
  turn_tracking_reward,
  velocity_reward,
)
from supermimic.utils.gencoord_helpers import decompose_rep_state


class VelocityWorldModelEnv(WorldModelEnv):
  def _configure_scoring(self, env) -> None:
    term = env.unwrapped.command_manager.get_term("velocity")
    assert isinstance(term, IndependentVelocityCommand)
    self.schedule = term.schedule
    self.use_gait_phase = term.gait_phase is not None
    self.dt = env.unwrapped.step_dt
    self.turn_tracking_weight = (
      env.cfg.rewards["turn_tracking"].weight
      if "turn_tracking" in env.cfg.rewards
      else 0.0
    )
    self.forward_precision_weight = (
      env.cfg.rewards["forward_precision"].weight
      if "forward_precision" in env.cfg.rewards
      else 0.0
    )
    self.steering_recovery_weight = (
      env.cfg.rewards["steering_recovery"].weight
      if "steering_recovery" in env.cfg.rewards
      else 0.0
    )
    self.posture = None
    if "posture" in env.cfg.rewards:
      posture = env.unwrapped.reward_manager.get_term_cfg("posture")
      assert isinstance(posture.func, VelocityPostureReward)
      self.posture = posture.func
      self.posture_weight = posture.weight
      self.moving_posture_scale = posture.params.get("moving_scale", 1.0)
    self.arm_swing = None
    if "arm_swing" in env.cfg.rewards:
      swing = env.unwrapped.reward_manager.get_term_cfg("arm_swing")
      assert isinstance(swing.func, VelocityArmSwingReward)
      self.arm_swing = swing.func
      self.arm_swing_weight = swing.weight
    self.forward_gait = None
    if "forward_gait" in env.cfg.rewards:
      gait = env.unwrapped.reward_manager.get_term_cfg("forward_gait")
      assert isinstance(gait.func, ForwardGaitPrior)
      self.forward_gait = gait.func
      self.forward_gait_weight = gait.weight
    self.forward_foot_cycle = None
    if "forward_foot_cycle" in env.cfg.rewards:
      foot_cycle = env.unwrapped.reward_manager.get_term_cfg("forward_foot_cycle")
      assert isinstance(foot_cycle.func, ForwardFootCycleReward)
      self.forward_foot_cycle = foot_cycle.func
      self.forward_foot_cycle_weight = foot_cycle.weight

  def _on_reset(self, batch: TensorDict) -> None:
    self.command = batch["velocity_command"]
    self.steps_left = batch["velocity_hold_steps"]
    self.category = batch["velocity_command_category"]
    self.gait_phase = batch["gait_phase"] if self.use_gait_phase else None
    self._reset_sensors(batch)

  def _score_transition(
    self, next_aux_state, next_rep_state, next_joint_state, action, last_action
  ) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, linear_b, angular_b, q, qd = decompose_rep_state(next_rep_state)
    quat = next_aux_state[..., 3:7]
    linear_w, angular_w = quat_apply(quat, linear_b), quat_apply(quat, angular_b)
    reward = velocity_reward(quat, linear_w, angular_w, self.command)
    if self.forward_precision_weight:
      reward = reward + self.forward_precision_weight * forward_precision_reward(
        quat, linear_w, angular_w, self.command
      )
    if self.turn_tracking_weight:
      reward = reward + self.turn_tracking_weight * turn_tracking_reward(
        angular_w[..., 2], self.command
      )
    if self.steering_recovery_weight:
      reward = reward + self.steering_recovery_weight * steering_recovery_reward(
        quat, linear_w, angular_w, self.command
      )
    height = next_aux_state[..., 2]
    if self.posture is not None:
      reward = reward + self.posture_weight * self.posture.score(
        height,
        quat,
        linear_w,
        angular_w,
        q,
        qd,
        self.command,
        self.moving_posture_scale,
      )
    if self.arm_swing is not None:
      reward = reward + self.arm_swing_weight * self.arm_swing.score(q, self.command)
    next_phase = None
    if self.gait_phase is not None:
      next_phase = advance_gait_phase(self.gait_phase, self.command, self.dt)
    if self.forward_gait is not None:
      reward = reward + self.forward_gait_weight * self.forward_gait.score(
        height, q, qd, self.command, next_phase, quat, linear_w
      )
    if self.forward_foot_cycle is not None:
      assert next_phase is not None
      reward = reward + self.forward_foot_cycle_weight * self.forward_foot_cycle.score(
        height, quat, linear_w, angular_w, q, qd, self.command, next_phase
      )
    severity = fall_severity(height, quat)
    done = torch.sigmoid((severity - 1.0) / self._termination_softness)
    return done, reward * self.dt

  def _compute_commands(self) -> None:
    if self.gait_phase is not None:
      self.gait_phase = advance_gait_phase(self.gait_phase, self.command, self.dt)
    self.command, self.steps_left, self.category = self.schedule.advance(
      self.command, self.steps_left, self.category
    )

  def get_observations(self) -> TensorDict:
    hold_seconds = self.steps_left[:, None] * self.dt
    critic_state = (self.rep_state, self.command, hold_seconds, self.sim_ctx)
    if self.gait_phase is not None:
      critic_state += (gait_phase_features(self.gait_phase),)
    critic = torch.cat(critic_state, dim=-1)
    obs = build_actor_obs_td({"velocity": self.command}, self._sensor_history(), critic)
    if self.gait_phase is not None:
      obs["gait_phase"] = self.gait_phase
    return obs
