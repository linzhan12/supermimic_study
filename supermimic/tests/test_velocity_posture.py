import torch

from supermimic.tasks.velocity_posture import arm_swing_score, lower_body_score


def test_forward_arm_reward_favors_moderate_opposite_swing():
  command = torch.tensor([[0.35, 0.0, 0.0], [0.35, 0.0, 0.0]])
  hips = torch.tensor([[0.25, -0.25], [-0.25, 0.25]], requires_grad=True)
  arms = torch.tensor([[-0.15, 0.15], [0.15, -0.15]], requires_grad=True)
  preferred = arm_swing_score(arms, hips, command)
  still = arm_swing_score(torch.zeros_like(arms), hips, command)
  reversed_arms = arm_swing_score(-arms, hips, command)
  assert torch.all(preferred > still)
  assert torch.all(still > reversed_arms)
  torch.testing.assert_close(
    arm_swing_score(arms, hips, torch.zeros_like(command)), torch.zeros(2)
  )
  preferred.sum().backward()
  assert arms.grad is not None and arms.grad.abs().max() > 0
  assert hips.grad is None


def test_zero_command_rewards_settled_pose_and_double_support():
  cmd = torch.zeros((1, 3))
  joint_error = torch.zeros((1, 29))
  foot_position = torch.zeros((1, 2, 3))
  foot_velocity = torch.zeros_like(foot_position)
  yaw = torch.zeros(1)

  settled = lower_body_score(joint_error, foot_position, foot_velocity, cmd, yaw, cmd)
  moving = lower_body_score(
    joint_error,
    foot_position,
    foot_velocity,
    torch.tensor([[0.30, 0.0, 0.0]]),
    yaw,
    cmd,
  )
  foot_position[:, 1, 2] = 0.12
  swinging = lower_body_score(joint_error, foot_position, foot_velocity, cmd, yaw, cmd)
  assert settled.item() > 0.9
  assert moving.item() < 0.01
  assert swinging.item() < 0.01


def test_moving_command_rewards_stable_support_without_static_leg_target():
  cmd = torch.tensor([[0.35, 0.0, 0.0]])
  joint_error = torch.zeros((1, 29))
  foot_position = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.16]]])
  foot_velocity = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
  yaw = torch.zeros(1)

  planted = lower_body_score(joint_error, foot_position, foot_velocity, cmd, yaw, cmd)
  torch.testing.assert_close(
    lower_body_score(joint_error + 0.5, foot_position, foot_velocity, cmd, yaw, cmd),
    planted,
  )
  foot_velocity[:, 0, 0] = 0.3
  slipping = lower_body_score(joint_error, foot_position, foot_velocity, cmd, yaw, cmd)
  assert planted.item() > 0.95
  assert slipping.item() < 0.15
