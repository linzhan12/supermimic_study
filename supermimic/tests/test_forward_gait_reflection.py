import torch

from supermimic.tasks.forward_gait import G1SagittalReflection


def test_reflection_applies_to_joint_targets_before_actuator_normalization():
  names = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
  )
  scale = torch.tensor((0.25, 0.4, 0.24, 0.35))
  offset = torch.tensor((-0.35, 0.09, -0.32, -0.04))
  reflection = G1SagittalReflection(list(names), names, scale, offset, blend=1.0)
  action = torch.tensor([[0.5, -0.1, -0.2, 0.3]])
  q = action * scale + offset
  signs = torch.tensor((1.0, -1.0, 1.0, -1.0))
  torch.testing.assert_close(
    reflection.action(action) * scale + offset, q[:, [2, 3, 0, 1]] * signs
  )
  torch.testing.assert_close(reflection.action(reflection.action(action)), action)
  state = torch.arange(18, dtype=torch.float32).reshape(1, 1, 18)
  torch.testing.assert_close(reflection.state(reflection.state(state)), state)
  torch.testing.assert_close(
    reflection.weight(
      torch.tensor([[0.35, 0.0, 0.0], [0.35, 0.0, 0.40], [0.0, 0.0, 0.0]])
    ),
    torch.tensor([[0.5], [0.0], [0.0]]),
  )
