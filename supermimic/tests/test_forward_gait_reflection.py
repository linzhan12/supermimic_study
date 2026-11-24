import torch

from supermimic.tasks.forward_gait import ForwardGaitAdapter, G1SagittalReflection


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


def test_expanding_walking_adapter_keeps_the_old_corrections_and_initializes_extra_joints():
  joints = (
    "hip_pitch",
    "hip_roll",
    "hip_yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
    "shoulder_pitch",
    "shoulder_roll",
    "elbow",
  )
  names = tuple(
    f"{side}_{joint}_joint" for side in ("left", "right") for joint in joints
  )
  names += ("waist_pitch_joint", "left_wrist_roll_joint", "right_wrist_roll_joint")
  scale = torch.linspace(0.15, 0.35, len(names))
  legacy = ForwardGaitAdapter(68, names, scale, 0.6, use_phase=True)
  expanded = ForwardGaitAdapter(68, names, scale, 0.6, use_phase=True, full_body=True)
  with torch.no_grad():
    legacy.net[-1].weight.normal_(std=0.2)
    legacy.net[-1].bias.normal_(std=0.2)
  checkpoint = legacy.state_dict()
  expanded.expand_checkpoint(checkpoint, "")
  expanded.load_state_dict(checkpoint)
  action = torch.randn(3, len(names))
  observation = torch.randn(3, 68)
  command = torch.tensor([[0.22, 0.0, 0.0], [0.35, 0.0, 0.0], [0.5, 0.0, 0.0]])
  phase = torch.tensor([0.25, 0.5, 0.75])
  torch.testing.assert_close(
    expanded(action, observation, command, phase),
    legacy(action, observation, command, phase),
  )
