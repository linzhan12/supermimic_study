import torch

from supermimic.locomotion_planner import (
  PREVIEW_STEPS,
  LocomotionPlanner,
  ReferenceClip,
  ReferenceWindow,
)
from supermimic.representations import (
  GOAL_FRAME_STRIDE,
  NUM_GOAL_FRAMES,
  _assemble_goal,
)


def window(length: int, angle: float = 0.0) -> ReferenceWindow:
  q = torch.full((length, 29), angle)
  p = torch.zeros(length, 3)
  p[:, 2] = 0.78
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(length, -1)
  feet = torch.tensor([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]).expand(length, -1, -1)
  return ReferenceWindow(
    q, torch.zeros_like(q), p, quat, feet, torch.ones(length, 2, dtype=torch.bool)
  )


def test_planned_goals_match_dense_reference_layout():
  reference = window(PREVIEW_STEPS)
  reference.q[:, 0] = torch.arange(PREVIEW_STEPS) * 0.01
  aux = torch.cat((reference.p[0], reference.quat[0]))[None]
  indices = torch.arange(NUM_GOAL_FRAMES) * GOAL_FRAME_STRIDE
  content = torch.cat((reference.q[indices], reference.qd[indices]), dim=-1).reshape(
    1, -1
  )
  _, expected = _assemble_goal(
    content,
    aux[:, :3],
    aux[:, 3:],
    reference.p[None, indices],
    reference.quat[None, indices],
  )
  goal = reference.actor_goal(torch.zeros(1, dtype=torch.long), aux)
  torch.testing.assert_close(goal, expected, atol=0, rtol=0)
  assert goal.shape == (1, 640)


def test_replan_preserves_pose_and_velocity_at_the_current_step():
  old = window(1)
  motion = window(90, 0.1)
  twist = torch.zeros((90, 3))
  twist[:, 0] = 0.22
  clip = ReferenceClip("walk", "walk", motion, twist, torch.zeros(3))
  planner = LocomotionPlanner([clip], old, 0.02)
  planner.update(torch.tensor([0.22, 0.0, 0.0]), torch.tensor([True, True]))
  assert planner.clip is clip
  preview = planner.preview()
  torch.testing.assert_close(preview.q[0], old.q[0])
  torch.testing.assert_close(preview.qd[0], old.qd[0])
  torch.testing.assert_close(preview.q[planner.blend_steps], motion.q[0])
  torch.testing.assert_close(preview.qd[planner.blend_steps], motion.qd[0])


def test_reference_hold_does_not_keep_a_stale_joint_velocity():
  old = window(1)
  old.qd.fill_(0.2)
  held = old.hold_after(0, PREVIEW_STEPS)
  torch.testing.assert_close(held.qd[0], old.qd[0])
  assert not held.qd[1:].any()
