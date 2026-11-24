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


def test_replanning_does_not_blend_an_unchanged_reference_again():
  old = window(1)
  motion = window(120, 0.10)
  motion.q[:, 0] = 0.10 + torch.arange(120) * 0.004
  motion.qd[:, 0] = 0.20
  twist = motion.q.new_zeros((120, 3))
  twist[:, 0] = 0.22
  clip = ReferenceClip("walk", "walk", motion, twist, torch.zeros(3))
  planner = LocomotionPlanner([clip], old, 0.02)
  target = torch.tensor([0.22, 0.0, 0.0])
  support = torch.tensor([True, True])
  planner.update(target, support)
  planned = planner.plan
  for _ in range(2 * planner.replan_steps):
    planner.advance()
    planner.update(target, support)
  assert planner.plan is planned
  assert planner.offset == 2 * planner.replan_steps
  torch.testing.assert_close(planner.preview().q[0], motion.q[planner.offset])


def test_trackability_ranking_does_not_select_an_incompatible_pace():
  old = window(1)
  forward = window(120)
  twist = forward.q.new_zeros((120, 3))
  twist[:, 0] = 0.35
  medium = ReferenceClip("medium", "walk", forward, twist, torch.zeros(3))
  slow = ReferenceClip("slow", "walk", forward, twist * 0.55, torch.zeros(3))
  planner = LocomotionPlanner([slow, medium], old, 0.02, pace_margin=0.25)
  target = torch.tensor([0.35, 0.0, 0.0])
  planner.filtered_command.copy_(target)
  planner.update(target, torch.tensor([True, True]))
  assert planner.clip is medium
  assert [candidate["clip"] for candidate in planner.last_costs] == ["medium"]


def test_recovery_waits_for_two_measured_support_feet_and_finishes_its_plan():
  initial = window(1)
  moving = window(120)
  twist = moving.q.new_zeros((120, 3))
  walk = ReferenceClip("walk", "walk", moving, twist, torch.zeros(3))
  planner = LocomotionPlanner([walk], initial, 0.02)
  planner.mode = "walk"
  stopped = window(21, 0.15)
  command = torch.zeros(3)
  planner.update(command, torch.tensor([True, False]), stop_fallback=lambda: stopped)
  assert planner.mode == "walk"

  support = torch.tensor([True, True])
  planner.update(command, support, stop_fallback=lambda: stopped)
  assert planner.mode == "stop"
  for _ in stopped.q[1:]:
    planner.advance()
    planner.update(command, support, stop_fallback=lambda: stopped)

  assert planner.mode == "stand"
  assert planner.plan is stopped
  assert not planner.preview().qd.any()
