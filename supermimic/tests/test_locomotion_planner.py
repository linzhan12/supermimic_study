import torch

from supermimic.locomotion_planner import (
  PREVIEW_STEPS,
  LocomotionPlanner,
  ReferenceClip,
  ReferenceWindow,
)
from supermimic.reference_trackability import projected_twist
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


def test_start_can_blend_arms_slower_when_measured_leg_support_matches():
  old = window(1)
  walk_frames = window(90)
  tw = walk_frames.q.new_zeros((90, 3))
  walk = ReferenceClip("walk", "walk", walk_frames, tw, torch.zeros(3))
  start = window(90, 0.45)
  start.q[:, :12] = 0.10
  entry = ReferenceClip("entry", "start", start, tw, torch.zeros(3))
  planner = LocomotionPlanner([entry, walk], old, 0.02, start_legs=tuple(range(12)))

  planner.update(torch.tensor([0.35, 0.0, 0.0]), torch.tensor([True, True]))
  preview = planner.preview()
  assert planner.clip is entry
  torch.testing.assert_close(preview.q[8, :12], start.q[8, :12])
  torch.testing.assert_close(preview.q[8, 12:], start.q[8, 12:] * 0.5)
  torch.testing.assert_close(preview.q[16], start.q[16])


def test_direction_filter_uses_signed_requested_turn():
  old = window(1)
  frames = window(90)
  left_twist = torch.tensor([0.35, 0.0, 0.4]).expand(90, -1)
  right_twist = torch.tensor([0.35, 0.0, -0.4]).expand(90, -1)
  left = ReferenceClip("left", "walk", frames, left_twist, torch.zeros(3))
  right = ReferenceClip("right", "walk", frames, right_twist, torch.zeros(3))
  planner = LocomotionPlanner([left, right], old, 0.02, directional_selection=True)
  planner.update(left_twist[0], torch.tensor([True, True]))
  assert planner.clip is left
  assert [candidate["clip"] for candidate in planner.last_costs] == ["left"]


def test_predicted_twist_uses_command_heading_and_wrapped_yaw():
  previous = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
  current = torch.tensor([[-0.01, 0.0, 0.0, 0.01, 0.0, 0.0, -1.0]])
  current[:, 3:] = torch.nn.functional.normalize(current[:, 3:], dim=-1)
  twist = projected_twist(previous, current, 0.02)
  torch.testing.assert_close(twist, torch.tensor([[0.5, 0.0, 1.0]]), atol=0.001, rtol=0)


def test_changed_lateral_command_can_end_compatible_start_after_its_seam():
  initial = window(1)
  frames = window(90)
  left = torch.tensor([0.25, 0.2, 0.0])
  right = torch.tensor([0.25, -0.2, 0.0])
  bias = torch.zeros(3)
  launch = ReferenceClip(
    "launch", "start", frames, left.expand(90, -1), bias, exit_twist=left
  )
  walk = ReferenceClip("right", "walk", frames, right.expand(90, -1), bias)
  planner = LocomotionPlanner([launch, walk], initial, 0.02, restart_from_rest=True)
  support = torch.tensor([True, True])
  planner.update(left, support)
  assert planner.clip is launch
  for _ in range(10):
    planner.advance()
  planner.update(right, support)
  assert planner.clip is walk


def test_heading_guide_preserves_joint_goals_and_is_inactive_for_straight_commands():
  reference = window(PREVIEW_STEPS)
  aux = torch.cat((reference.p[0], reference.quat[0]))[None]
  steps = torch.zeros(1, dtype=torch.long)
  plain = reference.actor_goal(steps, aux)
  guided = reference.actor_goal(steps, aux, yaw_rate=0.4, heading_strength=0.58)
  content = NUM_GOAL_FRAMES * 2 * reference.q.shape[1]
  torch.testing.assert_close(guided[:, :content], plain[:, :content], atol=0, rtol=0)
  assert not torch.equal(guided[:, content:], plain[:, content:])
  torch.testing.assert_close(
    reference.actor_goal(steps, aux, heading_strength=0.58), plain, atol=0, rtol=0
  )


def test_generated_step_allows_next_direction_when_feet_return_to_rest():
  initial = window(1)
  walk_frames = window(90)
  tw = walk_frames.q.new_zeros((90, 3))
  walk = ReferenceClip("walk", "walk", walk_frames, tw, torch.zeros(3))
  planner = LocomotionPlanner([walk], initial, 0.02, restart_from_rest=True)
  support = torch.tensor([True, True])
  first, second = window(21, 0.1), window(21, -0.1)
  planner.update(
    torch.tensor([0.25, 0.2, 0.0]),
    support,
    observed_twist=torch.zeros(3),
    contact_launch=lambda: [("first", first)],
  )
  assert planner.plan is first
  for _ in range(20):
    planner.advance()
  planner.update(
    torch.tensor([0.25, -0.2, 0.0]),
    support,
    observed_twist=torch.zeros(3),
    contact_launch=lambda: [("second", second)],
  )
  assert planner.plan is second


def test_generated_launches_are_padded_before_world_model_ranking():
  old = window(1)
  frames = window(90)
  twist = frames.q.new_zeros((90, 3))
  walk = ReferenceClip("walk", "walk", frames, twist, torch.zeros(3))
  planner = LocomotionPlanner(
    [walk], old, 0.02, restart_from_rest=True, trackability_weight=1.0
  )
  first, second = window(21), window(25, 0.12)

  def cost(candidates):
    assert all(len(candidate.q) >= PREVIEW_STEPS for candidate in candidates)
    return torch.tensor([0.9, 0.4])

  planner.update(
    torch.tensor([0.25, -0.2, 0.0]),
    torch.tensor([True, True]),
    cost,
    observed_twist=torch.zeros(3),
    contact_launch=lambda: [("first", first), ("second", second)],
  )
  assert planner.plan is second
  assert planner.last_costs[-1]["selected"]
