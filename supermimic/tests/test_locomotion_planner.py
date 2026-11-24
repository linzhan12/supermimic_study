import mujoco
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from supermimic.contact_tracking import JOINTS, ContactTrackingAdapter
from supermimic.locomotion_planner import (
  PREVIEW_STEPS,
  LocomotionPlanner,
  ReferenceClip,
  ReferenceWindow,
  reference_actor_goal,
)
from supermimic.reference_trackability import projected_twist
from supermimic.representations import (
  GOAL_FRAME_STRIDE,
  NUM_GOAL_FRAMES,
  _assemble_goal,
)
from supermimic.stance_bridge import StanceBridge


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


def test_measured_cycle_ingress_can_ablate_ranking_without_changing_the_seam():
  measured = window(1)
  target = torch.tensor([0.25, 0.20, 0.0])
  twist = target.expand(120, -1)
  zero = torch.zeros(3)
  first = ReferenceClip("first", "walk", window(120, 0.02), twist, zero, 60)
  second = ReferenceClip("second", "walk", window(120, 0.12), twist, zero, 60)
  start = ReferenceClip("start", "start", window(120), twist, zero, None, target)
  for ranking, expected in ((False, first), (True, second)):
    planner = LocomotionPlanner(
      [first, second, start],
      measured,
      0.02,
      directional_selection=True,
      restart_from_rest=True,
      trackability_weight=1.0,
    )
    planner.mode = "stop"
    planner.update(
      target,
      measured.contact[0],
      lambda windows: torch.tensor([float(w.q[10, 0] < 0.1) for w in windows]),
      observed_twist=zero,
      cycle_ingress_anchor=measured,
      rank_cycle_ingress=ranking,
    )
    assert planner.clip is expected
    assert all(entry["measured_cycle_ingress"] for entry in planner.last_costs)
    torch.testing.assert_close(planner.plan.q[0], measured.q[0], atol=0, rtol=0)


def test_cycle_ingress_can_blend_legs_before_a_larger_upper_body_change():
  measured = window(1)
  motion = window(120, 0.45)
  legs = tuple(range(12))
  motion.q[:, legs] = 0.12
  target = torch.tensor([0.25, 0.20, 0.0])
  zero = torch.zeros(3)
  twist = target.expand(120, -1)
  cycle = ReferenceClip("walk", "walk", motion, twist, zero, 60)
  start = ReferenceClip("start", "start", window(120), twist, zero, 0, target)
  planner = LocomotionPlanner(
    [cycle, start],
    measured,
    0.02,
    directional_selection=True,
    restart_from_rest=True,
    start_legs=legs,
    start_upper_blend=0.48,
  )
  planner.mode = "stop"
  planner.update(
    target,
    measured.contact[0],
    observed_twist=zero,
    cycle_ingress_anchor=measured,
    split_cycle_ingress=True,
  )
  assert planner.clip is cycle
  assert planner.committed_steps == planner.start_blend_steps
  position = planner.plan.q[planner.blend_steps]
  torch.testing.assert_close(position[:12], motion.q[0, :12])
  assert bool((position[12:] < motion.q[0, 12:]).all())


def test_axial_continuation_is_eligible_during_a_reference_contact_gap():
  old = window(1)
  frames = window(120)
  pace = torch.tensor([0.35, 0.0, 0.0])
  clip = ReferenceClip("period", "walk", frames, pace.expand(120, -1), torch.zeros(3))
  planner = LocomotionPlanner([clip], old, 0.02, preserve_axial_phase=True)
  support = torch.tensor([True, True])
  planner.update(pace, support)
  assert planner.clip is clip
  plan = planner.plan
  frames.contact[10] = False
  for _ in range(10):
    planner.advance()
  planner.update(pace, support)
  assert [choice["frame"] for choice in planner.last_costs] == [10]
  assert planner.plan is plan


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


def test_capture_stop_projects_a_measured_support_foot_and_lands_the_other():
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  root = torch.arange(7)
  joints = torch.as_tensor(model.jnt_qposadr[1:].copy())
  sites = torch.tensor([model.site(f"{side}_foot").id for side in ("left", "right")])
  suffix = {"hip_pitch_joint": -0.10, "knee_joint": 0.30, "ankle_pitch_joint": -0.20}
  names = [model.joint(i).name for i in range(1, model.njnt)]
  rest = torch.tensor([suffix.get(name.split("_", 1)[-1], 0.0) for name in names])
  legs = tuple(
    i
    for i, name in enumerate(names)
    if any(p in name for p in ("hip", "knee", "ankle"))
  )
  data = mujoco.MjData(model)
  data.qpos[:7] = (0, 0, 0.765, 1, 0, 0, 0)
  data.qpos[joints.numpy()] = rest.numpy()
  mujoco.mj_kinematics(model, data)
  measured = ReferenceWindow(
    rest[None],
    torch.zeros_like(rest[None]),
    torch.tensor(data.qpos[None, :3], dtype=torch.float32),
    torch.tensor(data.qpos[None, 3:7], dtype=torch.float32),
    torch.tensor(data.site_xpos[sites.numpy()][None], dtype=torch.float32),
    torch.tensor([[False, True]]),
  )
  bridge = StanceBridge(model, root, joints, legs, sites, rest, 0.02, 0.4)
  velocity = torch.tensor([0.15, 0.0, 0.0])
  plan = bridge.capture_stop(measured, velocity, torch.zeros(2, 3), 0.25)
  assert plan is not None
  assert torch.equal(plan.contact[0], measured.contact[0])
  assert bool(plan.contact[-1].all()) and bool(plan.contact[:, 1].all())
  assert plan.feet[-1, 0, 1] > plan.feet[-1, 1, 1]
  for q, p, quat in zip(plan.q, plan.p, plan.quat, strict=True):
    data.qpos[joints.numpy()], data.qpos[:3], data.qpos[3:7] = (
      q.numpy(),
      p.numpy(),
      quat.numpy(),
    )
    mujoco.mj_kinematics(model, data)
    actual = torch.tensor(data.site_xpos[sites[1]], dtype=torch.float32)
    assert (actual - measured.feet[0, 1]).norm() < 0.004


def test_contact_stand_refreshes_only_on_measured_drift_after_generated_stop():
  initial = window(1)
  frames = window(90)
  twist = frames.q.new_zeros((90, 3))
  walk = ReferenceClip("walk", "walk", frames, twist, torch.zeros(3))
  planner = LocomotionPlanner([walk], initial, 0.02, refresh_contact_stand=True)
  first, refresh = window(11, 0.1), window(11, 0.2)
  command = torch.zeros(3)
  support = torch.tensor([True, True])
  drifting = torch.tensor([0.06, 0.0, 0.0])
  planner.update(command, support, observed_twist=drifting, stop_fallback=lambda: first)
  assert planner.mode == "stand"
  planner.mode = "walk"
  planner.update(command, support, stop_fallback=lambda: first)
  for _ in first.q[1:]:
    planner.advance()
    planner.update(
      command, support, observed_twist=command, stop_fallback=lambda: refresh
    )
  assert planner.mode == "stand"
  assert planner.plan is first
  planner.update(
    command, support, observed_twist=drifting, stop_fallback=lambda: refresh
  )
  assert planner.mode == "stop"
  assert planner.plan is refresh


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


def test_axial_cycle_ranking_retains_multiple_steering_entries():
  old = window(1)
  frames = window(90)
  axial = torch.tensor([0.35, 0.0, 0.0])
  turn = torch.tensor([0.35, 0.0, 0.4])
  bias = torch.zeros(3)
  walk = ReferenceClip("straight", "walk", frames, axial.expand(90, -1), bias)
  left = ReferenceClip("left", "walk", frames, turn.expand(90, -1), bias)
  support = torch.tensor([True, True])
  for target, expected in ((axial, 1), (turn, 3)):
    planner = LocomotionPlanner(
      [walk, left],
      old,
      0.02,
      directional_selection=True,
      candidate_entries=3,
      axial_cycle_ranking=True,
    )
    planner.update(target, support)
    assert len(planner.last_costs) == expected


def test_changed_steering_direction_joins_at_measured_support_state():
  old = window(1)
  old.q.fill_(0.8)
  measured = window(1, 0.1)
  frames = window(90, 0.1)
  turn = torch.tensor([0.35, 0.0, 0.4])
  left = ReferenceClip("left", "walk", frames, turn.expand(90, -1), torch.zeros(3))
  planner = LocomotionPlanner([left], old, 0.02, directional_selection=True)
  support = torch.tensor([True, True])
  planner.update(turn, support, measured_steering_anchor=measured)
  assert planner.clip is left
  torch.testing.assert_close(planner.preview().q[0], measured.q[0])


def test_split_steering_requires_matching_measured_support_for_a_large_arm_gap():
  measured = window(1)
  frames = window(90, 0.5)
  frames.q[:, :12] = 0.08
  turn = torch.tensor([0.25, -0.2, 0.0])
  swing = torch.tensor([True, False])
  frames.contact[:] = swing
  measured.contact[:] = swing
  clip = ReferenceClip("right", "walk", frames, turn.expand(90, -1), torch.zeros(3))
  planner = LocomotionPlanner(
    [clip],
    measured,
    0.02,
    start_legs=tuple(range(12)),
    start_upper_blend=0.48,
    directional_selection=True,
    steering_split_seam=True,
  )
  planner.mode = "walk"
  planner.update(turn, torch.tensor([True, True]), measured_steering_anchor=measured)
  assert planner.clip is None
  planner.update(turn, swing, measured_steering_anchor=measured)
  assert planner.clip is clip
  assert planner.committed_steps == planner.blend_steps
  preview = planner.preview()
  torch.testing.assert_close(preview.q[8, :12], frames.q[0, :12])
  assert bool((preview.q[8, 12:] < frames.q[0, 12:]).all())


def test_support_projection_is_ranked_as_an_alternative_to_the_same_seam():
  old = window(1)
  frames = window(90)
  turn = torch.tensor([0.25, -0.2, 0.0])
  clip = ReferenceClip("right", "walk", frames, turn.expand(90, -1), torch.zeros(3))
  planner = LocomotionPlanner(
    [clip],
    old,
    0.02,
    directional_selection=True,
    rank_contact_seam=True,
    trackability_weight=1.0,
  )
  shifted = window(90, 0.12)
  planner.update(
    turn,
    torch.tensor([True, True]),
    lambda windows: torch.tensor([0.2, 0.1]),
    measured_steering_anchor=old,
    project_seam=lambda window: shifted,
  )
  assert planner.plan is shifted
  assert [row["projected_support"] for row in planner.last_costs] == [False, True]
  assert [row["selected"] for row in planner.last_costs] == [False, True]


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


def test_batched_reference_goals_match_execution_for_mixed_yaw_commands():
  ref = window(PREVIEW_STEPS)
  offsets = GOAL_FRAME_STRIDE * torch.arange(NUM_GOAL_FRAMES)
  aux = torch.cat((ref.p[0], ref.quat[0]))[None].expand(2, -1)
  expected = torch.cat(
    [
      ref.actor_goal(
        torch.zeros(1, dtype=torch.long),
        aux[i : i + 1],
        yaw_rate=yaw,
        heading_strength=0.58,
      )
      for i, yaw in enumerate((0.0, -0.4))
    ]
  )
  batch = reference_actor_goal(
    *(x[offsets][None].expand(2, -1, -1) for x in (ref.q, ref.qd, ref.p, ref.quat)),
    aux,
    offsets,
    yaw_rate=torch.tensor([[0.0], [-0.4]]),
    heading_strength=torch.full((2, 1), 0.58),
    dt=0.02,
  )
  torch.testing.assert_close(batch, expected, rtol=0, atol=2e-7)


def test_diagonal_heading_anchor_works_without_a_yaw_rate_command():
  ref = window(PREVIEW_STEPS)
  aux = torch.cat((ref.p[0], ref.quat[0]))[None]
  anchor = torch.tensor([[0.9950042, 0.0, 0.0, 0.0998334]])
  steps = torch.zeros(1, dtype=torch.long)
  plain = ref.actor_goal(steps, aux)
  locked = ref.actor_goal(steps, aux, heading_strength=0.58, heading_anchor=anchor)
  content = NUM_GOAL_FRAMES * 2 * ref.q.shape[-1]
  torch.testing.assert_close(plain[:, :content], locked[:, :content], rtol=0, atol=0)
  assert not torch.equal(plain[:, content:], locked[:, content:])


def test_contact_adapter_preserves_unselected_actions_and_bounds_physical_radians():
  names = (*JOINTS, "left_knee_joint", "left_elbow_joint")
  action_names = tuple(reversed(names))
  scale = torch.linspace(0.2, 0.6, len(names))
  adapter = ContactTrackingAdapter(names, action_names, scale, 0.06)
  original = torch.ones(1, len(names))
  state = torch.zeros(1, 10 + 2 * len(names))
  aux = torch.tensor([[0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]])
  command, q = torch.zeros(1, 3), torch.zeros(1, len(names))
  contact = torch.ones(1, 2, dtype=torch.bool)
  inputs = (original, state, aux, command, q, q, contact, contact)
  unchanged, _ = adapter(*inputs)
  torch.testing.assert_close(unchanged, original, rtol=0, atol=0)
  with torch.no_grad():
    adapter.head[-1].bias.fill_(100)
  out, delta = adapter(*inputs, gain=0.5)
  torch.testing.assert_close(out[:, :2], original[:, :2], rtol=0, atol=0)
  torch.testing.assert_close((out - original)[:, 2:] * scale[2:], delta)
  assert bool((delta.abs() <= 0.03).all())


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


def test_axial_generated_step_hands_off_instead_of_repeating_when_speed_is_low():
  initial = window(1)
  forward = torch.tensor([0.35, 0.0, 0.0])
  frames = window(90, 0.08)
  clip = ReferenceClip("walk", "walk", frames, forward.expand(90, -1), torch.zeros(3))
  planner = LocomotionPlanner(
    [clip],
    initial,
    0.02,
    restart_from_rest=True,
    contact_launch_axial=True,
  )
  start = window(21, 0.04)
  support = torch.tensor([True, True])

  def launches():
    return [("first-step", start)]

  planner.update(
    forward, support, observed_twist=torch.zeros(3), contact_launch=launches
  )
  assert planner.plan is start
  for _ in range(20):
    planner.advance()
    planner.update(
      forward, support, observed_twist=torch.zeros(3), contact_launch=launches
    )
  assert planner.clip is clip
