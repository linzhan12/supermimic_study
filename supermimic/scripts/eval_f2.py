"""Evaluate offline G1 motion tracking and joystick controls in MuJoCo."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import tyro
from f2_prediction_audit import PredictionAudit
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.contact_tracking import ContactTrackingAdapter
from supermimic.locomotion_planner import (
  LocomotionPlanner,
  ReferenceClip,
  ReferenceWindow,
  heading_twist,
)
from supermimic.reference_trackability import predict_tracking_cost, projected_twist
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.stance_bridge import StanceBridge
from supermimic.tasks.task_spec import VelocityTaskCfg, build_actor_obs_td

STAND_START = (
  ("stand", 1.0, (0.0, 0.0, 0.0)),
  ("forward", 1.5, (0.35, 0.0, 0.0)),
  ("left_turn", 1.0, (0.35, 0.0, 0.4)),
  ("right_turn", 1.0, (0.35, 0.0, -0.4)),
  ("release", 2.0, (0.0, 0.0, 0.0)),
  ("left_diagonal", 1.0, (0.25, 0.2, 0.0)),
  ("right_diagonal", 1.0, (0.25, -0.2, 0.0)),
  ("release_after_diagonal", 2.0, (0.0, 0.0, 0.0)),
)
WALK_START = (
  ("forward", 1.0, (0.35, 0.0, 0.0)),
  ("release", 2.0, (0.0, 0.0, 0.0)),
)
STEADY_FORWARD = (
  ("stand", 1.0, (0.0, 0.0, 0.0)),
  ("forward", 6.0, (0.35, 0.0, 0.0)),
)
MULTISPEED_PROFILE = (
  ("stand", 1.0, (0.0, 0.0, 0.0)),
  ("slow", 3.0, (0.22, 0.0, 0.0)),
  ("fast", 3.0, (0.50, 0.0, 0.0)),
  ("medium", 3.0, (0.35, 0.0, 0.0)),
  ("release", 2.0, (0.0, 0.0, 0.0)),
)
ROUTES = ("motion", "reference_velocity", "scripted_velocity")
FALL_HEIGHT_M = 0.45
FALL_TILT_RAD = math.pi / 3


@dataclass
class Config:
  checkpoint_dir: Path
  motion_file: Path
  output_dir: Path
  stand_frame: int = 0
  walk_frame: int = 205
  seed: int = 0
  device: str = "cpu"
  task_id: str = "SuperMimic-GenCoord-G1"
  record_posture: bool = False
  steady_forward: bool = False
  extra_forward_speeds: tuple[float, ...] = ()
  multispeed_profile: bool = False
  motion_only: bool = False
  motion_window_s: float = 4.0
  motion_frames: tuple[int, ...] = ()
  reference_bank: Path | None = None
  planned_only: bool = False
  cases: tuple[str, ...] = ()
  trackability_weight: float = 0.0
  prediction_steps: int = 10
  prediction_audit_stride: int = 0
  fixed_plan_dir: Path | None = None
  contact_adapter: Path | None = None
  adapter_gain: float = 1.0
  adapt_ranking: bool = False
  adapter_probe_phases: tuple[str, ...] = ()
  pace_margin: float | None = None
  stop_double_support_pose: float | None = None
  contact_stop_duration: float | None = None
  contact_stop_max_foot_speed: float | None = None
  refresh_contact_stand: bool = False
  start_upper_blend: float | None = None
  directional_selection: bool = False
  direct_steering_start: bool = False
  command_cost_weight: float = 0.0
  candidate_entries: int = 1
  axial_cycle_ranking: bool = False
  steering_measured_seam: bool = False
  start_search_time: float | None = None
  restart_from_rest: bool = False
  heading_strength: float = 0.0
  contact_launch_duration: float | None = None
  contact_launch_lean: float = 0.0
  contact_launch_forward_lead: float = 0.0
  rank_contact_launch: bool = False
  strict_contact: bool = True


def planned_twist(window: ReferenceWindow, dt: float) -> torch.Tensor:
  root = torch.cat((window.p[:2], window.quat[:2]), dim=-1)
  return projected_twist(root[:1], root[1:], dt)[0]


def reference_velocity(motion: MotionLoader, window: int) -> torch.Tensor:
  """Smooth within the single evaluation clip before projecting by reference yaw."""
  linear, angular = motion.root_lin_vel_w, motion.root_ang_vel_w
  if window > 1:
    frames = torch.arange(len(linear), device=linear.device)
    offsets = torch.arange(window, device=linear.device) - window // 2
    indices = (frames[:, None] + offsets).clamp(0, len(linear) - 1)
    linear, angular = linear[indices].mean(1), angular[indices].mean(1)
  return heading_twist(motion.root_quat_w, linear, angular)


def velocity_limits(cfg: VelocityTaskCfg, like: torch.Tensor) -> torch.Tensor:
  return like.new_tensor(
    (cfg.lin_vel_x_range, cfg.lin_vel_y_range, cfg.ang_vel_z_range)
  ).T


def arm_swing_metrics(rows: list[dict], joint_names: tuple[str, ...]) -> dict:
  names = tuple(
    f"{side}_{joint}_pitch_joint"
    for joint in ("shoulder", "hip")
    for side in ("left", "right")
  )
  columns = [joint_names.index(name) for name in names]
  joints = torch.tensor([r["posture"]["joint_pos_rad"] for r in rows])[:, columns]
  arms, hips = joints.split(2, dim=-1)
  amplitude = torch.quantile(arms, 0.95, dim=0) - torch.quantile(arms, 0.05, dim=0)
  phase_signals = torch.stack(
    (arms[:, 0], arms[:, 1], arms[:, 0] - arms[:, 1], hips[:, 0] - hips[:, 1])
  )
  corr = torch.nan_to_num(torch.corrcoef(phase_signals), nan=0.0)
  return {
    "steps": len(rows),
    "shoulder_pitch_excursion_deg": torch.rad2deg(amplitude).tolist(),
    "left_right_correlation": corr[0, 1].item(),
    "arm_hip_difference_correlation": corr[2, 3].item(),
    "forward_mae_mps": sum(
      abs(r["actual_velocity"][0] - r["target_velocity"][0]) for r in rows
    )
    / len(rows),
  }


def summarize(trace: list[dict], planned_steps: int, dt: float) -> dict:
  per_phase = {}
  for phase in dict.fromkeys(row["phase"] for row in trace):
    rows = [row for row in trace if row["phase"] == phase]
    per_phase[phase] = {
      "steps": len(rows),
      "velocity_mae": [
        sum(abs(r["actual_velocity"][i] - r["target_velocity"][i]) for r in rows)
        / len(rows)
        for i in range(3)
      ],
      "double_support_fraction": sum(all(r["foot_contact"]) for r in rows) / len(rows),
    }
    if "posture" in rows[0]:
      posture = [row["posture"] for row in rows]
      tail = posture[-round(0.5 / dt) :]
      stance_speed = [
        speed
        for row in rows
        for speed, contact in zip(
          row["posture"]["foot_speed_mps"], row["foot_contact"], strict=True
        )
        if contact
      ]
      per_phase[phase].update(
        torso_tilt_mean_deg=sum(p["torso_tilt_deg"] for p in posture) / len(posture),
        waist_mean_abs_deg=[
          math.degrees(sum(abs(row["waist_rad"][i]) for row in rows) / len(rows))
          for i in range(3)
        ],
        tail_0_5s_standing_joint_rms_rad=sum(p["standing_joint_rms_rad"] for p in tail)
        / len(tail),
        stance_foot_speed_mean_mps=sum(stance_speed) / len(stance_speed)
        if stance_speed
        else None,
      )
  reasons = trace[-1]["termination_reasons"]
  return {
    "planned_steps": planned_steps,
    "observed_steps": len(trace),
    "duration_s": len(trace) * dt,
    "completed": len(trace) == planned_steps and not reasons,
    "termination_reasons": reasons,
    "phases": per_phase,
  }


def make_eval_env(cfg: Config, task_id: str) -> ManagerBasedRlEnv:
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 1
  env_cfg.auto_reset = False
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.commands["motion"].motion_file = str(cfg.motion_file.resolve())
  env_cfg.commands["motion"].joint_position_range = (0.0, 0.0)
  env_cfg.terminations = {
    "pelvis_low": TerminationTermCfg(
      func=mdp.root_height_below_minimum, params={"minimum_height": FALL_HEIGHT_M}
    ),
    "pelvis_tilt": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": FALL_TILT_RAD}
    ),
  }
  env_cfg.scene.sensors = (
    *env_cfg.scene.sensors,
    ContactSensorCfg(
      name="p0_feet_ground",
      primary=ContactMatch("subtree", r"^(left|right)_ankle_roll_link$", "robot"),
      secondary=ContactMatch("body", "terrain"),
      fields=("found",),
      reduce="netforce",
    ),
  )
  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)


@torch.no_grad()
def evaluate(cfg: Config) -> None:
  if cfg.seed < 0:
    raise ValueError("Use a nonnegative seed for paired replay.")
  if cfg.contact_stop_duration is not None and (
    cfg.reference_bank is None
    or not math.isfinite(cfg.contact_stop_duration)
    or cfg.contact_stop_duration < 0.16
  ):
    raise ValueError("Contact stopping requires a reference bank and at least 0.16 s.")
  if cfg.contact_stop_max_foot_speed is not None and (
    cfg.contact_stop_duration is None
    or not math.isfinite(cfg.contact_stop_max_foot_speed)
    or cfg.contact_stop_max_foot_speed <= 0.0
  ):
    raise ValueError(
      "The contact stop foot speed requires a stop duration and a positive limit."
    )
  if cfg.start_upper_blend is not None and (
    cfg.reference_bank is None or not 0.16 <= cfg.start_upper_blend <= 0.6
  ):
    raise ValueError(
      "Choose a start upper-body blend from 0.16 to 0.6 s with a reference bank."
    )
  if cfg.direct_steering_start and (
    not cfg.directional_selection or cfg.start_upper_blend is None
  ):
    raise ValueError(
      "Direct steering starts require directional selection and a split start seam."
    )
  if (cfg.steady_forward or cfg.extra_forward_speeds) and not cfg.record_posture:
    raise ValueError("The steady forward test requires --record-posture True.")
  if cfg.motion_frames and not cfg.motion_only:
    raise ValueError("--motion-frames requires --motion-only.")
  if cfg.reference_bank is None and (cfg.planned_only or cfg.trackability_weight != 0):
    raise ValueError("Planned evaluation requires --reference-bank.")
  if cfg.prediction_audit_stride < 0 or (
    cfg.prediction_audit_stride and cfg.reference_bank is None
  ):
    raise ValueError(
      "The prediction audit requires a reference bank and a positive stride."
    )
  if cfg.contact_adapter is not None and cfg.reference_bank is None:
    raise ValueError("The contact adapter is only supported on the planned route.")
  if not 0.0 <= cfg.adapter_gain <= 1.0:
    raise ValueError("The contact adapter gain must be between zero and one.")
  if cfg.adapt_ranking and (not cfg.trackability_weight or cfg.contact_adapter is None):
    raise ValueError("Adapted ranking needs a contact adapter and world-model ranking.")
  if cfg.fixed_plan_dir is not None and (
    cfg.reference_bank is None or cfg.adapt_ranking
  ):
    raise ValueError("Fixed tape playback needs a reference bank and skips ranking.")
  if cfg.adapter_probe_phases and (
    cfg.fixed_plan_dir is None or cfg.contact_adapter is None
  ):
    raise ValueError(
      "Phase-restricted adapter probes need both a fixed tape and an adapter."
    )
  if cfg.motion_only and cfg.reference_bank is not None:
    raise ValueError("Fixed-reference and planned evaluations are separate modes.")
  if not math.isfinite(cfg.trackability_weight) or cfg.trackability_weight < 0:
    raise ValueError("The trackability weight must be finite and nonnegative.")
  if not math.isfinite(cfg.command_cost_weight) or cfg.command_cost_weight < 0:
    raise ValueError(
      "The world-model command cost weight must be finite and nonnegative."
    )
  if cfg.command_cost_weight and not cfg.trackability_weight:
    raise ValueError("Command cost requires world-model ranking.")
  if not 0.0 <= cfg.heading_strength <= 1.0:
    raise ValueError("Heading guidance strength must be in [0, 1].")
  if cfg.contact_launch_duration is not None and (
    cfg.reference_bank is None
    or not cfg.restart_from_rest
    or not 0.3 <= cfg.contact_launch_duration <= 0.8
  ):
    raise ValueError(
      "Contact launch needs measured restart and a duration from 0.3 to 0.8 s."
    )
  if not 0.0 <= cfg.contact_launch_lean <= 0.24 or (
    cfg.contact_launch_lean > 0.0 and cfg.contact_launch_duration is None
  ):
    raise ValueError(
      "Contact launch lean must be 0 to 0.24 rad with contact launch enabled."
    )
  if cfg.rank_contact_launch and (
    cfg.contact_launch_duration is None or not cfg.trackability_weight
  ):
    raise ValueError(
      "Ranking contact launches requires both contact launch and world-model ranking."
    )
  if not 0.0 <= cfg.contact_launch_forward_lead <= 0.12 or (
    cfg.contact_launch_forward_lead > 0.0 and cfg.contact_launch_duration is None
  ):
    raise ValueError(
      "Contact launch forward lead needs enabled launch and 0 to 0.12 m."
    )
  if not 1 <= cfg.candidate_entries <= 4:
    raise ValueError("Choose one to four entry phases per reference.")
  if cfg.axial_cycle_ranking and not cfg.directional_selection:
    raise ValueError("Axial cycle ranking needs direction-specific candidates.")
  if cfg.steering_measured_seam and not cfg.directional_selection:
    raise ValueError("Steering seams need direction-specific candidates.")
  if cfg.refresh_contact_stand and (
    cfg.contact_stop_duration is None or not cfg.restart_from_rest
  ):
    raise ValueError("Contact stand refresh requires measured stop and restart.")
  if cfg.start_search_time is not None and not 0.0 < cfg.start_search_time <= 1.0:
    raise ValueError("The start search window must be between zero and one second.")
  if not 1 <= cfg.prediction_steps <= 30:
    raise ValueError("Choose 1–30 world-model prediction steps.")
  if cfg.pace_margin is not None and (
    not math.isfinite(cfg.pace_margin) or cfg.pace_margin < 0
  ):
    raise ValueError("The reference pace margin must be finite and nonnegative.")
  if cfg.stop_double_support_pose is not None and (
    not math.isfinite(cfg.stop_double_support_pose)
    or cfg.stop_double_support_pose < 0.26
  ):
    raise ValueError(
      "The double-support stop pose limit must be finite and at least 0.26."
    )
  if cfg.motion_only and (
    cfg.steady_forward or cfg.extra_forward_speeds or cfg.multispeed_profile
  ):
    raise ValueError(
      "Motion-only collection and scripted joystick cases are separate modes."
    )
  if cfg.motion_only and (
    not math.isfinite(cfg.motion_window_s) or cfg.motion_window_s < 0.04
  ):
    raise ValueError("Motion-only windows must cover at least 0.04 seconds.")
  if any(not math.isfinite(speed) or speed <= 0 for speed in cfg.extra_forward_speeds):
    raise ValueError("Extra forward speeds must be finite positive m/s values.")
  if cfg.motion_file.suffix != ".npz" or not cfg.motion_file.is_file():
    raise ValueError("Pass one local G1 reference .npz file.")
  configure_torch_backends(allow_tf32=False, deterministic=True)
  config_path = cfg.checkpoint_dir / "local_config.json"
  saved = json.loads(config_path.read_text()) if config_path.is_file() else {}
  task_id = saved.get("task_id", cfg.task_id)
  runner_cfg = load_runner_cfg(saved) if saved else SuperMimicRunnerCfg()
  check_representation_matches(task_id, runner_cfg.representation)
  if not saved:
    print("[INFO] Using current runner defaults for this legacy checkpoint.")

  actor_path = cfg.checkpoint_dir / "actor.pt"
  weights = torch.load(actor_path, map_location="cpu", weights_only=True)
  velocity_obs_count = weights["goal_normalizers.velocity.count"].item()
  if velocity_obs_count == 0:
    raise ValueError("Choose a checkpoint with collected velocity goal statistics.")

  sim_env = make_eval_env(cfg, task_id)
  try:
    env = RslRlVecEnvWrapper(sim_env)
    wm = build_world_model(runner_cfg, env, cfg.device)
    actor = build_actor(runner_cfg, wm, cfg.device)
    actor.load_state_dict(weights)
    actor.eval()
    velocity_spec = next(spec for spec in actor.task_specs if spec.name == "velocity")
    assert velocity_spec.build_inference_goal_obs is not None
    motion_cmd = sim_env.command_manager.get_term("motion")
    assert isinstance(motion_cmd, MotionCommand)
    ref_raw = reference_velocity(
      motion_cmd.motion, runner_cfg.tasks.velocity.window_frames
    )
    lo, hi = velocity_limits(runner_cfg.tasks.velocity, ref_raw)
    ref_clipped = ref_raw.clamp(lo, hi)
    robot = sim_env.scene["robot"]
    data = robot.data
    feet = sim_env.scene["p0_feet_ground"]
    waist = [
      robot.joint_names.index(f"waist_{axis}_joint")
      for axis in ("yaw", "roll", "pitch")
    ]
    if cfg.record_posture or cfg.reference_bank:
      foot_sites, _ = robot.find_sites(
        tuple(name.replace("_ankle_roll_link", "_foot") for name in feet.primary_names),
        preserve_order=True,
      )
      torso = robot.body_names.index("torso_link")
      q_stand = data.default_joint_pos[0].clone()
    dt = sim_env.step_dt
    leg_indices = tuple(
      i
      for i, name in enumerate(robot.joint_names)
      if any(part in name for part in ("_hip_", "_knee_", "_ankle_"))
    )
    stance_bridge = None
    if cfg.contact_stop_duration is not None or cfg.contact_launch_duration is not None:
      stance_bridge = StanceBridge(
        sim_env.sim.mj_model,
        robot.indexing.free_joint_q_adr,
        robot.indexing.joint_q_adr,
        leg_indices,
        robot.indexing.site_ids[foot_sites],
        q_stand,
        dt,
        cfg.contact_stop_duration or 0.4,
      )
    clips = []
    if cfg.reference_bank is not None:
      clips = [
        ReferenceClip.load(path, tuple(robot.joint_names), cfg.device, dt)
        for path in sorted(cfg.reference_bank.glob("*.npz"))
      ]
      if not clips:
        raise ValueError(f"No reference clips under {cfg.reference_bank}.")
    dynamics_path = cfg.checkpoint_dir / "world_model.pt"
    if cfg.trackability_weight or cfg.prediction_audit_stride:
      wm.load_bundle(
        torch.load(dynamics_path, map_location=cfg.device, weights_only=True)
      )
      wm.dynamics.eval()
    adapter = None
    if cfg.contact_adapter is not None:
      trained = torch.load(
        cfg.contact_adapter, map_location=cfg.device, weights_only=True
      )
      if (
        runner_cfg.representation != "gencoord"
        or trained["actor_sha256"] != sha256(actor_path)
        or trained["world_model_sha256"] != sha256(dynamics_path)
        or not math.isclose(trained["dt"], dt)
      ):
        raise ValueError(
          "The adapter and the frozen F2 models or control interval differ."
        )
      adapter = ContactTrackingAdapter(
        tuple(robot.joint_names),
        tuple(sim_env.action_manager.get_term("joint_pos").target_names),
        wm.dynamics.action_scale,
        trained["limit_rad"],
      ).to(cfg.device)
      adapter.load_state_dict(trained["adapter"])
      adapter.eval()
    sim_ctx = wm.sim_context.read(sim_env)
    env_ids = torch.zeros(1, dtype=torch.long, device=cfg.device)
    scenarios = {}
    stand, walk = STAND_START, WALK_START
    if cfg.motion_only:
      stand = walk = (("reference", cfg.motion_window_s, (0.0, 0.0, 0.0)),)
    case_definitions = [
      ("stand_start", cfg.stand_frame, stand),
      ("walk_start", cfg.walk_frame, walk),
    ]
    if cfg.motion_frames:
      case_definitions = [
        (f"reference_{frame}", frame, stand) for frame in cfg.motion_frames
      ]
    if cfg.steady_forward:
      case_definitions.append(("steady_forward", cfg.stand_frame, STEADY_FORWARD))
    for speed in cfg.extra_forward_speeds:
      label = f"steady_forward_{speed:.3f}".replace(".", "p")
      phases = (STEADY_FORWARD[0], ("forward", STEADY_FORWARD[1][1], (speed, 0.0, 0.0)))
      case_definitions.append((label, cfg.stand_frame, phases))
    if cfg.multispeed_profile:
      case_definitions.append(
        ("multispeed_profile", cfg.stand_frame, MULTISPEED_PROFILE)
      )
    if cfg.cases:
      unknown = set(cfg.cases) - {case for case, _, _ in case_definitions}
      if unknown:
        raise ValueError(f"Unknown evaluation cases {sorted(unknown)}.")
      case_definitions = [row for row in case_definitions if row[0] in cfg.cases]
    for case, frame, phases in case_definitions:
      if case in scenarios:
        raise ValueError(f"Duplicate scenario: {case}")
      schedule = []
      for phase, duration, command in phases:
        target = ref_raw.new_tensor(command)
        if ((target < lo) | (target > hi)).any():
          raise ValueError(
            f"Scripted command {command} is outside the checkpoint range."
          )
        schedule.extend([(phase, target)] * round(duration / dt))
      if frame < 0 or frame + len(schedule) >= len(ref_raw):
        raise ValueError(f"{case} requires {len(schedule)} steps after frame {frame}.")
      scenarios[case] = (frame, schedule)
    if cfg.adapter_probe_phases:
      available = {phase for _, schedule in scenarios.values() for phase, _ in schedule}
      if set(cfg.adapter_probe_phases) - available:
        raise ValueError(
          "An adapter probe phase is absent from the selected schedules."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
      "task_id": task_id,
      "seed": cfg.seed,
      "device": cfg.device,
      "torch_version": torch.__version__,
      "actor_checkpoint": str(actor_path.resolve()),
      "actor_sha256": sha256(actor_path),
      "motion_file": str(cfg.motion_file.resolve()),
      "motion_sha256": sha256(cfg.motion_file),
      "velocity_goal_observations": velocity_obs_count,
      "runner_config": asdict(runner_cfg),
      "control_dt_s": dt,
      "control_hz": 1 / dt,
      "termination_thresholds": {
        "pelvis_low_m": FALL_HEIGHT_M,
        "pelvis_tilt_rad": FALL_TILT_RAD,
      },
      "foot_contact_order": feet.primary_names,
      "waist_order": [robot.joint_names[i] for i in waist],
      "action_order": sim_env.action_manager.get_term("joint_pos").target_names,
      "joint_position_order": robot.joint_names,
      "cases": {},
    }
    if cfg.reference_bank is not None:
      summary["planning"] = {
        "bank": str(cfg.reference_bank.resolve()),
        "clip_sha256": {
          p.name: sha256(p) for p in sorted(cfg.reference_bank.glob("*.npz"))
        },
        "strict_contact": cfg.strict_contact,
        "trackability_weight": cfg.trackability_weight,
        "prediction_steps": cfg.prediction_steps,
        "pace_margin": cfg.pace_margin,
        "stop_double_support_pose": cfg.stop_double_support_pose,
        "contact_stop_duration": cfg.contact_stop_duration,
        "contact_stop_max_foot_speed": cfg.contact_stop_max_foot_speed,
        "refresh_contact_stand": cfg.refresh_contact_stand,
        "start_upper_blend": cfg.start_upper_blend,
        "directional_selection": cfg.directional_selection,
        "direct_steering_start": cfg.direct_steering_start,
        "command_cost_weight": cfg.command_cost_weight,
        "candidate_entries": cfg.candidate_entries,
        "axial_cycle_ranking": cfg.axial_cycle_ranking,
        "steering_measured_seam": cfg.steering_measured_seam,
        "start_search_time": cfg.start_search_time,
        "restart_from_rest": cfg.restart_from_rest,
        "heading_strength": cfg.heading_strength,
        "contact_launch_duration": cfg.contact_launch_duration,
        "contact_launch_lean": cfg.contact_launch_lean,
        "contact_launch_forward_lead": cfg.contact_launch_forward_lead,
        "rank_contact_launch": cfg.rank_contact_launch,
        "world_model_sha256": sha256(dynamics_path)
        if cfg.trackability_weight or cfg.prediction_audit_stride
        else None,
      }
      if cfg.prediction_audit_stride:
        summary["planning"]["prediction_audit_stride"] = cfg.prediction_audit_stride
      if cfg.fixed_plan_dir is not None:
        summary["planning"]["fixed_plan_dir"] = str(cfg.fixed_plan_dir.resolve())
      if cfg.contact_adapter is not None:
        summary["planning"].update(
          contact_adapter_sha256=sha256(cfg.contact_adapter),
          adapter_gain=cfg.adapter_gain,
          adapt_ranking=cfg.adapt_ranking,
          adapter_probe_phases=cfg.adapter_probe_phases,
        )
    for case, (frame, schedule) in scenarios.items():
      initial = None
      case_summary = {"reference_frame": frame, "routes": {}}
      scripted_only = case.startswith("steady_forward") or case == "multispeed_profile"
      routes = ("scripted_velocity",) if scripted_only else ROUTES
      if cfg.motion_only:
        routes = ("motion",)
      if clips:
        routes = ("planned",) if cfg.planned_only else (*routes, "planned")
      for route in routes:
        task = "motion" if route in ("motion", "planned") else "velocity"
        actor.set_slot_mix({task: 1.0})
        sim_env.reset()
        motion_cmd.reset_to_frame(env_ids, frame)
        sim_env.scene.write_data_to_sim()
        sim_env.sim.forward()
        motion_cmd.update_relative_body_poses()
        sim_env.sim.sense()
        # The env reset cached observations before the reference teleport.
        obs = sim_env.observation_manager.compute(update_history=True)
        if initial is None:
          initial = (obs["full_state"].clone(), obs["actor"].clone())
          case_summary["initial_full_state"] = initial[0][0].tolist()
          case_summary["initial_velocity"] = heading_twist(
            data.root_link_quat_w, data.root_link_lin_vel_w, data.root_link_ang_vel_w
          )[0].tolist()
        else:
          torch.testing.assert_close(obs["full_state"], initial[0], rtol=0, atol=0)
          torch.testing.assert_close(obs["actor"], initial[1], rtol=0, atol=0)

        history = None
        trace = []
        planner = None
        audit = None
        tape = None
        moved = False
        if route == "planned":
          assert feet.data.found is not None
          initial_window = ReferenceWindow(
            data.joint_pos.clone(),
            data.joint_vel.clone(),
            data.root_link_pos_w.clone(),
            data.root_link_quat_w.clone(),
            data.site_pos_w[:, foot_sites].clone(),
            feet.data.found > 0,
          )
          planner = LocomotionPlanner(
            clips,
            initial_window,
            dt,
            strict_contact=cfg.strict_contact,
            trackability_weight=cfg.trackability_weight,
            prediction_steps=cfg.prediction_steps,
            pace_margin=cfg.pace_margin,
            stop_double_support_pose=cfg.stop_double_support_pose,
            start_legs=leg_indices if cfg.start_upper_blend is not None else (),
            start_upper_blend=cfg.start_upper_blend or 0.32,
            directional_selection=cfg.directional_selection,
            direct_steering_start=cfg.direct_steering_start,
            candidate_entries=cfg.candidate_entries,
            start_search_time=cfg.start_search_time,
            restart_from_rest=cfg.restart_from_rest,
            axial_cycle_ranking=cfg.axial_cycle_ranking,
            refresh_contact_stand=cfg.refresh_contact_stand,
          )
          if math.hypot(*case_summary["initial_velocity"][:2]) > 0.1:
            planner.mode = "walk"
          if cfg.prediction_audit_stride:
            audit = PredictionAudit(
              cfg.prediction_audit_stride,
              dt,
              cfg.heading_strength,
              adapter,
              cfg.adapter_gain,
            )
          if cfg.fixed_plan_dir is not None:
            tape_path = cfg.fixed_plan_dir / f"{case}_prediction.pt"
            tape = torch.load(tape_path, map_location=cfg.device, weights_only=True)
            if (
              tape["window_steps"] != list(range(len(schedule)))
              or not math.isclose(tape["dt"], dt)
              or not math.isclose(tape["heading_strength"], cfg.heading_strength)
              or tape["phase"] != [phase for phase, _ in schedule]
            ):
              raise ValueError(
                f"{tape_path}: need a full stride-1 capture of this schedule."
              )
            torch.testing.assert_close(
              tape["command"],
              torch.stack([command for _, command in schedule]),
              atol=0,
              rtol=0,
            )
            torch.testing.assert_close(
              tape["aux"][:1],
              wm.representation.extract_aux_state(obs["full_state"]),
              atol=0,
              rtol=0,
            )
            torch.testing.assert_close(
              tape["state"][:1],
              wm.representation.represent(obs["full_state"]),
              atol=0,
              rtol=0,
            )
            case_summary["fixed_reference_sha256"] = sha256(tape_path)
        for step, (phase, scripted_target) in enumerate(schedule):
          ref_frame = int(motion_cmd.time_steps[0])
          if route in ("scripted_velocity", "planned"):
            target = scripted_target
          else:
            phase, target = "reference", ref_clipped[ref_frame]
          state = wm.representation.represent(obs["full_state"])
          sensor = wm.representation.compute_actor_sensor(
            state,
            sim_ctx,
            sim_actor_obs=obs["actor"],
          ).unsqueeze(1)
          history = (
            sensor.repeat(1, actor.history_length, 1)
            if history is None
            else torch.cat((history[:, 1:], sensor), dim=1)
          )
          preview = None
          adapt_here = False
          if planner is not None:
            assert feet.data.found is not None
            aux = wm.representation.extract_aux_state(obs["full_state"])
            contact = feet.data.found > 0
            if adapter is not None:
              moved |= bool(target.abs().max() > 0.025)
              adapt_here = moved and (
                not cfg.adapter_probe_phases or phase in cfg.adapter_probe_phases
              )
            pure_forward = cfg.axial_cycle_ranking and bool(
              (target[0] > 0.025) & (target[1:].abs().max() < 0.025)
            )

            score = partial(
              predict_tracking_cost,
              actor=actor,
              dynamics=wm.dynamics,
              representation=wm.representation,
              state=state,
              aux=aux,
              history=history,
              context=sim_ctx,
              steps=cfg.prediction_steps,
              command=target,
              command_weight=0.0
              if pure_forward and planner.mode == "walk"
              else cfg.command_cost_weight,
              dt=dt,
              heading_strength=cfg.heading_strength,
              adapter=adapter if adapt_here and cfg.adapt_ranking else None,
              measured_contact=contact,
              adapter_gain=cfg.adapter_gain,
            )
            stop_fallback = None
            contact_launch = None
            measured = None
            if stance_bridge is not None or cfg.steering_measured_seam:
              measured = ReferenceWindow(
                data.joint_pos,
                data.joint_vel,
                data.root_link_pos_w,
                data.root_link_quat_w,
                data.site_pos_w[:, foot_sites],
                contact,
              )
            if stance_bridge is not None:
              assert measured is not None
              foot_speed = data.site_lin_vel_w[0, foot_sites, :2].norm(dim=-1)
              if cfg.contact_stop_duration is not None:
                stop_fallback = partial(
                  stance_bridge, measured, foot_speed, cfg.contact_stop_max_foot_speed
                )
              if cfg.contact_launch_duration is not None:
                contact_launch = partial(
                  stance_bridge.launch_candidates,
                  measured,
                  target,
                  cfg.contact_launch_duration,
                  foot_speed,
                  cfg.contact_launch_lean,
                  cfg.rank_contact_launch,
                  cfg.contact_launch_forward_lead,
                )
            measured_twist = (
              heading_twist(
                data.root_link_quat_w,
                data.root_link_lin_vel_w,
                data.root_link_ang_vel_w,
              )[0]
              if cfg.restart_from_rest
              else None
            )
            if tape is None:
              planner.update(
                target,
                contact[0],
                score,
                stop_fallback=stop_fallback,
                observed_twist=measured_twist,
                contact_launch=contact_launch,
                measured_steering_anchor=measured
                if cfg.steering_measured_seam
                else None,
              )
            else:
              planner.plan = ReferenceWindow(
                **{key: frames[step] for key, frames in tape["windows"].items()}
              )
              planner.offset = 0
            preview = planner.preview()
            goal = preview.actor_goal(
              torch.zeros_like(env_ids),
              aux,
              yaw_rate=float(target[2]) if cfg.heading_strength else 0.0,
              heading_strength=cfg.heading_strength,
              dt=dt,
            )
          elif route == "motion":
            goal = extract_actor_goal(obs["goal"])
          else:
            goal = velocity_spec.build_inference_goal_obs(target[None])
          goal_at_t = goal[0].tolist()
          reference_q = motion_cmd.motion.joint_pos[ref_frame]
          actor_obs = build_actor_obs_td({task: goal}, history)
          action = actor(actor_obs)
          adjustment = None
          if preview is not None and adapter is not None and adapt_here:
            action, adjustment = adapter(
              action,
              state,
              aux,
              target[None],
              preview.q[1:2],
              preview.qd[1:2],
              contact,
              preview.contact[:1],
              cfg.adapter_gain,
            )
          if audit is not None:
            before_aux = aux.clone()
          obs, reward, terminated, truncated, _ = sim_env.step(action)
          if audit is not None and not (terminated | truncated)[0].item():
            assert planner is not None
            audit.record(
              planner,
              phase,
              state,
              before_aux,
              history,
              goal,
              action,
              target,
              contact,
              wm.representation.represent(obs["full_state"]),
              wm.representation.extract_aux_state(obs["full_state"]),
              feet.data.found > 0,
              adapt_here,
            )
          actual = heading_twist(
            data.root_link_quat_w, data.root_link_lin_vel_w, data.root_link_ang_vel_w
          )
          reasons = [
            name
            for name in sim_env.termination_manager.active_terms
            if sim_env.termination_manager.get_term(name)[0].item()
          ]
          assert feet.data.found is not None
          row = {
            "step": step,
            "command_time_s": step * dt,
            "state_time_s": (step + 1) * dt,
            "phase": phase,
            "reference_frame": ref_frame,
            "reference_velocity_unclipped": ref_raw[ref_frame].tolist(),
            "target_velocity": target.tolist(),
            "actor_goal": goal_at_t,
            "actual_velocity": actual[0].tolist(),
            "reference_pelvis_position_w_m": motion_cmd.motion.root_pos_w[
              ref_frame
            ].tolist(),
            "reference_pelvis_quaternion_wxyz": motion_cmd.motion.root_quat_w[
              ref_frame
            ].tolist(),
            "reference_joint_pos_rad": reference_q.tolist(),
            "pelvis_position_w_m": data.root_link_pos_w[0].tolist(),
            "pelvis_quaternion_wxyz": data.root_link_quat_w[0].tolist(),
            "waist_rad": data.joint_pos[0, waist].tolist(),
            "foot_contact": (feet.data.found[0] > 0).tolist(),
            "executed_action": sim_env.action_manager.action[0].tolist(),
            "joint_target_rad": data.joint_pos_target[0].tolist(),
            "legacy_tracking_reward": reward[0].item(),
            "termination_reasons": reasons,
          }
          if adjustment is not None:
            row["contact_adjustment_rad"] = adjustment[0].tolist()
          if preview is not None and planner is not None:
            source = planner.clip
            row.update(
              reference_joint_pos_rad=preview.q[1].tolist(),
              reference_pelvis_position_w_m=preview.p[1].tolist(),
              reference_pelvis_quaternion_wxyz=preview.quat[1].tolist(),
              reference_velocity_unclipped=planned_twist(preview, dt).tolist(),
              planner_filtered_velocity=planner.filtered_command.tolist(),
              planner_mode=planner.mode,
              planner_source=source.name if source is not None else None,
              planner_source_frame=planner.frame + planner.offset
              if source is not None
              else None,
              planner_contact=preview.contact[1].tolist(),
              planner_candidates=planner.last_costs,
            )
            reference_q = preview.q[1]
            planner.advance()
          if cfg.record_posture:
            torso_quat = data.body_link_quat_w[0, torso]
            torso_tilt = 2 * torch.atan2(
              torso_quat[1:3].norm(), torso_quat[[0, 3]].norm()
            )
            q = data.joint_pos[0]
            row["posture"] = {
              "joint_pos_rad": q.tolist(),
              "foot_position_w_m": data.site_pos_w[0, foot_sites].tolist(),
              "reference_joint_rms_rad": (q - reference_q)
              .square()
              .mean()
              .sqrt()
              .item(),
              "standing_joint_rms_rad": (q - q_stand).square().mean().sqrt().item(),
              "torso_tilt_deg": math.degrees(torso_tilt.item()),
              "foot_speed_mps": data.site_lin_vel_w[0, foot_sites, :2]
              .norm(dim=-1)
              .tolist(),
            }
          trace.append(row)
          if (terminated | truncated)[0].item():
            break
        with (cfg.output_dir / f"{case}_{route}.jsonl").open("w") as stream:
          for row in trace:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if audit is not None:
          audit.write(cfg.output_dir / f"{case}_prediction.jsonl", wm, actor, sim_ctx)
        result = summarize(trace, len(schedule), dt)
        if case.startswith("steady_forward") and result["completed"]:
          result["arm_swing"] = arm_swing_metrics(
            trace[-round(4.0 / dt) :], robot.joint_names
          )
        case_summary["routes"][route] = result
        print(
          f"{case}/{route}: {len(trace)}/{len(schedule)} steps, {result['termination_reasons']}"
        )
      summary["cases"][case] = case_summary
    (cfg.output_dir / "summary.json").write_text(
      json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
  finally:
    sim_env.close()


def sha256(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
  evaluate(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
