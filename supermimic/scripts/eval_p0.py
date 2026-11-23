"""Run three closed-loop G1 policies from paired reference snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.tasks.task_spec import VelocityTaskCfg, build_actor_obs_td
from supermimic.tasks.velocity_tracking import (
  GAIT_START_PHASE,
  advance_gait_phase,
  heading_velocity,
)

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


def reference_velocity(motion: MotionLoader, window: int) -> torch.Tensor:
  """Smooth within the single evaluation clip before projecting by reference yaw."""
  linear, angular = motion.root_lin_vel_w, motion.root_ang_vel_w
  if window > 1:
    frames = torch.arange(len(linear), device=linear.device)
    offsets = torch.arange(window, device=linear.device) - window // 2
    indices = (frames[:, None] + offsets).clamp(0, len(linear) - 1)
    linear, angular = linear[indices].mean(1), angular[indices].mean(1)
  return heading_velocity(motion.root_quat_w, linear, angular)


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


@torch.no_grad()
def evaluate(cfg: Config) -> None:
  if cfg.seed < 0:
    raise ValueError("Use a nonnegative seed for paired replay.")
  if (cfg.steady_forward or cfg.extra_forward_speeds) and not cfg.record_posture:
    raise ValueError("The steady forward test requires --record-posture True.")
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
  sim_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
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
    if cfg.record_posture:
      foot_sites, _ = robot.find_sites(
        tuple(name.replace("_ankle_roll_link", "_foot") for name in feet.primary_names),
        preserve_order=True,
      )
      torso = robot.body_names.index("torso_link")
      q_stand = data.default_joint_pos[0].clone()
    dt = sim_env.step_dt
    sim_ctx = wm.sim_context.read(sim_env)
    env_ids = torch.zeros(1, dtype=torch.long, device=cfg.device)
    scenarios = {}
    case_definitions = [
      ("stand_start", cfg.stand_frame, STAND_START),
      ("walk_start", cfg.walk_frame, WALK_START),
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
    for case, (frame, schedule) in scenarios.items():
      initial = None
      case_summary = {"reference_frame": frame, "routes": {}}
      scripted_only = case.startswith("steady_forward") or case == "multispeed_profile"
      routes = ("scripted_velocity",) if scripted_only else ROUTES
      for route in routes:
        task = "motion" if route == "motion" else "velocity"
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
          case_summary["initial_velocity"] = heading_velocity(
            data.root_link_quat_w, data.root_link_lin_vel_w, data.root_link_ang_vel_w
          )[0].tolist()
        else:
          torch.testing.assert_close(obs["full_state"], initial[0], rtol=0, atol=0)
          torch.testing.assert_close(obs["actor"], initial[1], rtol=0, atol=0)

        history = None
        trace = []
        gait_phase = ref_raw.new_full((1,), GAIT_START_PHASE)
        for step, (phase, scripted_target) in enumerate(schedule):
          ref_frame = int(motion_cmd.time_steps[0])
          if route == "scripted_velocity":
            target = scripted_target
          else:
            phase, target = "reference", ref_clipped[ref_frame]
          goal = (
            extract_actor_goal(obs["goal"])
            if route == "motion"
            else velocity_spec.build_inference_goal_obs(target[None])
          )
          goal_at_t = goal[0].tolist()
          sensor = wm.representation.compute_actor_sensor(
            wm.representation.represent(obs["full_state"]),
            sim_ctx,
            sim_actor_obs=obs["actor"],
          ).unsqueeze(1)
          history = (
            sensor.repeat(1, actor.history_length, 1)
            if history is None
            else torch.cat((history[:, 1:], sensor), dim=1)
          )
          actor_obs = build_actor_obs_td({task: goal}, history)
          if runner_cfg.forward_gait_phase and task == "velocity":
            actor_obs["gait_phase"] = gait_phase
            gait_phase = advance_gait_phase(gait_phase, goal, dt)
          action = actor(actor_obs)
          obs, reward, terminated, truncated, _ = sim_env.step(action)
          actual = heading_velocity(
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
            "pelvis_position_w_m": data.root_link_pos_w[0].tolist(),
            "pelvis_quaternion_wxyz": data.root_link_quat_w[0].tolist(),
            "waist_rad": data.joint_pos[0, waist].tolist(),
            "foot_contact": (feet.data.found[0] > 0).tolist(),
            "executed_action": sim_env.action_manager.action[0].tolist(),
            "joint_target_rad": data.joint_pos_target[0].tolist(),
            "legacy_tracking_reward": reward[0].item(),
            "termination_reasons": reasons,
          }
          if cfg.record_posture:
            torso_quat = data.body_link_quat_w[0, torso]
            torso_tilt = 2 * torch.atan2(
              torso_quat[1:3].norm(), torso_quat[[0, 3]].norm()
            )
            q = data.joint_pos[0]
            row["posture"] = {
              "joint_pos_rad": q.tolist(),
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
