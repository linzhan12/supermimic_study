"""Fit a bounded F2 tracking adapter through the frozen SuperMimic dynamics."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from eval_f2 import Config as EvalConfig
from eval_f2 import make_eval_env, sha256
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import quat_apply

from supermimic.contact_tracking import ContactTrackingAdapter
from supermimic.locomotion_planner import PREVIEW_STEPS, reference_actor_goal
from supermimic.reference_trackability import projected_twist
from supermimic.representations import GOAL_FRAME_STRIDE, NUM_GOAL_FRAMES
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.gencoord_helpers import decompose_rep_state


@dataclass
class Config:
  checkpoint_dir: Path
  motion_file: Path
  output_dir: Path
  train: tuple[Path, ...]
  validation: tuple[Path, ...] = ()
  device: str = "cuda:0"
  steps: int = 600
  batch: int = 32
  horizon: int = 8
  limit_rad: float = 0.06
  seed: int = 0
  command_weight: float = 1.0
  learning_rate: float = 0.001


def collect(paths: tuple[Path, ...], dt: float, horizon: int, device: str) -> dict:
  blocks = {}
  group_names = []
  for path in paths:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if abs(data["dt"] - dt) > 1e-7:
      raise ValueError(f"{path} has a different control interval.")
    phase, revision = data["phase"], data["reference_revision"]
    samples = [
      i
      for i, step in enumerate(data["window_steps"])
      if phase[step] != "stand"
      and step + horizon <= len(phase)
      and phase[step + horizon - 1] == phase[step]
      and revision[step + horizon - 1] == revision[step]
    ]
    if not samples:
      raise ValueError(f"{path} has no complete windows within one planned reference.")
    steps = [data["window_steps"][i] for i in samples]
    count = len(steps)
    block = {
      k: data[k][steps] for k in ("state", "aux", "history", "command", "contact")
    }
    block.update(
      context=data["context"].expand(count, -1),
      heading_strength=torch.full((count,), data["heading_strength"]),
      heading_anchor=data.get("heading_anchor", data["aux"][:, 3:])[steps],
      heading_lock=data.get("heading_lock", torch.zeros(len(phase), dtype=torch.bool))[
        steps
      ],
    )
    for k in ("q", "qd", "p", "quat", "contact"):
      block[f"ref_{k}"] = data["windows"][k][samples, : PREVIEW_STEPS + horizon - 1]
    for k, value in block.items():
      blocks.setdefault(k, []).append(value)
    group_names.extend(phase[index] for index in steps)
  masses = {g: group_names.count(g) for g in set(group_names)}
  return {
    **{k: torch.cat(value).to(device) for k, value in blocks.items()},
    "probabilities": torch.tensor(
      [masses[g] ** -0.5 for g in group_names], device=device
    ),
    "groups": masses,
  }


def foot_kinematics(env, device: str):
  robot = env.scene["robot"]
  feet = env.scene["p0_feet_ground"]
  bodies, _ = robot.find_bodies(feet.primary_names, preserve_order=True)
  sites, _ = robot.find_sites(
    tuple(name.replace("_ankle_roll_link", "_foot") for name in feet.primary_names),
    preserve_order=True,
  )
  model = env.sim.mj_model
  fk = ForwardKinematics(
    model,
    robot.indexing.body_ids,
    robot.indexing.joint_q_adr,
    torch.as_tensor(bodies),
    device=device,
  )
  offsets = torch.as_tensor(
    model.site_pos[robot.indexing.site_ids[sites].cpu().numpy()],
    device=device,
    dtype=torch.float32,
  )
  return fk, offsets


def objective(batch, wm, actor, adapter, feet, offsets, cfg: Config, gain: float):
  state, aux, history = batch["state"], batch["aux"], batch["history"]
  command, context = batch["command"], batch["context"]
  ref_q, ref_qd = batch["ref_q"], batch["ref_qd"]
  ref_p, ref_quat, contacts = batch["ref_p"], batch["ref_quat"], batch["ref_contact"]
  dt = wm.representation.dt
  origin = aux
  stance_loss = pose_loss = smoothness = state.new_zeros(())
  previous = state.new_zeros((len(state), len(adapter.joints)))
  fk, site_offsets = feet
  for step in range(cfg.horizon):
    indices = offsets + step
    # This split preserves the per-archive heading configuration inside a batch.
    yaw = command[:, 2:3]
    goal = reference_actor_goal(
      ref_q[:, indices],
      ref_qd[:, indices],
      ref_p[:, indices],
      ref_quat[:, indices],
      aux,
      offsets,
      yaw_rate=yaw,
      heading_strength=batch["heading_strength"][:, None],
      dt=dt,
      heading_anchor=batch["heading_anchor"],
      heading_lock=batch["heading_lock"],
    )
    nominal = actor(build_actor_obs_td({"motion": goal}, history))
    supported = contacts[:, step]
    action, correction = adapter(
      nominal,
      state,
      aux,
      command,
      ref_q[:, step + 1],
      ref_qd[:, step + 1],
      batch["contact"] if step == 0 else supported,
      supported,
      gain,
    )
    aux, state = wm.dynamics(state, action, aux, context)
    z, _, v, w, q, qd = decompose_rep_state(state)
    rot = aux[:, 3:]
    _, quat, lin, ang = fk(
      aux[:, :3],
      rot,
      q,
      quat_apply(rot, v),
      quat_apply(rot, w),
      qd,
    )
    lever = quat_apply(quat, site_offsets.expand(len(state), -1, -1))
    speed = (lin + torch.cross(ang, lever, dim=-1))[..., :2].norm(dim=-1)
    stationary = supported & contacts[:, step + 1] & contacts[:, max(0, step - 1)]
    stance_loss = (
      stance_loss
      + (((speed - 0.20).clamp_min(0.0) / 0.25).square() * stationary).mean()
    )
    pose_loss = pose_loss + ((q - ref_q[:, step + 1]) / 0.25).square().mean()
    pose_loss = (
      pose_loss + 0.3 * ((z[:, 0] - ref_p[:, step + 1, 2]) / 0.08).square().mean()
    )
    scaled = correction / cfg.limit_rad
    smoothness = smoothness + 0.05 * scaled.square().mean()
    if step:
      smoothness = smoothness + 0.08 * (scaled - previous).square().mean()
    previous = scaled
    sensor = wm.representation.compute_actor_sensor(
      state,
      context,
      aux_state=aux,
      joint_state=torch.cat((q, qd), dim=-1),
      last_action=action,
    )
    history = torch.cat((history[:, 1:], sensor[:, None]), dim=1)
  twist = projected_twist(origin, aux, dt * cfg.horizon)
  error = (twist - command) / state.new_tensor((0.18, 0.16, 0.50))
  velocity_loss = (
    F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").sum(-1).mean()
  )
  return torch.stack(
    (
      velocity_loss,
      pose_loss / cfg.horizon,
      stance_loss / cfg.horizon,
      smoothness / cfg.horizon,
    )
  )


def main(cfg: Config) -> None:
  if (
    not 1 <= cfg.horizon <= 20
    or cfg.steps < 1
    or cfg.batch < 1
    or cfg.command_weight < 0
  ):
    raise ValueError("Choose a horizon of 1–20 and positive iteration/batch counts.")
  if set(p.resolve() for p in cfg.train) & set(p.resolve() for p in cfg.validation):
    raise ValueError("The validation captures must be separate from training.")
  torch.manual_seed(cfg.seed)
  torch.set_float32_matmul_precision("highest")
  config_path = cfg.checkpoint_dir / "local_config.json"
  saved = json.loads(config_path.read_text()) if config_path.is_file() else {}
  runner = load_runner_cfg(saved) if saved else SuperMimicRunnerCfg()
  if runner.representation != "gencoord":
    raise ValueError(
      "Contact adaptation requires the generalized-coordinate F2 checkpoint."
    )
  config = EvalConfig(
    cfg.checkpoint_dir, cfg.motion_file, cfg.output_dir, device=cfg.device
  )
  sim = make_eval_env(config, saved.get("task_id", config.task_id))
  try:
    env = RslRlVecEnvWrapper(sim)
    wm = build_world_model(runner, env, cfg.device)
    actor = build_actor(runner, wm, cfg.device)
    actor_path, model_path = (
      cfg.checkpoint_dir / name for name in ("actor.pt", "world_model.pt")
    )
    actor.load_state_dict(
      torch.load(actor_path, map_location=cfg.device, weights_only=True)
    )
    wm.load_bundle(torch.load(model_path, map_location=cfg.device, weights_only=True))
    actor.set_slot_mix({"motion": 1.0})
    actor.eval().requires_grad_(False)
    wm.dynamics.eval().requires_grad_(False)
    adapter = ContactTrackingAdapter(
      tuple(sim.scene["robot"].joint_names),
      tuple(sim.action_manager.get_term("joint_pos").target_names),
      wm.dynamics.action_scale,
      cfg.limit_rad,
    ).to(cfg.device)
    fit = collect(cfg.train, sim.step_dt, cfg.horizon, cfg.device)
    validate = (
      collect(cfg.validation, sim.step_dt, cfg.horizon, cfg.device)
      if cfg.validation
      else None
    )
    feet = foot_kinematics(sim, cfg.device)
    offsets = GOAL_FRAME_STRIDE * torch.arange(NUM_GOAL_FRAMES, device=cfg.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg.learning_rate)
    weights = torch.tensor((cfg.command_weight, 0.7, 0.35, 1.0), device=cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)

    def sample(data, index):
      return {k: v[index] for k, v in data.items() if isinstance(v, torch.Tensor)}

    records = []
    with torch.no_grad():
      for name, data in (("train", fit), ("validation", validate)):
        if data is not None:
          batch = sample(data, slice(None))
          loss = objective(batch, wm, actor, adapter, feet, offsets, cfg, 0)
          records.append({"iteration": 0, name: loss.tolist()})
    for step in range(1, cfg.steps + 1):
      index = torch.multinomial(fit["probabilities"], cfg.batch, replacement=True)
      batch = sample(fit, index)
      loss = objective(batch, wm, actor, adapter, feet, offsets, cfg, 1)
      optimizer.zero_grad(set_to_none=True)
      (loss @ weights).backward()
      nn_norm = torch.nn.utils.clip_grad_norm_(
        adapter.parameters(), 1.0, error_if_nonfinite=True
      )
      optimizer.step()
      if step % 50 == 0 or step == cfg.steps:
        row = {
          "iteration": step,
          "train_batch": loss.tolist(),
          "gradient_norm": float(nn_norm),
        }
        if validate is not None:
          with torch.no_grad():
            row["validation"] = objective(
              sample(validate, slice(None)), wm, actor, adapter, feet, offsets, cfg, 1
            ).tolist()
        records.append(row)
        print(json.dumps(row), flush=True)
    torch.save(
      {
        "adapter": adapter.state_dict(),
        "actor_sha256": sha256(actor_path),
        "world_model_sha256": sha256(model_path),
        "limit_rad": cfg.limit_rad,
        "dt": sim.step_dt,
      },
      cfg.output_dir / "contact_adapter.pt",
    )
    (cfg.output_dir / "metrics.json").write_text(
      json.dumps(
        {
          "config": asdict(cfg),
          "train_groups": fit["groups"],
          "validation_groups": validate["groups"] if validate else {},
          "terms": ["command", "pose", "support_slip", "regularization"],
          "losses": records,
        },
        indent=2,
        default=str,
        allow_nan=False,
      )
      + "\n"
    )
  finally:
    sim.close()


if __name__ == "__main__":
  main(tyro.cli(Config))
