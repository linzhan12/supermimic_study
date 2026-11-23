"""Replay a local speed checkpoint without accessing W&B."""

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.scripts.viewers import SuperMimicViserViewer, create_wm_vis_model


@dataclass
class Config:
  checkpoint_dir: Path
  motion_file: str | None = None
  device: str | None = None
  port: int = 8080
  start_paused: bool = False
  initial_task: Literal["motion", "velocity"] = "velocity"
  num_steps_rollout: int = 1000000
  task_id: str = "SuperMimic-GenCoord-G1"
  # Used when the original training entry has not saved local_config.json.
  runner: SuperMimicRunnerCfg = field(default_factory=SuperMimicRunnerCfg)


def main(cfg: Config):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  config_path = cfg.checkpoint_dir / "local_config.json"
  saved = json.loads(config_path.read_text()) if config_path.exists() else {}
  runner_cfg = load_runner_cfg(saved) if saved else cfg.runner
  task_id = saved.get("task_id", cfg.task_id)
  check_representation_matches(task_id, runner_cfg.representation)
  if not saved:
    print(
      "[INFO] No saved config; rebuilding with current runner defaults/CLI overrides. "
      "These must match the checkpoint's training architecture."
    )
  for name in ("world_model.pt", "actor.pt"):
    if not (cfg.checkpoint_dir / name).is_file():
      raise FileNotFoundError(cfg.checkpoint_dir / name)
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  source = cfg.motion_file or saved.get("motion_file")
  if not source:
    raise ValueError(
      "Pass --motion-file with the local motion path/glob used for training."
    )
  patterns = source.split(",") if isinstance(source, str) else source
  files = []
  for pattern in patterns:
    matches = sorted(glob.glob(str(Path(pattern.strip()).expanduser())))
    if not matches:
      raise FileNotFoundError(f"No local motions matched: {pattern}")
    files.extend(matches)
  env_cfg.commands["motion"].motion_file = files
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  try:
    wm = build_world_model(runner_cfg, env, device)
    wm.load_bundle(
      torch.load(
        cfg.checkpoint_dir / "world_model.pt", map_location=device, weights_only=True
      )
    )
    wm.dynamics.eval()
    actor = build_actor(runner_cfg, wm, device)
    actor.load_state_dict(
      torch.load(
        cfg.checkpoint_dir / "actor.pt", map_location=device, weights_only=True
      )
    )
    actor.eval()
    server = viser.ViserServer(host="127.0.0.1", port=cfg.port, label="SuperMimic")
    try:
      viewer = SuperMimicViserViewer(
        env=env,
        actor=actor,
        dynamics=wm.dynamics,
        representation=wm.representation,
        wm_vis_model=create_wm_vis_model(env.unwrapped.sim.mj_model),
        num_steps_rollout=cfg.num_steps_rollout,
        motion_names=[Path(f).stem for f in files],
        sim_context=wm.sim_context,
        task_specs=actor.task_specs,
        initial_task=cfg.initial_task,
        viser_server=server,
      )
      if cfg.start_paused:
        viewer.pause()
      print(f"[INFO] Local playback: http://127.0.0.1:{server.get_port()}", flush=True)
      viewer.run()
    finally:
      server.stop()
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
