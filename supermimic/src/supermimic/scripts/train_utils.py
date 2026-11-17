"""Shared utilities for training scripts."""

from __future__ import annotations

from dataclasses import fields

import mujoco_warp as mjwarp
import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sim import SimulationCfg
from mjlab.utils.torch import configure_torch_backends

from supermimic.utils.wandb_artifacts import download_artifact_cached


class QuietSolverWarningCfg(SimulationCfg):
  """Suppress frequent solver-iteration prints, preserving other warnings."""

  def apply_wp_opt(self, wp_opt: mjwarp.Option) -> None:
    super().apply_wp_opt(wp_opt)
    # 这里只过滤两类提示的打印位，不改变求解器参数，也不会消除求解器问题。
    suppressed = mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS
    wp_opt.warn_overflow = int(wp_opt.warn_overflow) & ~int(suppressed)


def download_registry_motions(registry_name: str | None) -> str | list[str] | None:
  """Download motion files from W&B registry artifacts.

  Supports comma-separated registry names for multi-motion.
  Returns a single path string if one motion, a list if multiple, or None.
  """
  if not registry_name:
    return None

  names = [n.strip() for n in registry_name.split(",")]
  api = wandb.Api()

  paths = []
  for name in names:
    if ":" not in name:
      name = name + ":latest"
    artifact = api.artifact(name)
    paths.append(str(download_artifact_cached(artifact) / "motion.npz"))

  print(f"[INFO] Loaded {len(paths)} motion(s)")
  return paths if len(paths) > 1 else paths[0]


def setup_device(device: str | None = None) -> str:
  """Configure torch backends and resolve device."""
  configure_torch_backends()
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def create_env(
  env_cfg: ManagerBasedRlEnvCfg, num_envs: int, device: str
) -> RslRlVecEnvWrapper:
  """Set ``env_cfg.scene.num_envs`` and build a wrapped RL environment."""
  # 并行环境数是同时模拟的机器人数量，不是训练迭代次数。
  env_cfg.scene.num_envs = num_envs
  env_cfg.sim = QuietSolverWarningCfg(
    **{f.name: getattr(env_cfg.sim, f.name) for f in fields(SimulationCfg) if f.init}
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env)
