"""训练模型回放入口：恢复网络、动作资源和配置，并在物理仿真中运行策略。

文件位置：supermimic/src/supermimic/scripts/play.py。
本文件消费训练产生的 W&B 记录和模型 artifact，不调用策略或世界模型学习器。
主流程为 main 解析命令行 → play 恢复配置和单环境仿真 → 加载网络 → viewer.run。
本地 motion_file 只覆盖参考动作来源，网络结构和权重仍来自 wandb_path 指定运行。
播放需要网络结构、表示、观察布局和权重相互匹配；仅提供 actor.pt 不足以重建本流程。
"""

# 【文件作用】训练模型回放入口
# 从 W&B 恢复训练配置与权重，加载参考动作，在单环境仿真中运行策略并显示 viewer。
# 【阅读顺序】解析任务 → 重建配置和环境 → 恢复世界模型与策略 → 选择 viewer → 运行并关闭环境。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


from __future__ import annotations



# glob 展开本地动作路径通配符；排序后保持动作列表顺序稳定。
import glob
# os.path.expanduser 展开 ~，os.environ 检测原生显示环境。
import os
# dataclass 为 PlayConfig 生成初始化方法，供 tyro 按字段解析参数。
from dataclasses import dataclass
# Path 组合模型/动作路径，stem 提取动作显示名称。
from pathlib import Path
# Literal 限定初始任务和 viewer 模式的合法字符串，帮助 tyro 生成候选值。
from typing import Literal

# 导入 mjlab 基础任务会执行注册代码；虽然不直接调用该名字，也不能随意删除。
import mjlab.tasks  # noqa: F401
# PyTorch 负责设备选择、读取权重及后端设置所需张量/网络接口。
import torch
# tyro 分两阶段解析任务名和 PlayConfig，不负责创建仿真或加载模型。
import tyro
# mjlab 物理 RL 环境，按任务配置创建场景、管理器和 MuJoCo 仿真。
from mjlab.envs import ManagerBasedRlEnv
# 将物理环境包装为本项目网络工厂和 viewer 使用的向量环境接口。
from mjlab.rl import RslRlVecEnvWrapper
# list_tasks 查询已注册任务，load_env_cfg 取得任务的播放配置。
from mjlab.tasks.registry import list_tasks, load_env_cfg
# MotionCommandCfg 用于确认 motion 命令类型，之后可安全覆盖其 motion_file 配置。
from mjlab.tasks.tracking.mdp import MotionCommandCfg
# 统一配置 PyTorch 后端选项；这里在选择/创建网络前调用。
from mjlab.utils.torch import configure_torch_backends

# 来自 supermimic/src/supermimic/tasks/__init__.py：注册本项目两种坐标跟踪任务。
import supermimic.tasks  # noqa: F401
# 来自 supermimic/src/supermimic/runners/run_config.py：恢复保存的配置并检查任务表示匹配。
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
# 来自 supermimic/src/supermimic/runners/supermimic_runner.py：复用训练网络工厂，避免播放另建不兼容架构。
from supermimic.runners.supermimic_runner import build_actor, build_world_model
# 来自 supermimic/src/supermimic/scripts/viewers.py：原生/Viser viewer 负责交互循环，
# create_wm_vis_model 复制适合显示预测状态的 MuJoCo 可视化模型。
from supermimic.scripts.viewers import (
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
# 来自 supermimic/src/supermimic/utils/wandb_artifacts.py：缓存下载动作资源，
# download_model_dir 定位本运行最新的模型 artifact 并返回本地目录。
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)


@dataclass
class PlayConfig:
  """一次交互式回放的配置数据，由 tyro 根据字段生成命令行选项。

  wandb_path 是必填的 W&B 运行路径，用来找训练配置及最新模型；motion_file 可选，
  用于指定单个本地文件或 glob。initial_task 决定初始目标模态，播放界面可再次切换。
  device=None 时按 CUDA 可用性选择设备；num_steps_rollout 是交给 viewer 的运行步数。
  viewer 可选 native、viser、auto，默认直接使用 viser；只有 auto 才检测桌面显示。
  本类只保存参数，不连接 W&B、不加载权重，也不创建物理环境。
  """
  wandb_path: str
  motion_file: str | None = None
  initial_task: Literal["motion", "velocity"] = "motion"
  """界面初始任务，可在 viewer 下拉框中切换。"""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser"] = "viser"


# 函数作用：按指定任务和运行记录重建策略，在单环境仿真 viewer 中执行。
def play(task_id: str, cfg: PlayConfig) -> None:
  """按指定任务和运行记录重建策略，在单环境仿真 viewer 中执行。

  参数 task_id 是注册任务名称，决定机器人、完整状态与参考命令布局；cfg 提供运行记录、
  设备、动作覆盖和 viewer 设置。无张量输入，返回 None；运行结果通过界面和控制台展示。

  流程分为六段：配置 PyTorch 后端并选设备；加载 play 环境配置并设为单环境；
  读取 W&B 的 runner_cfg 并检查状态表示；优先解析本地动作，否则下载关联 motions；
  创建包装环境，按保存配置建立世界模型和 actor 后恢复权重；选择 viewer 并运行。
  模型的 eval 模式关闭训练相关行为，但并不在本函数内给整个 viewer 循环加 no_grad。

  副作用包括访问 W&B、下载/缓存文件、分配仿真和网络资源及运行交互式界面。
  显式 motion_file 未包含通配符时，本函数不提前检查文件存在性，由后续加载器处理。
  viewer 正常返回后关闭环境；本函数没有 try/finally，不能保证任意异常都走到 close。
  它不会训练网络、保存新模型或恢复旧运行的优化器与经验池。
  """
  # 【源码讲解】
  # 先按 task_id 创建播放环境配置，再从 W&B run 恢复训练时的 runner 配置。
  # 核对表示匹配后解析动作来源，建立单环境仿真并加载世界模型和 actor 权重。
  # 网络切换 eval 模式，viewer 负责持续采集观察、生成动作和展示。
  # 本地 motion_file 只覆盖参考动作，模型及训练配置仍来自指定 W&B run。
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading environment: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  # 播放只需要一个物理实例，降低展示开销；训练并行数从保存配置重建网络时并不直接复用。
  env_cfg.scene.num_envs = 1
  # 播放启用物理 actor 观察组的噪声；训练采集与世界模型合成的噪声处理在各自路径中完成。
  env_cfg.observations["actor"].enable_corruption = True

  # W&B API 在回放函数中导入并查询运行配置、输入动作资源和已记录的模型版本。
  import wandb

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  run = wandb.Api().run(cfg.wandb_path)
  # 从运行记录恢复网络宽度、历史长度、任务和状态表示，不能直接用当前默认配置加载旧权重。
  runner_cfg = load_runner_cfg(run.config)
  check_representation_matches(task_id, runner_cfg.representation)

  # 查找训练时关联的动作资源；若指定本地 motion_file 则优先使用本地来源。
  motion_artifacts = [a for a in run.used_artifacts() if a.type == "motions"]
  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  motion_names: list[str] = []
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    # 通配符没有匹配项就明确报错，避免创建环境后才发现没有动作可读。
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
    command_term.motion_file = files
    motion_names = [Path(f).stem for f in files]
    print(f"[INFO] Using {len(files)} local motion(s)")
  # 没有本地覆盖时使用训练运行关联的 motions，每个资源按本项目约定读取 motion.npz。
  elif motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(download_artifact_cached(artifact) / "motion.npz"))
    command_term.motion_file = (
      motion_files if len(motion_files) > 1 else motion_files[0]
    )
    motion_names = [a.name.split(":")[0] for a in motion_artifacts]
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")

  # 先实例化物理环境再包装接口；后续 build_world_model 会 reset 一次以读取实际观察维度。
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  wm = build_world_model(runner_cfg, env, device)
  representation = wm.representation
  dynamics = wm.dynamics

  model_dir = download_model_dir(run)
  # map_location 将权重加载到当前选定设备；模型结构已按训练配置重建。
  wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=device))
  dynamics.eval()

  actor = build_actor(runner_cfg, wm, device)
  actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
  # 播放关闭训练行为，例如辅助对齐计算和归一化统计刷新；这里执行而非训练策略。
  actor.eval()

  # 建立展示预测身体的模型，供 viewer 对比世界模型与物理轨迹，不创建新的学习器。
  wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)

  if cfg.viewer == "auto":
    # auto 模式检查桌面显示环境变量，有显示时选原生 viewer，否则选浏览器可视化。
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    # 显式指定 viewer 时尊重配置，不再根据当前桌面环境自动切换。
    resolved = cfg.viewer
  print(f"[INFO] Starting viewer: {resolved}")

  if resolved == "native":
    SuperMimicNativeViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    ).run()
  else:
    SuperMimicViserViewer(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      motion_names=motion_names,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    ).run()

  # viewer 正常结束后释放环境资源；该调用不负责执行训练或保存新的模型。
  env.close()


# 函数作用：从命令行选出任务和播放参数，然后调用 play。
def main() -> None:
  """从命令行选出任务和播放参数，然后调用 play。

  无显式参数，读取进程命令行；返回 None。第一阶段从已注册任务中解析任务名，
  return_unknown_args=True 将剩余选项交给第二阶段 PlayConfig 解析。
  任务包导入的注册副作用必须先执行，list_tasks 才能列出本项目任务。
  解析使用 mjlab.TYRO_FLAGS 保持与训练入口一致；真正联网与仿真在 play 中发生。
  """
  # 【源码讲解】
  # 与训练入口一致，先解析任务名，再解析剩余 PlayConfig 参数。
  # 任务注册模块必须已导入，注册表才能提供对应环境配置。
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(PlayConfig, args=remaining_args, config=mjlab.TYRO_FLAGS)
  play(chosen_task, args)


if __name__ == "__main__":
  main()
