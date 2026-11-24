# 【文件作用】SuperMimic 的训练组织层：把物理采集、模型学习和策略学习连成循环。
#
# 先区分本文件的四种对象：
#   配置对象：BaseRunnerCfg / SuperMimicRunnerCfg，保存参数，本身不执行训练。
#   模型容器：WorldModel，持有状态表示、动力学网络和仿真上下文，不是 nn.Module。
#   公共组织器：BaseRunner，建立经验池、世界模型、统计量、日志等公共设施。
#   具体组织器：SuperMimicRunner，建立多任务策略，决定每次采集和更新如何衔接。
#
# 【入口与对象关系】
# supermimic/src/supermimic/scripts/train.py 构造 SuperMimicRunner，再调用 run()。
# BaseRunner.__init__ 先调用 _init_common()，再动态调用子类 _init_policy()。
# 世界模型学习器与想象环境持有同一 Dynamics 对象；模型更新后无需复制给想象环境。
# 策略学习器持有 actor；物理采集也从这个 actor 取动作，使用的是同一套策略参数。
#
# 【一轮数据流】
# 物理观察 → actor 动作加探索噪声 → 物理环境 step → ReplayBuffer
#   → 连续窗口训练 WorldModelLearner → 回放起点初始化想象环境
#   → PolicyLearner 在想象轨迹中优化 actor → 用新策略继续采集。
# 物理引擎不参与自动微分；策略梯度通过学习到的可微 Dynamics 传递。
#
# 【本文件使用的维度符号】
# E=物理环境数，B=想象/训练批大小，H=策略观察历史长度，T=模型训练窗口帧数；
# A=动作数，S=完整状态宽度，R=模型表示宽度，C=随机化上下文宽度，G=完整目标宽度。
# E 与 B 可以不同；一次外层循环、一次参数更新和一个回合也不是同一个计数。
#
# 【阅读顺序】先看导入说明和配置，再看 build_world_model/build_actor，
# 然后看 BaseRunner 的初始化、采集缓存工具，最后看 SuperMimicRunner.step/run。

# 推迟解析类型注解，允许下方在类型标注中引用尚未定义的配置与容器。
from __future__ import annotations

# time.time 用于统计墙钟耗时，区分物理采集区间、世界模型训练和策略训练耗时。
import json
import time
# ThreadPoolExecutor 创建单工作线程，后台上传模型；Future 是上传任务的句柄。
# 保存下个版本前调用 Future.result() 等待上次任务，防止上传还在读文件时就覆盖文件。
from concurrent.futures import Future, ThreadPoolExecutor
# dataclass 自动生成配置/容器的初始化方法；field(default_factory=...) 为每个实例创建独立子配置。
# asdict 将嵌套配置转成字典，记录到 W&B，播放时据此重建网络结构。
from dataclasses import asdict, dataclass, field
# datetime.now 生成本地时间戳，作为日志目录的一层名称。
from datetime import datetime
# Path 组合日志与检查点路径、创建目录；这里的相对路径以程序工作目录为基准。
from pathlib import Path
# Annotated 为策略配置附带 tyro 子命令名称；Literal 限定采样模式的合法字符串。
# 这些是类型/命令行元数据，不负责执行模型学习。
from typing import Annotated, Literal

# PyTorch 提供张量、设备迁移、自动微分控制、网络编译和权重保存/加载。
# 本文件主要组织调用；实际 loss.backward 与优化器更新位于各 learner。
import torch
# nn.Module 是网络模块基类；本文件用它标注公共 MLP 工厂的返回类型。
import torch.nn as nn
# 本文件使用 tyro.conf.subcommand 定义 bptt / actor-critic 配置变体；参数实际由训练入口解析。
import tyro
# Weights & Biases 管理实验配置、指标和版本化文件资源；run 是一次训练记录。
# artifact 指一组可版本化文件；本文件既关联动作来源，也上传模型检查点。
import wandb
# rsl_rl 的 MLPModel 接收命名观察组并内置观察归一化，用于 BaseRunner._build_mlp。
# 这是给复用基类的 runner 提供的工厂；当前多编码器 actor 并不由 MLPModel 构建。
from rsl_rl.models import MLPModel
# TensorDict 是共享批次形状的张量字典。例：state:[E,S]、action:[E,A]，共同 batch_size=[E]。
# 它让回放采样、按任务组织观察和按 batch 堆叠拥有统一接口；不是普通二维矩阵。
from tensordict import TensorDict

# 来自 supermimic/src/supermimic/envs/motion_tracking_env.py：单任务想象环境。
# BaseRunner._create_wm_env 保留此公共工厂；本文件具体 runner 的主路径使用下方多任务版本。
from supermimic.envs.motion_tracking_env import MotionTrackingEnv
# 来自 supermimic/src/supermimic/envs/multi_task_world_model_env.py：多任务可微想象环境。
# 用 Dynamics 推进预测状态，提供任务目标、跟踪奖励和软终止，不直接推进 MuJoCo。
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.envs.velocity_world_model_env import VelocityWorldModelEnv
# 来自 supermimic/src/supermimic/learners/policy_learner.py：策略训练算法和对应配置。
# BpttPolicyLearner 在模型中展开动作序列，用回报对 actor 反传；Cfg 决定批大小、步数和学习率等。
# ActorCriticPolicyLearner 在此基础上增加末端价值估计和 critic 拟合；其 Cfg 类型也用于选择算法分支。
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
# 来自 supermimic/src/supermimic/learners/wm_learner.py：世界模型监督学习器与配置。
# 它消费真实仿真连续窗口，优化 Dynamics 的多步预测误差；不负责优化 actor。
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
# 来自 supermimic/src/supermimic/models.py：Dynamics 是状态/动作到下一状态的可微预测网络；
# MultiEncoderAE 为各任务编码目标，再用共享解码器输出动作；build_mlp 构造普通张量 MLP。
# 这里的 build_mlp 用于当前 runner 的 critic，与基类返回 MLPModel 的 _build_mlp 不同。
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.tasks.forward_gait import (
  ForwardGaitAdapter,
  G1SagittalReflection,
  warm_start_gait_adapter,
)
from supermimic.tasks.steering_recovery import SteeringRecoveryAdapter
# 来自 supermimic/src/supermimic/replay_buffer.py：FieldSpec 描述单字段宽度与类型，
# ReplayBuffer 为每个物理环境维护独立时间环，提供不跨回合的连续窗口和历史起点采样。
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
# 来自 supermimic/src/supermimic/representations.py：StateRepresentation 约定状态语义与积分接口；
# create_representation 按 gencoord/maxcoord 选择具体实现，让 runner 不必硬编码两种坐标算法。
from supermimic.representations import GenCoordRepresentation, StateRepresentation, create_representation
# 来自 supermimic/src/supermimic/runners/exporter.py：把满足条件的传感器策略导出为 ONNX。
# 返回的文件路径加入同一次模型上传；不是每次策略更新都执行导出。
from supermimic.runners.exporter import export_actor_as_onnx
# 来自 supermimic/src/supermimic/runners/reset_sampling.py：根据物理采集回报估计动作时间桶难度，
# 组合均匀探索、低回报重点练习和历史练习，更新下一次重置的参考动作起点分布。
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
# 来自 supermimic/src/supermimic/sim_context.py：读取并归一化摩擦、质心、编码器偏置等条件。
# 同一状态和动作在不同条件下后果不同，所以该上下文既进入动力学输入，也随数据写入回放。
from supermimic.sim_context import SimContext
# 来自 supermimic/src/supermimic/tasks/task_spec.py：TasksCfg 保存任务参数；
# resolve_task_specs 把配置解析成有序任务定义，规定目标宽度和目标提取方法；
# build_actor_obs_td 把历史和 goal_任务名 等字段包装成策略所需 TensorDict。
from supermimic.tasks.task_spec import (
  TasksCfg,
  build_actor_obs_td,
  resolve_task_specs,
)
from supermimic.tasks.velocity_tracking import IndependentVelocityCommand
from supermimic.tasks.velocity_posture import G1ArmSwingControl
# 来自 supermimic/src/supermimic/utils/helpers.py：extract_action_scale_offset 读取动作缩放与偏置，
# 使 Dynamics 与物理环境采用相同命令尺度；unpack_obs_dims 读取 full_state 和 goal 的特征宽度。
from supermimic.utils.helpers import (
  extract_action_scale_offset,
  unpack_obs_dims,
)
# 来自 supermimic/src/supermimic/utils/logger.py：累积真实环境的回合奖励、长度和额外指标，
# 在日志间隔汇总吞吐量、训练损失并写控制台与 W&B；不执行参数更新。
from supermimic.utils.logger import Logger
# 来自 supermimic/src/supermimic/utils/wandb_artifacts.py：download_model_dir 找到并缓存下载最新模型资源；
# log_model_artifact 将检查点文件登记为模型版本并执行版本保留策略，供后台上传线程调用。
from supermimic.utils.wandb_artifacts import download_model_dir, log_model_artifact


@dataclass
class BaseRunnerCfg:
  """所有训练组织器共用的配置，只描述设置，不创建环境或训练网络。

  由训练入口构造或由子类配置继承。字段分为实验记录、回放容量、状态表示、
  策略观察、日志保存频率和世界模型训练参数。BaseRunner 会读取这些设置，
  但具体策略结构和更新总次数由 SuperMimicRunnerCfg 等子配置补充。

  注意区分 buffer_size（每物理环境时间槽数）、history_length（策略历史帧数）
  以及 wm.horizon（监督窗口帧数），它们改变的是三种不同的时间范围。
  配置里的 headless 在本文件没有消费点，不能仅修改该字段就推断会启动或关闭 viewer。
  """

  # 训练记录所属 W&B 项目；run_name 是显示名称，未指定时按任务和时间自动生成。
  wandb_project: str = "supermimic"
  run_name: str | None = None
  # 可选的世界模型初始化来源；这里恢复模型权重，不恢复经验池或优化器训练进度。
  wandb_run_path: str | None = None
  init_checkpoint_dir: str | None = None
  # 当前文件没有读取 headless；它不是本训练循环里实际控制显示窗口的开关。
  headless: bool = True
  # 每个物理环境保存 1000 个时间槽；总缓存条数约为 num_envs × buffer_size。
  buffer_size: int = 1000
  # 选择模型状态与预测布局；必须与任务提供的完整状态匹配。
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  # 选择直接使用模型状态或部署形状传感器输入，后者支持多帧历史。
  actor_obs: str = "rep_state"  # "rep_state" or "proprio"
  # 每隔多少个外层采集循环输出一次日志，条件为 step % log_interval == 0。
  log_interval: int = 10
  # 每隔多少个外层循环保存；step 从 0 开始，因此首次循环也触发保存分支。
  save_interval: int = 1000
  # 历史长度只影响策略输入；大于 1 时当前实现要求 proprio，并为重置填充历史。
  history_length: int = 1
  # 软终止 sigmoid 的温度；越小越接近硬阈值，阈值附近的变化也越陡。
  termination_softness: float = 0.05
  # 动力学的三个隐藏层各为 2048 个特征；输入/输出宽度另由环境和表示推断。
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  # default_factory 为每个 runner 单独创建配置，避免多个配置对象共享可变实例。
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)


# Annotated 附带命令行元数据，不改变配置字段本身；两种配置被暴露为可选子命令。
BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]


@dataclass
class SuperMimicRunnerCfg(BaseRunnerCfg):
  """SuperMimic 模型学习加多任务策略学习的完整配置，继承公共设置。

  新增 actor/critic 网络宽度、目标潜变量宽度、任务槽位比例、策略算法、
  采集与学习复用比例，以及参考动作重置采样方式。默认策略为 actor-critic，
  默认槽位全部分给 motion；velocity 虽然有任务定义，默认比例为零。

  slot_mix 改变的是同一批环境使用哪种目标编码器，各槽位仍计算动作跟踪奖励。
  sampling_mode 改变的是环境重置到参考动作哪个时刻，与任务槽位分配是两回事。
  真正的实例化发生在 SuperMimicRunner._init_policy；配置对象本身不持有网络。
  """
  # 外层采集循环次数；每次采集全部并行环境，不等于回合数或梯度更新数。
  max_iterations: int = 1000000
  # 控制数据复用强度；run 会结合采集量、世界模型批数、批大小和窗口长度换算更新频率。
  replay_ratio: int = 16
  # 各任务目标编码后的共同潜变量宽度；共享解码器不需要直接处理不同宽度的目标。
  latent_dim: int = 32
  # 各目标编码器及共享动作解码器使用这组隐藏层宽度，具体连接见 supermimic/src/supermimic/models.py 的 MultiEncoderAE。
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  # critic 将 R+G+C 维特权观察映射为一个标量状态价值，不与 actor 共用此处的 MLP 权重。
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_bc_weight: float = 1e-1
  """非锚点编码器向同状态锚点潜变量对齐的损失权重；设为 0 则关闭。"""
  independent_velocity: bool = False
  """Fine-tune the velocity encoder with independent flat-ground commands."""
  posture_weight: float = 0.0
  """Weight for the phase-conditioned posture reward in independent velocity training."""
  moving_posture_scale: float = 1.0
  """Posture weight multiplier during nonzero velocity commands."""
  arm_swing_weight: float = 0.0
  """Weight for contralateral arm swing during forward walking."""
  arm_swing_control: float = 0.0
  """Blend weight of phase-driven G1 shoulder targets in the velocity branch."""
  arm_swing_roll: float = 0.0
  """Symmetric outward shoulder roll target for phase-driven walking, in radians."""
  arm_swing_speed_gate: float = 0.0
  """Positive forward speed at which phase-driven shoulder targets start ramping in."""
  stop_encoder_init_dir: str | None = None
  """Initialize a separate near-zero velocity encoder from a local checkpoint."""
  freeze_moving_encoder: bool = True
  """Freeze the original velocity encoder when a separate stop encoder is present."""
  turn_tracking_weight: float = 0.0
  """Additional narrow yaw tracking reward on turning commands."""
  forward_gait_weight: float = 0.0
  """Weight of the speed-conditioned demonstration reward for straight walking."""
  forward_gait_action_limit: float = 0.0
  """Train a straight-walking adapter with at most this many radians of action correction."""
  forward_gait_command_fraction: float = 0.0
  """Probability of replacing an ordinary sampled command with forward walking."""
  forward_gait_phase: bool = False
  """Condition straight walking and its imitation reward on a continuous gait phase."""
  forward_gait_teacher_steps: int = 0
  """Supervised warm-start updates from the frozen full-reference policy."""
  forward_gait_full_body: bool = False
  """Track continuous mocap targets and allow corrections to all G1 joints."""
  forward_gait_demo_files: str | None = None
  """Comma-separated physical full-reference demonstrations for full-body gait shaping."""
  forward_gait_weight_transfer: bool = False
  """Track torso motion and phase-dependent lateral transfer in straight walking."""
  forward_gait_foot_orientation: bool = False
  """Track the demonstrated foot axes relative to the robot's heading."""
  forward_gait_foot_placement: bool = False
  """Track demonstrated feet relative to the robot's heading and the ground."""
  forward_gait_demo_reset_fraction: float = 0.0
  """Fraction of forward resets initialized from a matched physical gait pose and phase."""
  forward_gait_arm_weight: float = 0.0
  """Extra weight for matching the demonstrated shoulders and elbows."""
  forward_gait_arm_demo_files: str | None = None
  """Optional comma-separated source for the phase-aligned arm targets."""
  forward_gait_fast_stance_weight: float = 0.0
  """Extra weight for fast-walking knee and foot heading targets during support."""
  forward_gait_fast_lateral_scale: float = 1.0
  """Scale demonstrated lateral velocity from 0.35 to 0.50 m/s."""
  forward_foot_cycle_weight: float = 0.0
  """Weight for phase-aligned alternation, clearance and placement of the feet."""
  forward_gait_reflection: float = 0.0
  """Blend straight-walking actions with half-cycle reflected state feedback."""
  forward_precision_weight: float = 0.0
  """Additional tight velocity tracking for straight walking."""
  forward_gait_hold_s: tuple[float, float] | None = None
  """Override the hold interval for added forward-walking commands."""
  steering_recovery_limit: float = 0.0
  """Train a bounded non-straight action correction in radians."""
  steering_recovery_weight: float = 0.0
  """Weight of narrow heading-velocity tracking for turning and lateral motion."""
  train_steering_recovery: bool = True
  """When a steering adapter is present, update it instead of the walking adapter."""
  tasks: TasksCfg = field(default_factory=TasksCfg)
  """各任务目标提取、命令范围和平滑窗口等设置。"""
  slot_mix: dict[str, float] = field(
    default_factory=lambda: {"motion": 1.0, "velocity": 0.0}
  )
  """物理采集和想象批次中各任务的占比；比例非负且总和为 1。"""
  # 默认使用带价值网络的 BPTT；联合类型及 Annotated 元数据让 tyro 生成配置子命令。
  policy: BpttCfg | ActorCriticCfg = field(default_factory=ActorCriticCfg)

  sampling_mode: Literal["uniform", "adaptive", "start", "return"] = "return"
  """参考状态初始化的采样模式；return 模式重点练习低回报动作时间桶。

  训练入口在创建环境前将 return 转为 motion 命令的 external 模式，
  其他选项由 mjlab 的原生采样逻辑处理。
  """
  # 仅 return 重置模式使用；配置桶价值的估计窗口及探索、重点练习比例。
  return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)


@dataclass
class WorldModel:
  """把状态语义、预测网络和仿真条件打包的容器，由 build_world_model 创建。

  representation 负责完整状态与模型状态转换、预测积分和奖励所需状态恢复；
  dynamics 持有可训练的动力学参数；sim_context 定义物理条件的列布局与尺度。
  num_goals 是完整世界模型目标宽度，num_actions 是控制向量宽度。

  本类是 dataclass，不是神经网络基类 nn.Module，不提供 forward 或独立优化器。
  Runner、WorldModelLearner 和想象环境共享其中的对象引用，而非各自拷贝网络。
  bundle/load_bundle 只定义检查点中模型与上下文的内容，架构仍需由配置和环境重建。
  """

  representation: StateRepresentation
  dynamics: Dynamics
  sim_context: SimContext
  num_goals: int
  num_actions: int
  action_names: tuple[str, ...] = ()

  # 函数作用：返回用于保存世界模型的状态字典组合。
  def bundle(self) -> dict:
    """返回用于保存世界模型的状态字典组合。

    无额外参数，读取当前 dynamics 和 sim_context 的 state_dict，返回包含
    "dynamics" 与 "sim_context" 两个键的字典。网络 buffer 也会按 PyTorch 的
    state_dict 规则保存，因此动作尺度及归一化等状态不只是训练权重。

    调用者是 run 的检查点分支；本函数只打包，不创建文件、不上传，也不包含
    actor、critic、优化器、回放缓存或外层循环进度。
    """
    return {
      "dynamics": self.dynamics.state_dict(),
      "sim_context": self.sim_context.state_dict(),
    }

  # 函数作用：把已加载的检查点内容写入当前模型和上下文对象。
  def load_bundle(self, bundle: dict) -> None:
    """把已加载的检查点内容写入当前模型和上下文对象。

    参数 bundle 必须有 dynamics 与 sim_context 两个键，内容对应 bundle()。
    返回 None；副作用是原地恢复当前实例的权重与持久化缓冲，不替换对象引用。

    调用前必须先用匹配的配置和环境建好结构。load_state_dict 会检查键和尺寸；
    本函数不做架构猜测、不下载文件，也不恢复策略和优化器状态。
    """
    self.dynamics.load_state_dict(bundle["dynamics"])
    self.sim_context.load_state_dict(bundle["sim_context"])


# 函数作用：从实际物理环境推断尺寸，构建并返回一个 WorldModel 容器。
def build_world_model(cfg: BaseRunnerCfg, env, device: str) -> WorldModel:
  """从实际物理环境推断尺寸，构建并返回一个 WorldModel 容器。

  参数：cfg 提供表示名称、观察模式、历史长度与动力学隐藏层宽度；env 是已创建的
  RL 包装环境；device 指定模型和相关张量的设备。函数会调用 env.reset() 读取
  观察宽度，这是可观察的环境副作用，而非纯配置解析。

  流程：验证历史观察配置 → 从 full_state/goal 获取 S、G，读取 A → 绑定仿真
  上下文获得 C → 以控制步长、物理子步数和默认关节姿态创建表示 → 绑定环境索引
  及运动学 → 读取动作尺度 → 创建输入含 R+A+C 特征的 Dynamics → 打包返回。

  长历史要求 proprio 模式；完整状态布局必须与 cfg.representation 相符。
  这里不创建 actor，也不执行参数更新。BaseRunner 和播放脚本都可复用该工厂。
  """
  if cfg.history_length > 1 and cfg.actor_obs != "proprio":
    raise RuntimeError(
      "history_length > 1 requires actor_obs='proprio' (--runner.actor-obs proprio)"
    )

  # 先获取一次环境观察，用其维度建立对应大小的表示和网络。
  obs, _ = env.reset()
  num_full_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

  # unwrapped 去掉 RL 包装层，直接读取机器人资产和物理随机化事件。
  # 上下文宽度 C 决定动力学输入增加多少列；模型需知道不同环境的物理条件。
  sim_context = SimContext.from_env(env.unwrapped)
  sim_ctx_dim = sim_context.output_dim
  print(
    f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})"
  )

  # 默认姿态在并行环境中共享，取第 0 个环境的 [关节数] 向量。
  # 用于相对关节观察等计算；不是从动作参考序列读取当前目标姿态。
  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  representation = create_representation(
    cfg.representation,
    num_full_state,
    num_actions,
    env.unwrapped.step_dt,
    env.unwrapped.cfg.decimation,
    default_joint_pos,
    actor_obs_mode=cfg.actor_obs,
  ).to(device)

  # 建立表示与环境之间的联系：关节软限位、正/逆运动学及传感器偏置映射等。
  # 先构建表示再 bind，是因为这些信息依赖实际实例化后的机器人索引。
  representation.bind(env.unwrapped, sim_context)

  print(
    f"[INFO] Representation: {cfg.representation} "
    f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
    f"pred={representation.prediction_dim})"
  )

  # 读取物理动作管理器的尺度，保证模型与仿真解释同一原始动作时得到相同关节命令。
  action_scale, action_offset = extract_action_scale_offset(env)
  dynamics = Dynamics(
    representation,
    num_actions,
    sim_ctx_dim,
    action_scale,
    action_offset,
    hidden_dims=cfg.dynamics_hidden_dims,
  ).to(device)

  return WorldModel(
    representation=representation,
    dynamics=dynamics,
    sim_context=sim_context,
    num_goals=num_goals,
    num_actions=num_actions,
    action_names=tuple(env.unwrapped.action_manager.get_term(
      env.unwrapped.action_manager.active_terms[0]
    ).target_names),
  )


# 函数作用：根据世界模型的维度信息与任务配置创建多编码器策略。
def build_actor(
  cfg: SuperMimicRunnerCfg, wm: WorldModel, device: str
) -> MultiEncoderAE:
  """根据世界模型的维度信息与任务配置创建多编码器策略。

  参数 cfg 提供目标潜变量、任务占比、隐藏层和历史长度；wm 提供状态表示、
  动作宽度及完整目标宽度；device 决定网络所在设备。返回 MultiEncoderAE，
  其 task_specs 保存解析后的有序任务定义，供后续提取目标和分配批次使用。

  单帧输入宽度在 rep_state 模式为 R，在 proprio 模式为传感器宽度 D。
  解码器内部再展平 H 帧历史。每个任务目标先经独立编码器得到同宽度潜变量，
  再与历史拼接送入共享解码器输出 A 维动作。这里仅建网络，不生成训练样本。
  """
  # 【源码讲解】
  # 从任务定义得到各目标宽度，构造独立编码器和共享解码器。
  # 网络观察宽度由表示及历史长度决定，收集与想象共用同一个 actor。
  representation = wm.representation
  # 这里只计算单帧宽度；历史长度由解码器输入的 history_length × obs_dim 体现。
  obs_dim = (
    representation.rep_state_dim
    if cfg.actor_obs == "rep_state"
    else representation.proprio_obs_dim(wm.num_actions)
  )
  arm_control = None
  if cfg.arm_swing_control:
    if not isinstance(representation, GenCoordRepresentation) or cfg.actor_obs != "rep_state":
      raise ValueError("Arm swing control requires G1 GenCoord rep_state observations.")
    arm_control = G1ArmSwingControl(
      representation.fk.joint_names, wm.action_names, wm.dynamics.action_scale, cfg.arm_swing_control,
      offset=wm.dynamics.action_offset, roll=cfg.arm_swing_roll,
      speed_gate=cfg.arm_swing_speed_gate,
    )
  gait_adapter = None
  if cfg.forward_gait_action_limit:
    gait_adapter = ForwardGaitAdapter(
      obs_dim, wm.action_names, wm.dynamics.action_scale, cfg.forward_gait_action_limit,
      use_phase=cfg.forward_gait_phase,
      full_body=cfg.forward_gait_full_body,
    )
    if cfg.forward_gait_reflection:
      if not isinstance(representation, GenCoordRepresentation):
        raise ValueError("Walking reflection requires G1 GenCoord state.")
      gait_adapter.reflection = G1SagittalReflection(
        representation.fk.joint_names, wm.action_names, wm.dynamics.action_scale,
        wm.dynamics.action_offset, cfg.forward_gait_reflection,
      )
  steering_adapter = None
  if cfg.steering_recovery_limit:
    steering_adapter = SteeringRecoveryAdapter(
      obs_dim, wm.action_names, wm.dynamics.action_scale, cfg.steering_recovery_limit
    )
  return MultiEncoderAE(
    obs_dim=obs_dim,
    action_dim=wm.num_actions,
    task_specs=resolve_task_specs(cfg.tasks, representation, wm.num_goals, device),
    slot_mix=cfg.slot_mix,
    latent_dim=cfg.latent_dim,
    hidden_dims=cfg.actor_hidden_dims,
    latent_bc_weight=cfg.latent_bc_weight,
    history_length=cfg.history_length,
    arm_swing_control=arm_control,
    separate_stop_encoder=cfg.stop_encoder_init_dir is not None,
    forward_gait_adapter=gait_adapter,
    steering_recovery_adapter=steering_adapter,
  ).to(device)


# 函数作用：用另一条 W&B 运行记录的世界模型初始化当前模型。
def load_wandb_checkpoint(wandb_run_path: str, wm: WorldModel, device: str) -> None:
  """用另一条 W&B 运行记录的世界模型初始化当前模型。

  参数 wandb_run_path 是运行记录路径；wm 是已经构建好的目标容器；device
  用于 torch.load 的 map_location，允许权重加载到本次选择的设备。
  返回 None，原地改变 wm 内模型和上下文状态。

  流程：查询运行记录 → 找到并下载最新模型 artifact 所在目录 → 加载该目录的
  world_model.pt → 调用 load_bundle。此入口只初始化世界模型，actor/critic
  和优化器仍由本次运行重新建立；也不会读取旧回放数据或接续旧循环步数。
  """
  print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
  run = wandb.Api().run(wandb_run_path)
  wm.load_bundle(
    torch.load(download_model_dir(run) / "world_model.pt", map_location=device)
  )


class Ratio:
  """把可能小于 1 的更新频率转换成每次采集后应执行的整数更新次数。

  run 每采集一次就调用本对象。ratio=0 表示不更新，ratio<0 表示每次更新一次；
  正值在首次调用安排一次更新，之后根据累计步数差兑现整数次更新并保留小数进度。
  例如 ratio=0.25 时，首次返回 1，此后通常隔四步返回 1；ratio=2 时，
  首次返回 1，之后每隔一步可返回 2。

  _prev 代表上次已兑现更新对应的进度位置，虽然初始化注解为 int | None，
  后续除法会使其成为浮点进度。该状态跨调用保留，因此不能每轮重新创建 Ratio。
  """
  # 函数作用：保存频率并将已兑现进度标记为尚未初始化。
  def __init__(self, ratio: float) -> None:
    """保存频率并将已兑现进度标记为尚未初始化。

    ratio 是每个采集循环对应的更新频率，而非训练 batch 大小。
    初始化不消耗训练步；第一次 __call__ 才记录调用者传入的 step。
    返回 None，状态由 _ratio 和 _prev 两个成员保存。
    """
    self._ratio = ratio
    self._prev: int | None = None

  # 函数作用：根据当前采集循环编号，返回本次到期的整数更新次数。
  def __call__(self, step: int) -> int:
    """根据当前采集循环编号，返回本次到期的整数更新次数。

    参数 step 会转为 int，正常由 run 的递增循环编号提供。零和负频率先特殊处理；
    正频率首次调用返回 1。后续取 int((step-_prev)*ratio)，再只推进已兑现部分，
    这样低频率的小数进度不会每步都被丢弃。

    返回值交给 range 使用，可能为 0、1 或大于 1。此函数会更新 _prev，因此
    对同一对象重复查询也有状态含义；它不是单纯的 step 到次数的无状态公式。
    """
    step = int(step)
    # Ratio 是有状态的频率调度器：0 禁用更新，负值表示每次都更新一次。
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    # 将累计到期的更新次数取整；不足 1 的部分留在时间差里，后续调用继续累积。
    # 例如 ratio=0.25，大致每 4 次采集安排一次更新，而非每次都截断为 0。
    repeats = int((step - self._prev) * self._ratio)
    # 只推进已兑现更新对应的进度，保留小数欠账，避免长期更新频率偏低。
    self._prev += repeats / self._ratio
    return repeats


class BaseRunner[CfgT: BaseRunnerCfg]:
  """为不同训练算法提供共享基础设施的有状态组织器。

  构造时接收已创建的物理环境 env、配置 cfg、任务名、可选动作 registry 来源
  及 device；先建立世界模型、回放、模型学习器，再让子类建立策略，最后建立日志。
  cfg 类型由 CfgT 限定，class BaseRunner[CfgT: BaseRunnerCfg] 表示子类可传入
  公共配置的扩展类型，方便静态检查识别子类字段，不会改变运行时训练算法。

  本类持有 _wm 与其成员的快捷引用、motion_cmd、action_manager、replay_buffer、
  world_model_learner，以及归一化待处理队列与真实观察历史。下划线是内部约定，
  不表示这些张量被复制或自动脱离梯度图。

  _init_policy 和 _update_actor_normalization 由子类实现，基类版本会报错。
  _build_mlp/_create_wm_env 是共享工厂，另一个 PPO runner 也可复用它们；
  SuperMimicRunner 的多任务策略主路径使用自己的网络与想象环境工厂。
  """

  cfg: CfgT

  # 函数作用：保存运行依赖，并按固定顺序初始化公共设施、策略和日志。
  def __init__(
    self,
    env,
    cfg: CfgT,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
  ):
    """保存运行依赖，并按固定顺序初始化公共设施、策略和日志。

    参数 env 已由训练入口创建；cfg 为具体 runner 配置；task_id 用于日志目录和
    运行名称；registry_name 仅用于记录动作来源；device 用于模型、缓存和张量。
    返回 None，初始化后的成员由 step/run 持续复用。

    _init_common 必须先于 _init_policy：子类策略初始化要读取世界模型、动作维数
    和经验池。self._init_policy() 会调用实际子类的覆盖方法，而非强制调用基类版本。
    Logger 在策略初始化完成后创建，只负责统计与输出，不建立策略或模型优化器。
    """
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device

    # 先建立世界模型和经验池，再初始化策略；后者会引用前者，顺序不可调换。
    # 调用 self._init_policy 会分派到子类实现，属于模板式初始化。
    self._init_common()
    self._init_policy()

    self.logger = Logger(
      run_name=cfg.run_name,
      num_envs=env.num_envs,
      max_episode_length=env.max_episode_length,
      device=device,
    )

  # 函数作用：建立所有 runner 共享的模型、环境引用、经验池与模型学习器。
  def _init_common(self) -> None:
    """建立所有 runner 共享的模型、环境引用、经验池与模型学习器。

    无显式参数，读取 self.cfg/self.env/self.device。先创建 WorldModel，并把其中
    对象保存在 _representation/_dynamics/_sim_context 等快捷成员；可选加载旧模型。
    随后从底层环境取得动作管理器和 motion 命令，按动态 schema 创建 [E,容量,...]
    的 ReplayBuffer，再把同一 Dynamics 交给 WorldModelLearner。

    返回 None；还初始化归一化队列与历史缓存。此时尚未建立 policy_learner，
    因此该函数不能依赖 actor。旧模型加载只影响世界模型，不会预填经验池。
    """
    # 把模型组合对象保存在 _wm，并保留常用成员引用；这些引用指向同一网络，并非副本。
    wm = build_world_model(self.cfg, self.env, self.device)
    self._wm = wm
    self._representation = wm.representation
    self._dynamics = wm.dynamics
    self._sim_context = wm.sim_context
    self._num_goals = wm.num_goals
    self._num_actions = wm.num_actions

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)
    if self.cfg.init_checkpoint_dir is not None:
      path = Path(self.cfg.init_checkpoint_dir).expanduser() / "world_model.pt"
      wm.load_bundle(torch.load(path, map_location=self.device, weights_only=True))

    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion

    # 字段 schema 由表示维数动态生成；buffer_size 是每个环境的时间长度。
    # 创建在训练设备上，后续采样可直接交给学习器，代价是缓存也占用该设备内存。
    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
    )
    self.world_model_learner = WorldModelLearner(
      self._dynamics, self._representation, self.cfg.wm, device=self.device
    )

    # 采集在 inference_mode 内执行，统计量更新延后到该作用域之外。
    # 队列保存当前状态、动作和策略输入快照，避免执行下一步后底层缓存被改写。
    self._pending_norm_updates: list[tuple] = []
    self._sim_history: torch.Tensor | None = None

  # 函数作用：在物理环境重置后，为长历史策略填充初始观察窗口。
  def _seed_sim_history(self) -> None:
    """在物理环境重置后，为长历史策略填充初始观察窗口。

    读取当前环境 actor 传感器 [E,D]，添加一次传感器噪声，再扩展重复为 [E,H,D]。
    由于没有重置前的合法历史，全部槽位复用同一个带噪帧，而非各自加独立噪声。
    返回 None，结果保存在 _sim_history。H=1 时立即返回，由单帧路径直接组装观察。

    调用者是 run，顺序必须位于 env.reset() 之后、首次 step() 之前；否则可能
    使用旧回合或未初始化的数据。后续单环境重置由 _advance_sim_history 处理。
    """
    if self.cfg.history_length == 1:
      return
    seed = self._representation.add_proprio_noise(
      self.env.get_observations()["actor"].to(self.device)
    )
    # [E,D] 先增加历史轴为 [E,1,D]，再重复为 [E,H,D]。
    # 刚重置没有过去 H 帧，重复同一个带噪样本，保持与想象预热的填充约定一致。
    self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

  # 函数作用：给本次物理采集提供统一形状的观察历史。
  def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
    """给本次物理采集提供统一形状的观察历史。

    参数 sensor 为本步 [E,D] 状态或传感器输入。H=1 时加一条历史轴并返回 [E,1,D]；
    H>1 时返回已由历史环维护的 _sim_history:[E,H,D]，此时不使用本参数替换末帧。
    最新历史已在上一次 _advance_sim_history 写入，初始历史由 _seed_sim_history 建立。

    不更新历史、不重新加噪。若长历史缓存仍是 None，assert 会暴露初始化顺序问题。
    """
    if self.cfg.history_length == 1:
      # 单帧模式也保留历史轴，输出 [E,1,D]，使解码器不必为 H=1 特判。
      return sensor.unsqueeze(1)
    assert self._sim_history is not None
    return self._sim_history

  # 函数作用：在物理 step 后追加下一观察，并为结束的环境重新填充历史。
  def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
    """在物理 step 后追加下一观察，并为结束的环境重新填充历史。

    参数 next_actor:[E,D] 来自环境返回的下一传感器观察，dones:[E] 标记该步结束。
    H=1 时无需维护历史；否则加一次噪声，将旧窗口去掉首帧并追加新帧。
    对 done 环境，用新帧广播替换整段 [H,D]，防止上一回合数据进入新回合策略输入。

    返回 None，原地更新成员 _sim_history。此实现依赖环境在结束步返回重置后的
    下一观察；它与经验池 sample_causal 的历史填充约定相配合。
    """
    # 【源码讲解】
    # 正常环境移除最旧帧并追加最新传感器；已结束环境将所有历史位重填。
    # 重填复用同一个带噪新帧，避免把前一回合历史混入新回合。
    if self.cfg.history_length == 1:
      return
    assert self._sim_history is not None
    next_sensor = self._representation.add_proprio_noise(next_actor)
    shifted = torch.cat([self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1)
    # [E] 变为 [E,1,1]，可广播到整段 [E,H,D] 历史；一个环境结束就替换其全部历史。
    reset = dones.bool().view(-1, 1, 1)
    self._sim_history = torch.where(reset, next_sensor.unsqueeze(1), shifted)

  # 函数作用：计算扁平策略输入和价值网络输入的特征宽度。
  def _policy_input_dims(self) -> tuple[int, int]:
    """计算扁平策略输入和价值网络输入的特征宽度。

    无显式参数，读取当前表示、动作数、目标宽度、上下文宽度和历史长度。
    返回 (actor_dim, critic_dim)：前者为 H 帧状态/传感器加策略目标，后者为
    R+G+C，包含完整模型状态、完整世界模型目标和物理上下文。

    这两个维数服务不同接口，不能互换。SuperMimicRunner 只使用返回的 critic_dim，
    其多编码器 actor 由 build_actor 分别设置历史和各任务目标宽度。
    """
    actor_dim = self._representation.actor_obs_dim(
      self._num_actions, self._num_goals, self.cfg.history_length
    )
    # 价值网络使用模型状态、完整世界模型目标和上下文，拥有比部署传感器更多的信息。
    critic_dim = (
      self._representation.rep_state_dim
      + self._num_goals
      + self._sim_context.output_dim
    )
    # 返回顺序固定为策略宽度、价值网络宽度；多编码器 actor 另有构建函数，不能混用两者输入。
    return actor_dim, critic_dim

  # 函数作用：创建接受命名 TensorDict 观察的公共 rsl_rl 网络。
  def _build_mlp(
    self,
    name: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    **kwargs: object,
  ) -> nn.Module:
    """创建接受命名 TensorDict 观察的公共 rsl_rl 网络。

    参数 name 指输入字段名；input_dim/output_dim 是最后一维宽度；hidden_dims
    列出隐藏层；kwargs 透传给 MLPModel，例如激活函数或分布配置。函数先造 [1,D]
    的虚拟观察以声明接口，再启用观察归一化并将网络迁移到 self.device。
    返回 nn.Module，虚拟样本仅供构建，不写回放也不用于优化。

    当前 SuperMimicRunner 的 critic 用 models.build_mlp 创建普通张量网络；
    这个带下划线的工厂主要供共享 BaseRunner 的其他 runner 使用。
    """
    # 构造只有一个虚拟样本的观察字典，让 MLPModel 推断名称对应的特征宽度。
    # 这里只描述网络接口，不是真实采集样本，也不写入经验池。
    obs = TensorDict(
      {name: torch.zeros(1, input_dim, device=self.device)},
      batch_size=[1],
      device=self.device,
    )
    return MLPModel(
      obs=obs,
      obs_groups={name: [name]},
      obs_set=name,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      obs_normalization=True,
      **kwargs,
    ).to(self.device)

  # 函数作用：创建单任务动作跟踪想象环境的公共工厂。
  def _create_wm_env(self, batch_size: int) -> MotionTrackingEnv:
    """创建单任务动作跟踪想象环境的公共工厂。

    参数 batch_size 是想象并行数 B，允许与物理环境数 E 不同。返回
    MotionTrackingEnv，它共享本 runner 的 Dynamics、表示、物理任务参考和经验池，
    并使用本配置的历史长度与软终止平滑度。这里只构造对象，不采集真实轨迹。

    当前 SuperMimicRunner._init_policy 直接创建 MultiTaskWorldModelEnv，
    因此阅读主训练路径时不能把本工厂当成其实际调用点。
    """
    # 创建单任务想象环境的公共工厂；传入模型和经验池引用而非复制。
    # 当前 SuperMimicRunner 的主初始化另行使用 MultiTaskWorldModelEnv。
    return MotionTrackingEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=batch_size,
      replay_buffer=self.replay_buffer,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )

  # 函数作用：判断经验池是否必须保存真实传感器，以预热长历史想象起点。
  @property
  def _stores_actor_obs(self) -> bool:
    """判断经验池是否必须保存真实传感器，以预热长历史想象起点。

    这是 property，调用时写 self._stores_actor_obs 而不是加括号。
    只有 actor_obs_mode 为 proprio 且 history_length>1 时返回 True；其他情况
    可从单帧状态构造起点，无需额外保存每步传感器。

    返回布尔值，同时控制 _buffer_schema 的字段和 step 中 extra_fields 的写入，
    两处必须一致，否则 ReplayBuffer.insert 的 schema 检查会失败。
    """
    return (
      self._representation.actor_obs_mode == "proprio" and self.cfg.history_length > 1
    )

  # 函数作用：声明经验池每条物理转移保存哪些字段及其类型和宽度。
  def _buffer_schema(self) -> dict[str, FieldSpec]:
    """声明经验池每条物理转移保存哪些字段及其类型和宽度。

    无参数，返回字典 str→FieldSpec。state 是动作前完整状态 [E,S]；action 和
    prev_action 分别为本步原始动作与上一步动作 [E,A]；motion_time 是动作前参考帧；
    terminated/truncated 保存本步转移的两种结束标志。sim_ctx 保存 [E,C] 物理条件，
    长历史 proprio 另存 [E,D] 的 actor_obs。

    FieldSpec 默认浮点向量，帧号用 long，结束信号用 bool；标量不额外添加尾轴。
    输入记录 [E,...] 在回放中存储为 [E,capacity,...]，时间轴位于环境轴之后。
    这里不存 next_state，后续相邻记录
    的 state 充当下一帧标签，因此采样时必须排除跨回合窗口。
    """
    # 【源码讲解】
    # 字段宽度由当前表示、动作与上下文推断，避免固定机器人维度。
    # 仅 proprio 长历史路径保存 actor_obs，减少不需要的缓存占用。
    schema = {
      "state": FieldSpec(self._representation.full_state_dim),
      "action": FieldSpec(self._num_actions),
      "prev_action": FieldSpec(self._num_actions),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }
    if self._stores_actor_obs:
      schema["actor_obs"] = FieldSpec(
        self._representation.proprio_obs_dim(self._num_actions)
      )
    schema["sim_ctx"] = FieldSpec(self._sim_context.output_dim)
    return schema

  # 函数作用：约定子类必须建立策略训练所需成员，基类没有默认算法。
  def _init_policy(self) -> None:
    """约定子类必须建立策略训练所需成员，基类没有默认算法。

    由 BaseRunner.__init__ 在公共模型与经验池创建后调用；子类应据此构造 actor、
    相应想象环境及 policy_learner。成功实现返回 None，并把需要跨步复用的对象
    保存为实例成员。

    本方法体直接抛出 NotImplementedError，提醒调用者不能把 BaseRunner 当成
    完整训练器使用。SuperMimicRunner 与其他 runner 可以提供不同策略训练实现。
    """
    # 基类只约定接口，由具体 runner 提供实现；直接调用基类对应方法会明确报错。
    raise NotImplementedError

  # 函数作用：执行一次物理环境转移，并保存与动作前状态严格对齐的记录。
  def _step_and_store(
    self,
    actions: torch.Tensor,
    curr_states: torch.Tensor,
    actor_td: TensorDict | None = None,
    extra_fields: dict[str, torch.Tensor] | None = None,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """执行一次物理环境转移，并保存与动作前状态严格对齐的记录。

    参数 actions:[E,A] 是原始策略动作；curr_states:[E,S] 是执行前完整状态；
    actor_td 可选，为当前策略输入，提供时排入归一化更新队列；extra_fields
    附加 sim_ctx 和可选 actor_obs，其键必须与 _buffer_schema 对应。

    先复制参考帧、当前动作和上一动作，再调用 env.step，因为动作管理器和参考帧
    都会被推进。用 extras.time_outs 区分 truncated 与其他 terminated，写回放时
    记录 s_t/a_t 和这次转移的结束标志。返回环境原本的四元组
    (next_obs,rewards,dones,extras)，其中 rewards/dones 通常为 [E]。

    当前记录不把自动重置后的 next_obs 当作显式监督标签；由窗口采样检查边界。
    函数还会复制用于统计的张量，避免归一化队列引用被下一次 step 改写的数据。
    """
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    # 环境 step 会推进参考帧，所以提前复制，确保记录的是执行动作前的帧号。
    motion_times = motion_cmd.time_steps.clone()

    # detach 断开动作计算图，clone 保存独立快照；物理采集的数据作为监督标签存储。
    # prev_action 在执行本步前读取，否则动作管理器更新后会丢失上一动作。
    if self.env.clip_actions is not None:
      actions = actions.clamp(-self.env.clip_actions, self.env.clip_actions)
    raw_actions = actions.detach().clone()
    prev_action = self.action_manager.action.clone()

    # 物理仿真执行动作；回放中存放的 curr_states 是这次执行之前的状态。
    next_obs, rewards, dones, extras = self.env.step(actions)

    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    # 将总体结束信号扣除超时，得到非超时终止。
    # 回放采样用两种信号共同划分回合；return 采样稍后还会单独区分片段结束。
    terminated = dones.bool() & ~truncated

    fields: dict[str, torch.Tensor] = {
      "state": curr_states,
      "action": raw_actions,
      "prev_action": prev_action,
      "motion_time": motion_times,
      "terminated": terminated,
      "truncated": truncated,
    }
    # 将上下文及可选真实传感器字段合并；最终键集合必须与 schema 完全一致。
    if extra_fields:
      fields.update(extra_fields)
    self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))

    # 只有提供策略输入时才排队刷新归一化统计。克隆保证队列不引用可变环境缓冲。
    if actor_td is not None:
      self._pending_norm_updates.append(
        (
          curr_states.clone(),
          raw_actions.clone(),
          actor_td.detach().clone(),
        )
      )
    return next_obs, rewards, dones, extras

  # 函数作用：在物理采集的推理作用域之外，消费待处理样本并刷新输入统计。
  def _flush_normalization(self) -> None:
    """在物理采集的推理作用域之外，消费待处理样本并刷新输入统计。

    无参数，读取 _pending_norm_updates 中的 (states,actions,actor_td) 三元组。
    队列为空直接返回；否则合并新状态/动作，更新世界模型归一化器，再调用子类的
    策略归一化接口，最后清空队列。返回 None，不在此处执行优化器参数更新。

    归一化统计是网络 buffer，刷新它与梯度训练不同；延后处理是为了使这些训练
    相关状态更新发生在 inference_mode 之外。同一数据不应重复计入统计。
    """
    if not self._pending_norm_updates:
      return
    # 把多组三元组转为三组样本序列，再沿 batch 轴合并状态和动作。
    # 归一化更新完清空队列，防止同一批样本被重复计入统计。
    states, actions, actor_inputs = zip(*self._pending_norm_updates, strict=True)
    self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
    self._update_actor_normalization(actor_inputs)
    self._pending_norm_updates.clear()

  # 函数作用：约定子类如何消费新采集策略输入并更新观察统计。
  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    """约定子类如何消费新采集策略输入并更新观察统计。

    参数 actor_inputs 是若干 TensorDict 组成的元组，每项是一个采集批次的策略输入。
    返回 None，副作用由子类实现；基类不知道 actor 用扁平输入还是多任务目标，
    因此直接抛出 NotImplementedError，不在这里猜测观察布局。

    调用点位于 _flush_normalization，子类无需自己再次收集样本或推进环境。
    """
    # 基类只约定接口，由具体 runner 提供实现；直接调用基类对应方法会明确报错。
    raise NotImplementedError

  # 函数作用：创建本次运行的本地日志目录并返回 Path 对象。
  def _create_log_dir(self, *extra_parts: str) -> Path:
    """创建本次运行的本地日志目录并返回 Path 对象。

    可变参数 extra_parts 是夹在任务名与时间戳之间的额外目录层级；当前
    SuperMimicRunner.run 不传额外层级。目录为 logs/任务名/额外层级/时间戳，
    相对于程序启动时的工作目录，而非相对于本源码文件的位置。

    mkdir(parents=True,exist_ok=True) 负责创建父目录并允许已有目录。
    返回路径稍后用于保存 world_model.pt、actor.pt、可选 critic.pt 及导出文件。
    """
    # 日志目录相对于程序启动工作目录，以任务名和时间区分运行。
    # extra_parts 可由其他 runner 增加子目录层级；parents=True 自动创建父目录。
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("logs", self.task_id, *extra_parts, timestamp)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    return log_dir

  # 函数作用：建立本次实验记录，保存配置并关联 registry 动作数据来源。
  def _init_wandb(self, log_dir: Path, name_suffix: str = "") -> None:
    """建立本次实验记录，保存配置并关联 registry 动作数据来源。

    参数 log_dir 提供时间戳名称；name_suffix 可在默认运行名中附加标记。
    用 asdict 保存 env_cfg 与 runner_cfg，运行名优先采用 cfg.run_name。
    若 registry_name 非空，按逗号拆分 artifact 名称，缺版本时补 :latest，再用
    use_artifact 记录使用关系，便于播放查找参考动作来源。

    返回 None，副作用包括初始化全局 wandb.run 和访问 W&B 服务。
    它不上传模型、不下载动作，也不把本地动作路径自动变为 registry artifact。
    """
    # 把 dataclass 转为可序列化字典，记录实际训练配置。
    # 播放时依赖 runner_cfg 重建网络，仅保存权重不足以推断所有架构设置。
    wandb_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
    }

    default_name = f"{self.task_id}{name_suffix}-{log_dir.name}"
    wandb.init(
      project=self.cfg.wandb_project,
      name=self.cfg.run_name or default_name,
      config=wandb_config,
    )

    # 仅 registry 来源需要关联动作 artifact；本地动作入口已把 registry_name 清空。
    if self.registry_name:
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        # 未给 artifact 版本/别名时补 latest；use_artifact 记录本次运行所使用的数据来源。
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        wandb.run.use_artifact(artifact)

  # 函数作用：正常结束时关闭物理环境并结束当前 W&B 记录。
  def _cleanup(self) -> None:
    """正常结束时关闭物理环境并结束当前 W&B 记录。

    无参数，返回 None。getattr 查询可选 close 方法，存在时调用；随后
    wandb.finish 完成实验日志收尾。该方法本身不负责等待模型上传线程，调用者
    run 在执行它之前先 shutdown 上传线程池。

    当前 run 在正常循环结束时调用这里，没有用本方法包裹整个循环的异常清理；
    不要据此假设任意中途异常都会自动执行资源释放。
    """
    env_close_fn = getattr(self.env, "close", None)
    if env_close_fn:
      env_close_fn()
    wandb.finish()


class SuperMimicRunner(BaseRunner[SuperMimicRunnerCfg]):
  """本文件实际执行 SuperMimic 联合训练的具体组织器。

  继承 BaseRunner 的模型、经验池和物理采集工具，新增多编码器 actor、可选 critic、
  MultiTaskWorldModelEnv 和 return 重置采样器。构造过程沿用父类 __init__，
  由覆盖的 _init_policy 建立本算法专用对象；run 才真正进入训练循环。

  step 只做一次物理并行采集及统计刷新；run 根据 Ratio 决定每次采集后更新几次，
  每次先监督拟合世界模型，再通过想象轨迹更新策略。模型学习器和策略学习器分别
  拥有优化器，actor 更新阶段冻结动力学参数，但保留对动作输入的梯度。

  多任务在这里表示目标模态不同，而非每种任务使用独立奖励或独立机器人。
  各编码器分担连续批次槽位，共享动作解码器，统一通过动作跟踪奖励学习。
  """

  # 函数作用：在公共模型和回放准备好后，建立本算法的策略、想象环境及学习器。
  def _buffer_schema(self) -> dict[str, FieldSpec]:
    schema = super()._buffer_schema()
    if self.cfg.independent_velocity:
      schema.update(
        velocity_command=FieldSpec(3),
        velocity_hold_steps=FieldSpec(dtype=torch.long),
        velocity_command_category=FieldSpec(dtype=torch.long),
      )
      if self.cfg.forward_gait_phase:
        schema["gait_phase"] = FieldSpec()
    return schema

  def _init_policy(self) -> None:
    """在公共模型和回放准备好后，建立本算法的策略、想象环境及学习器。

    无显式参数，读取 cfg 和父类成员。先 build_actor，再筛出 slot_mix>0 的任务；
    输出物理槽位区间，计算 R+G+C 的 critic 输入宽度并建标量价值网络。
    随后用 policy.batch_size 创建共享 Dynamics 和回放的多任务想象环境，
    可用 CUDA 时编译 reset/step，检查并建立重置采样器，最后按配置类型选择
    ActorCriticPolicyLearner 或 BpttPolicyLearner。

    返回 None，写入 wm_env、policy_learner、_active_task_specs 与 _sampler。
    当前代码即使选择纯 BPTT 也先构建 critic，但该分支不把它传给学习器。
    任务列表须保留 motion 作为锚点；该约束由多任务环境进一步检查。
    """
    actor = build_actor(self.cfg, self._wm, self.device)
    if self.cfg.init_checkpoint_dir is not None:
      path = Path(self.cfg.init_checkpoint_dir).expanduser() / "actor.pt"
      state = torch.load(path, map_location=self.device, weights_only=True)
      if self.cfg.stop_encoder_init_dir is not None and "stop_encoder.0.weight" not in state:
        stop_path = Path(self.cfg.stop_encoder_init_dir).expanduser() / "actor.pt"
        source = torch.load(stop_path, map_location=self.device, weights_only=True)
        prefix = "encoders.velocity."
        state.update({
          f"stop_encoder.{name.removeprefix(prefix)}": value
          for name, value in source.items() if name.startswith(prefix)
        })
      for adapter_name in ("forward_gait_adapter", "steering_recovery_adapter"):
        prefix = f"{adapter_name}."
        if getattr(actor, adapter_name) is not None and not any(name.startswith(prefix) for name in state):
          state.update({name: value for name, value in actor.state_dict().items() if name.startswith(prefix)})
      if self.cfg.forward_gait_full_body:
        assert isinstance(actor.forward_gait_adapter, ForwardGaitAdapter)
        actor.forward_gait_adapter.expand_checkpoint(state, "forward_gait_adapter.")
      actor.load_state_dict(state)
    self._velocity_command = None
    if self.cfg.independent_velocity:
      actor.requires_grad_(False)
      if actor.steering_recovery_adapter is not None and self.cfg.train_steering_recovery:
        actor.steering_recovery_adapter.requires_grad_(True)
      elif actor.forward_gait_adapter is not None:
        actor.forward_gait_adapter.requires_grad_(True)
      else:
        if actor.stop_encoder is not None:
          actor.stop_encoder.requires_grad_(True)
        if actor.stop_encoder is None or not self.cfg.freeze_moving_encoder:
          actor.encoders["velocity"].requires_grad_(True)
      term = self.env.unwrapped.command_manager.get_term("velocity")
      assert isinstance(term, IndependentVelocityCommand)
      self._velocity_command = term
      if self.cfg.forward_gait_teacher_steps:
        gait = self.env.unwrapped.reward_manager.get_term_cfg("forward_gait").func
        warm_start_gait_adapter(
          actor, self._representation, gait, self.cfg.forward_gait_teacher_steps,
          self._wm.action_names,
        )
    # 只为占有非零批次比例的任务生成目标；编码器仍按配置的任务顺序分配槽位。
    self._active_task_specs = [
      ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name, 0.0) > 0.0
    ]

    print("[INFO] Sim slot ranges:")
    for ts, (lo, hi) in zip(
      actor.task_specs, actor.slot_ranges(self.env.num_envs), strict=True
    ):
      print(f"  {ts.name}: [{lo}, {hi}) — goal_dim={ts.goal_dim}")
    # 只取价值网络输入宽度；actor 已由 build_actor 根据多任务接口创建。
    # 价值网络输出 1 个标量，即每个想象状态的未来折扣回报估计。
    _, critic_input_dim = self._policy_input_dims()
    if self._velocity_command is not None:
      critic_input_dim += 4 + 2 * self.cfg.forward_gait_phase - self._num_goals
    critic = build_mlp(
      critic_input_dim,
      self.cfg.critic_hidden_dims,
      output_dim=1,
    ).to(self.device)

    policy_cfg = self.cfg.policy

    # 想象并行数来自 policy.batch_size，不要求等于物理环境数。
    # 共享 _dynamics 与 replay_buffer，所以模型更新后想象环境立即使用最新参数。
    wm_env_kwargs = dict(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=policy_cfg.batch_size,
      replay_buffer=self.replay_buffer,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )
    if self._velocity_command is not None:
      self.wm_env = VelocityWorldModelEnv(**wm_env_kwargs)
    else:
      self.wm_env = MultiTaskWorldModelEnv(
        **wm_env_kwargs, task_specs=self._active_task_specs
      )
    # CUDA 可用时编译 reset 和 step 的张量路径以降低 Python 开销；算法逻辑不变。
    if torch.cuda.is_available():
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)

    self._configure_reset_sampling()

    # 按配置对象类型选择学习器；两者都通过世界模型反传，区别是是否增加价值补偿与拟合。
    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor, critic, self.wm_env, policy_cfg, device=self.device
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor, self.wm_env, policy_cfg, device=self.device
      )

  # 函数作用：把物理状态与参考时刻转换成多任务策略当前需要的输入。
  def _build_sim_obs_td(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_actor_obs: torch.Tensor,
    motion_times: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> TensorDict:
    """把物理状态与参考时刻转换成多任务策略当前需要的输入。

    参数 rep_state:[E,R]、aux_state:[E,7] 分别是模型表示和根位置/四元数；
    sim_actor_obs:[E,D] 是真实传感器，motion_times:[E] 是动作前参考帧号，
    sim_ctx:[E,C] 是归一化物理条件。proprio 模式选真实传感器并加噪，
    rep_state 模式直接使用表示；再按每个活跃 TaskSpec 提取整批目标。

    返回 TensorDict：history:[E,H,D或R] 加 goal_任务名:[E,G_task]。
    采集 actor 不需 critic 输入。H>1 时历史来自已维护的 _sim_history，
    本函数新算的单帧传感器不会替换历史环，也不会推进参考时间。
    """
    # 物理采集可直接读取实际传感器；rep_state 模式则给 actor 转换后的模型状态。
    proprio = self.cfg.actor_obs == "proprio"
    sensor = self._representation.compute_actor_sensor(
      rep_state,
      sim_ctx,
      aux_state=aux_state,
      sim_actor_obs=sim_actor_obs if proprio else None,
      noisy=proprio,
    )
    # 每个活跃任务为整批环境生成目标；actor 再按槽位区间选择对应任务目标。
    # 目标必须使用动作执行前的 motion_times，与当前状态和输出动作时间对齐。
    if self._velocity_command is None:
      goals = {
        ts.name: ts.extract_training_goal(self.motion, motion_times, aux_state)
        for ts in self._active_task_specs
      }
    else:
      # The command manager mutates its buffers after the physics step.
      goals = {"velocity": self._velocity_command.command.clone()}
    actor_td = build_actor_obs_td(goals, self._collection_history(sensor))
    if self._velocity_command is not None and self._velocity_command.gait_phase is not None:
      actor_td["gait_phase"] = self._velocity_command.gait_phase.clone()
    return actor_td

  # 函数作用：确认入口正确解析了重置模式，并按需创建 return 采样器。
  def _configure_reset_sampling(self) -> None:
    """确认入口正确解析了重置模式，并按需创建 return 采样器。

    无显式参数，读取 motion_cmd.cfg.sampling_mode 与 runner 配置。return 在
    物理命令中应表现为 external，其余模式要求名称直接一致；不一致则报错。
    非 return 模式保留 _sampler=None，后续由 mjlab 原生采样规则决定重置起点。

    return 模式按物理环境数、总参考帧、桶数和最大奖励创建 ReturnSampler。
    返回 None。该函数不立刻推进环境，也不改任务槽位比例；分布会在 step 收集
    足够回报后通过 motion_cmd.set_sampling_distribution 发布。
    """
    self._sampler: ReturnSampler | None = None
    env_mode = self.motion_cmd.cfg.sampling_mode
    # return 是 runner 的策略名，mjlab 对应 external，由 runner 定期写入外部分布。
    # 环境创建后再改该模式可能不完整，因此这里检查入口配置是否已正确解析。
    expected = (
      "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
    )
    if env_mode != expected:
      raise RuntimeError(
        f"runner sampling_mode='{self.cfg.sampling_mode}' needs the motion command "
        f"in '{expected}' mode but the env was built with '{env_mode}'; resolve it "
        "into the env cfg before env creation ('external' also needs an mjlab with "
        "MotionCommand external-mode support)"
      )
    if self.cfg.sampling_mode != "return":
      return
    self._sampler = ReturnSampler(
      cfg=self.cfg.return_sampler,
      bin_count=self.motion_cmd.bin_count,
      frame_count=self.motion.time_step_total,
      num_envs=self.env.num_envs,
      max_step_reward=self.wm_env.max_step_reward,
      device=self.device,
    )

  # 函数作用：采集一个物理控制步，为每个并行环境记录一条转移。
  def step(self) -> int:
    """采集一个物理控制步，为每个并行环境记录一条转移。

    无显式参数，读取当前环境观察。流程为：保存动作前状态与帧号 → 读取上下文，
    构造历史与任务目标 → actor 输出 [E,A] 并加探索噪声 → _step_and_store
    推进物理环境并入回放 → 更新历史及日志 → 可选更新参考动作重置概率。
    这些步骤位于 inference_mode，物理采集不建立反向传播图。

    return 采样把同时超时的失败视为超时，并从普通超时中排除片段结束，分别传入
    采样器。退出推理作用域后刷新归一化统计。返回整数 E 供 run 累计样本数。
    本函数不调用世界模型或策略优化器，不等于执行了一次训练参数更新。
    """
    # 物理仿真采集不反向传播，关闭计算图可节省内存；后续参数训练在此作用域外进行。
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)

      # 保存动作前帧号；env.step() 会原地修改 motion 命令的时间步。
      motion_times = self.motion_cmd.time_steps.to(self.device).clone()

      # 完整状态转成模型使用的表示；aux 保留积分和奖励计算需要的根位姿。
      rep_state = self._representation.represent(curr_states)
      aux_state = self._representation.extract_aux_state(curr_states)
      sim_ctx = self._sim_context.read(self.env.unwrapped)
      actor_td = self._build_sim_obs_td(
        rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx
      )

      # 采集时在策略输出上加随机噪声，以探索不同的状态和动作。
      actions = self.policy_learner.actor(actor_td)
      actions = actions + torch.randn_like(actions) * self.policy_learner.cfg.noise

      # 存储当前物理条件；长历史模式额外保存无此处人工加噪的实际 actor 观察。
      # 想象起点预热时再按统一规则加噪，避免把同一扰动重复叠加。
      extra_fields = {"sim_ctx": sim_ctx}
      if self._stores_actor_obs:
        extra_fields["actor_obs"] = sim_actor_obs
      if self._velocity_command is not None:
        extra_fields.update(
          velocity_command=actor_td["goal_velocity"],
          velocity_hold_steps=self._velocity_command.steps_left.clone(),
          velocity_command_category=self._velocity_command.category.clone(),
        )
        if self.cfg.forward_gait_phase:
          extra_fields["gait_phase"] = actor_td["gait_phase"]

      next_obs, rewards, dones, extras = self._step_and_store(
        actions,
        curr_states,
        actor_td=actor_td,
        extra_fields=extra_fields,
      )
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      # Logger 接收底层 terminated，而非总体 dones；它内部另按最大回合长度处理统计。
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

      if self._sampler is not None:
        # 同时发生失败与超时时，在采样回报统计中按超时处理。
        manager = self.env.unwrapped.termination_manager
        truncated = manager.time_outs.to(self.device, torch.bool)
        clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
        weights = self._sampler.record(
          motion_times,
          rewards,
          terminated.to(self.device, torch.bool) & ~truncated,
          clip_end,
          truncated & ~clip_end,
        )
        # 采样器未满窗口时返回 None；满窗口才向 motion 命令发布新的重置概率。
        if weights is not None:
          self.motion_cmd.set_sampling_distribution(weights)

    # 退出 inference_mode 后刷新归一化统计，避免训练相关缓冲区在推理上下文内更新。
    self._flush_normalization()
    # 一次 step 为 E 个物理环境各采集一条记录，返回 E 供训练循环累计样本量。
    return self.env.num_envs

  # 函数作用：更新多任务 actor 的最新传感器与目标归一化统计。
  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    """更新多任务 actor 的最新传感器与目标归一化统计。

    参数 actor_inputs 为新采集的策略输入快照元组，布局与 _build_sim_obs_td 返回值
    一致。先把 actor 设为训练模式，再逐批调用 MultiEncoderAE.update_normalization。
    该实现从历史取最新帧，并按任务槽位更新对应目标统计，避免重复历史过度计数。

    返回 None，仅刷新统计缓冲，不计算策略损失，不执行反向传播或 optimizer.step。
    调用者 _flush_normalization 在推理作用域外执行它。
    """
    # 经验归一化器通常只在训练模式接收统计更新；此处显式开启，逐批输入新数据。
    if self.cfg.independent_velocity:
      return
    self.policy_learner.actor.train()
    for actor_td in actor_inputs:
      self.policy_learner.actor.update_normalization(actor_td)

  # 函数作用：执行本次训练的完整生命周期，是训练入口最终调用的方法。
  def run(self) -> None:
    """执行本次训练的完整生命周期，是训练入口最终调用的方法。

    无参数，返回 None。先创建本地目录和 W&B 记录，重置物理环境并初始化历史，
    按采集量与模型批次规模计算 Ratio，然后进入 max_iterations 次采集循环。
    每轮先 step；对到期更新次数，若缓存够长则抽连续窗口训练世界模型，再训练策略。
    日志按 log_interval 汇总，检查点按 save_interval 保存并在单后台线程上传。

    一次模型 update 内含 wm.num_batches 个优化批次；一次策略 update 也可能含
    多个批次，因而外层循环数不是优化器 step 数。两个取模条件在 step=0 也成立，
    所以初始数据不足时仍可能先记录日志或保存尚未充分训练的模型。

    保存前等待上一上传任务；保存模型/actor/可选 critic，满足条件时导出 ONNX。
    检查点不含优化器、经验池或完整进度。循环正常结束后等待线程池并清理环境；
    当前实现未在主循环外围提供 try/finally，因此异常清理不由此流程保证。
    """
    # 【源码讲解】
    # 每次外层迭代先采集一次所有物理环境，再按 Ratio 决定更新次数。
    # 世界模型先学习新的物理数据，策略随后利用当前模型展开想象轨迹。
    # 初始缓存不足 horizon 时跳过更新，日志和 checkpoint 按配置间隔处理。
    log_dir = self._create_log_dir()
    if self.cfg.independent_velocity:
      (log_dir / "local_config.json").write_text(json.dumps({
        "task_id": self.task_id,
        "runner_cfg": asdict(self.cfg),
        "motion_file": self.motion_cmd.cfg.motion_file,
        "seed": self.env.cfg.seed,
        "num_envs": self.env.num_envs,
        "control_dt_s": self.env.unwrapped.step_dt,
      }, indent=2) + "\n")
    self._init_wandb(log_dir)

    self.env.reset()
    self._seed_sim_history()

    print(f"[INFO] Starting training loop for {self.cfg.max_iterations} iterations...")

    # 用并行采集量、模型训练批量和 replay_ratio 计算更新频率；
    # 一次外层循环不一定恰好对应一次模型更新。
    should_train = Ratio(
      self.env.num_envs
      / self.cfg.wm.num_batches
      / self.cfg.wm.batch_size
      / self.cfg.wm.horizon
      * self.cfg.replay_ratio
    )

    num_collected_steps = 0
    start_time = time.time()
    wm_learn_time = 0.0
    policy_learn_time = 0.0
    loss_dict: dict[str, torch.Tensor] = {}
    log_count = 0
    # 用单后台线程上传模型 artifact，避免网络上传阻塞每次训练。
    # upload 保存上次任务的 Future，用于下次覆盖文件前等待上传完成。
    uploader = ThreadPoolExecutor(max_workers=1)
    upload: Future | None = None

    for step in range(self.cfg.max_iterations):
      # 先收集数据再更新模型与策略；统计的是环境转移数，每个循环增加物理环境数。
      num_collected_steps += self.step()

      for _ in range(should_train(step)):
        # 检查每环境已填充的帧数；不足窗口长度不能训练。
        # 达到长度也仍需 sample 检查回合边界，频繁重置可能导致没有合法长窗口。
        if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
          continue

        # 每个批次抽取连续窗口；缓冲区会避免窗口内部跨越回合结束。
        # 这是惰性生成器：学习器消费时才逐批采样，不是提前存下所有样本的列表。
        data = (
          self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
          for _ in range(self.cfg.wm.num_batches)
        )

        start_wm_time = time.time()
        # 世界模型训练入口：学习从当前状态和动作预测仿真器的下一状态。
        wm_logs = self.world_model_learner.update(data)
        wm_learn_time += time.time() - start_wm_time

        # 先拟合世界模型，再用当前世界模型中的想象轨迹更新策略。
        start_policy_time = time.time()
        # BPTT 训练入口：在刚更新的可微世界模型中展开轨迹并优化 actor。
        policy_logs = self.policy_learner.update()
        policy_learn_time += time.time() - start_policy_time

        # 按指标名合并本次日志；同名键被新值覆盖，这里不对多次更新再求平均。
        loss_dict.update(wm_logs)
        loss_dict.update(policy_logs)

      if step % self.cfg.log_interval == 0:
        elapsed_time = time.time() - start_time
        total_train_time = wm_learn_time + policy_learn_time
        # 把本日志区间总时长减去两类学习耗时作为其余耗时估计。
        # 因此 collect_time 也可能含日志、保存或上传等待等开销，不是纯仿真计时。
        collect_time = elapsed_time - total_train_time

        self.logger.log(
          it=log_count,
          total_it=self.cfg.max_iterations // self.cfg.log_interval,
          collect_time=collect_time,
          wm_learn_time=wm_learn_time,
          policy_learn_time=policy_learn_time,
          loss_dict=loss_dict,
          num_collected_steps=num_collected_steps,
        )

        # 日志横轴计数独立于外层 step；清零本区间时长和样本计数，为下一日志窗口累计。
        log_count += 1
        start_time = time.time()
        wm_learn_time = 0
        policy_learn_time = 0
        num_collected_steps = 0

      if step % self.cfg.save_interval == 0 or (
        self.cfg.independent_velocity and step == self.cfg.max_iterations - 1
      ):
        assert wandb.run is not None  # 进入循环前已初始化 W&B 运行
        # 旧版本上传任务结束后才能覆盖本地同名文件。
        # 上传线程可能仍在读取本地文件；先等上个版本上传完，再覆盖同名检查点。
        if upload is not None:
          upload.result()
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(self._wm.bundle(), world_model_path)
        torch.save(self.policy_learner.actor.state_dict(), actor_path)
        # 基础检查点包含世界模型与 actor；有 critic 时额外保存。
        # 此处没有保存优化器、回放缓存和循环进度，不能视为完整训练状态快照。
        checkpoint_files = [world_model_path, actor_path]
        critic = getattr(self.policy_learner, "critic", None)
        if critic is not None:
          critic_path = str(log_dir / "critic.pt")
          torch.save(critic.state_dict(), critic_path)
          checkpoint_files.append(critic_path)
        # 总动作帧数过大时不进行 ONNX 导出。
        # 仅传感器策略且动作总帧数不大时导出 ONNX，避免大数据集的导出检查成本。
        if (
          self._representation.actor_obs_mode == "proprio"
          and self.motion.time_step_total <= 15_000
        ):
          checkpoint_files += export_actor_as_onnx(
            actor=self.policy_learner.actor,
            env=self.env.unwrapped,
            path=str(log_dir),
            run_name=wandb.run.name,
            term_widths=self._representation.proprio_term_widths(self._num_actions),
          )
        # 异步提交本次文件上传；返回 Future，供下一次保存前等待。
        upload = uploader.submit(log_model_artifact, checkpoint_files)

    # 正常循环结束时等待后台任务完成，再关闭物理环境并结束 W&B 运行。
    uploader.shutdown()
    self._cleanup()
