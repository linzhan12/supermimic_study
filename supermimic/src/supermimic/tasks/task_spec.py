"""定义多编码器策略的任务目标协议，不负责奖励或参数更新。

TaskSpec 将名称、目标宽度、训练目标提取回调和可选推理控件打包，供 runner、
多任务想象环境、viewer 与导出器共享。motion 任务使用参考动作目标，velocity
任务使用机体系三维速度命令；两者训练时仍使用同一动作跟踪奖励。

阅读顺序：控件数据类 → TaskSpec 接口 → 观察字典组装 → 任务工厂 → 任务配置解析。
B=批大小，H=传感器历史长度，D=每帧传感器宽度，G=某任务的目标宽度。
"""

# 【文件作用】多任务目标接口
# 规定任务目标的维度、训练提取方式及推理控件，提供动作跟踪和速度命令两种目标。
# 【阅读顺序】目标控件数据类 → TaskSpec → 观察打包 → motion/velocity 工厂 → 配置解析。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟类型注解求值，使 TYPE_CHECKING 内的类型仅用于静态检查而无需运行时导入。
from __future__ import annotations

# 描述任务回调的函数类型，明确训练提取、推理转换和回放预计算三种接口。
from collections.abc import Callable
# dataclass 为配置和任务定义生成初始化；field(default_factory=...) 为每个配置独立创建子对象。
from dataclasses import dataclass, field
# 仅静态类型检查时导入模型/表示类型，降低运行时循环导入的可能；实际对象由调用者传入。
from typing import TYPE_CHECKING

# PyTorch 提供张量索引、批次变换、自动微分和编译；本文件的数据形状约定由这些运算实现。
import torch
# 用根四元数逆旋转世界速度，得到与当前机体朝向相关的速度命令。
from mjlab.utils.lab_api.math import quat_apply_inverse
# 共享 batch_size 的张量字典，统一承载状态、历史、任务目标和训练批次，不只是任意键值集合。
from tensordict import TensorDict

# 从完整世界模型目标宽度推导 actor 目标宽度，排除仅模型使用的相对位置部分。
from supermimic.representations import actor_goal_dim

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionLoader

  from supermimic.representations import StateRepresentation


@dataclass(frozen=True)
class ContinuousGoalSpec:
  """描述一个连续原始命令维度的播放界面滑块。

  name 是界面标签，min/max 限定范围，step 是控件步长，default 是初值。
  这是 frozen dataclass，实例用于描述控件而非存储实时变化的用户命令。
  它只影响推理输入界面，不定义训练目标提取函数，也不直接改变奖励。
  """

  name: str
  min: float
  max: float
  step: float
  default: float


@dataclass(frozen=True)
class DiscreteGoalSpec:
  """描述一个只允许固定数值集合的原始命令维度。

  name 是下拉框标签，choices 是合法数值元组，default 是初值。
  与连续滑块一样，这个不可变数据类提供界面规范；实时命令值由 viewer 管理。
  任务可通过 goal_specs 组合多个连续或离散维度。
  """

  name: str
  choices: tuple[float, ...]
  default: float


type GoalSpec = ContinuousGoalSpec | DiscreteGoalSpec


ExtractFromMotion = Callable[["MotionLoader", torch.Tensor, torch.Tensor], torch.Tensor]
BuildInferenceGoalObs = Callable[[torch.Tensor], torch.Tensor]
PrecomputeReplayGoal = Callable[["MotionLoader"], torch.Tensor]


@dataclass(frozen=True)
class TaskSpec:
  """一个任务从训练数据或用户命令到目标编码器输入的完整接口。

  name 决定编码器键与 goal_名称 观察字段；goal_dim 是编码器实际输入宽度。
  extract_training_goal 接收 motion、帧索引 [B] 和辅助根状态 [B,7]，返回 [B,G]。
  推理时可用 goal_specs 描述控件，以 build_inference_goal_obs 将原始命令转换为目标；
  raw_goal_dim 与 goal_dim 允许不同，precompute_replay_goal 可提供无噪逐帧命令表。

  此类只保存定义，不分配槽位、不推进环境、不计算奖励。任务列表顺序由
  resolve_task_specs 明确创建，actor 的槽位比例另由 runner.slot_mix 决定。
  """

  name: str
  # 编码器实际接收的目标特征宽度，必须与 extract_training_goal 返回张量最后一维一致。
  goal_dim: int
  # 训练目标提取函数，接收参考动作、帧索引和当前根状态，返回整批目标。
  extract_training_goal: ExtractFromMotion
  # 描述播放界面的滑块或选项；这是原始用户命令的界面定义，不参与奖励计算。
  goal_specs: tuple[GoalSpec, ...] = ()
  # 把原始用户命令转换为编码器目标；None 表示该任务没有这条通用推理入口。
  build_inference_goal_obs: BuildInferenceGoalObs | None = None
  # 原始用户命令宽度可与编码器目标宽度不同，导出/播放需明确区分两者。
  raw_goal_dim: int = 0
  precompute_replay_goal: PrecomputeReplayGoal | None = None
  """可选的无噪逐帧命令表，供 ONNX 检查和 viewer 回放，形状为[总帧数,原始命令宽度]。"""


# 函数作用：按多编码器策略约定组装输入字典，保留共同的批次轴。
def build_actor_obs_td(
  goals_by_name: dict[str, torch.Tensor],
  history: torch.Tensor,
  critic: torch.Tensor | None = None,
) -> TensorDict:
  """按多编码器策略约定组装输入字典，保留共同的批次轴。

  history 为 [B,H,D]，最新帧在末尾；goals_by_name 的每项是 [B,G_task]；
  critic 可选，为 [B,D_critic] 的特权观察。返回 TensorDict，batch_size=[B]，
  字段为 history、goal_任务名 及可选 critic。所有字段必须与同一批状态对齐。
  函数只打包引用，不拷贝或选择槽位；actor.forward 再读取各任务分配的区间。
  """
  # 【源码讲解】
  # history 为 [B,H,D]，各任务目标为 [B,G_task]，critic 可选。
  # 键名 goal_任务名 是多编码器和想象环境之间的接口，不能任意重命名。
  # 各任务目标共享同一批次索引，策略根据任务分配选择对应目标。
  obs: dict[str, torch.Tensor] = {"history": history}
  if critic is not None:
    obs["critic"] = critic
  obs.update({f"goal_{name}": goal for name, goal in goals_by_name.items()})
  return TensorDict(obs, batch_size=[history.shape[0]], device=history.device)


# 函数作用：构建参考动作跟踪目标的 TaskSpec。
def make_motion_task(representation: "StateRepresentation", num_goals: int) -> TaskSpec:
  """构建参考动作跟踪目标的 TaskSpec。

  representation 提供参考窗口与根相对目标构造，num_goals 是完整世界模型目标宽度。
  返回名称为 motion 的任务，goal_dim 通过 actor_goal_dim 排除世界模型专用平移项。
  内部 extract 闭包保留 representation，以保证物理采集与想象环境采用同一目标布局。
  此任务不提供通用三维用户控件，也不在此处单独定义运动跟踪奖励。
  """
  # 【源码讲解】
  # 闭包捕获 representation，按参考帧和当前根位姿构建 actor 目标。
  # compute_wm_goal 的第一个返回值包含世界模型专用部分，此任务只取第二项。

  def extract(motion, motion_times, aux_state):
    """从指定参考时刻提取策略侧的动作窗口目标。

    motion 是动作加载器，motion_times:[B] 为当前参考帧，aux_state:[B,7] 提供当前根位姿。
    compute_wm_goal 同时返回完整目标和策略目标，此处只取后者，返回 [B,G_motion]。
    相对方向依赖当前根位姿，因此不能简单按帧查一张完全固定的目标表。
    """
    _, actor_goal = representation.compute_wm_goal(motion, motion_times, aux_state)
    return actor_goal

  return TaskSpec(
    name="motion",
    goal_dim=actor_goal_dim(num_goals),
    extract_training_goal=extract,
  )


# 函数作用：构建机体系 [vx,vy,wz] 三维速度命令任务。
def make_velocity_task(
  lin_vel_x_range: tuple[float, float],
  lin_vel_y_range: tuple[float, float],
  ang_vel_z_range: tuple[float, float],
  window_frames: int,
  command_noise: tuple[float, float, float],
  device: str = "cpu",
) -> TaskSpec:
  """构建机体系 [vx,vy,wz] 三维速度命令任务。

  三个 range 参数指定命令范围；window_frames 指定参考世界速度的中心平均长度；
  command_noise 为逐轴均匀噪声半宽；device 决定闭包常量的初始设备。
  返回 velocity TaskSpec，训练提取速度并加噪、裁剪；推理将用户三维命令直接作为目标。

  平均使用离线参考，允许查看未来帧以避免因果平均带来的滞后。当前平均边界按
  拼接动作总数组裁剪，不是逐片段裁剪，因此片段交界附近可混入相邻片段速度。
  """
  # 【源码讲解】
  # 速度任务把世界系参考线/角速度投影到当前根坐标系，取 vx、vy、wz。
  # 训练加逐轴均匀噪声并裁到范围；推理直接使用用户三维命令。
  # 中心平均窗口利用离线参考，不受在线因果限制；当前边界裁剪针对拼接总数组。
  lo = torch.tensor(
    [lin_vel_x_range[0], lin_vel_y_range[0], ang_vel_z_range[0]], device=device
  )
  hi = torch.tensor(
    [lin_vel_x_range[1], lin_vel_y_range[1], ang_vel_z_range[1]], device=device
  )
  noise_half = torch.tensor(command_noise, device=device)

  def _clamp(cmd):
    """将三维命令逐轴裁剪到配置范围并返回同形状张量。

    cmd 最后一维为3，lo/hi 按需迁移到 cmd.device 后广播。
    此处仅限制范围，不加噪、不进行坐标变换，训练和回放预计算复用同一路径。
    """
    return torch.clamp(cmd, lo.to(cmd.device), hi.to(cmd.device))

  def _centered_mean(track, motion_times):
    """在每个参考时刻周围取世界速度均值。

    track 通常是 [总帧数,3]，motion_times 为 [B]，返回 [B,3]。
    窗口长度0或1时直接取当前帧；否则生成中心偏移形成 [B,窗口长] 索引，
    裁到总数组边界再沿窗口维平均。偶数窗口的偏移按整数除法分配，不单独补对称帧。
    """
    if window_frames <= 1:
      return track[motion_times]
    half = window_frames // 2
    offsets = (
      torch.arange(window_frames, device=motion_times.device, dtype=motion_times.dtype)
      - half
    )
    # [B,1] 时刻与 [窗口长] 偏移广播为 [B,窗口长]；限制到拼接数组范围后取均值。
    # 当前实现不逐片段裁剪中心平均，片段边界附近可能混入相邻片段的速度。
    idx = (motion_times.unsqueeze(-1) + offsets).clamp(min=0, max=track.shape[0] - 1)
    return track[idx].mean(dim=1)

  def _body_frame_cmd(motion, times, quat):
    """把平滑后的世界系线/角速度投影到给定根方向。

    motion 提供参考速度，times:[B] 选时刻，quat:[B,4] 是用于坐标变换的根四元数。
    先对世界速度做中心平均，再分别逆旋转到机体系，取线速度x/y和角速度z，
    返回 [B,3]。训练使用当前预测根方向，回放命令表则使用参考根方向。
    """
    v_b = quat_apply_inverse(quat, _centered_mean(motion.root_lin_vel_w, times))
    w_b = quat_apply_inverse(quat, _centered_mean(motion.root_ang_vel_w, times))
    return torch.stack([v_b[:, 0], v_b[:, 1], w_b[:, 2]], dim=-1)

  def extract(motion, motion_times, aux_state):
    """生成一批训练速度目标，并加入独立的逐步命令扰动。

    输入 motion、motion_times:[B]、aux_state:[B,7]，从后者取当前根方向投影参考速度。
    均匀噪声范围为每轴 [-半宽,半宽)，叠加后裁到配置范围，返回 [B,3]。
    随机数每次调用重新生成；本函数不推进参考帧，不修改参考动作数组。
    """
    # 使用当前机器人/预测根方向投影速度，而非固定使用参考方向；命令随当前朝向改变。
    cmd = _body_frame_cmd(motion, motion_times, aux_state[:, 3:7])
    # rand_like 生成 [0,1) 随机数，线性映射到 [-1,1) 后乘逐轴半宽，得到均匀扰动。
    noise = (torch.rand_like(cmd) * 2 - 1) * noise_half.to(cmd.device)
    return _clamp(cmd + noise)

  def precompute_replay_goal(motion):
    """为整条参考时间轴生成可重复的无噪速度命令表。

    motion 提供全部帧的速度及根方向。以参考根方向投影中心平均速度，保留训练时
    同样的范围裁剪，但不添加命令噪声，返回 [总帧数,3]。
    供 viewer 回放及导出检查使用；这是假设机器人沿参考状态的命令，而非任意预测状态目标。
    """
    device = motion.root_lin_vel_w.device
    # 为全部参考帧建立索引，离线一次算完；结果按帧查表用于回放与导出核对。
    times = torch.arange(motion.root_lin_vel_w.shape[0], device=device)
    # 回放命令表假设机器人沿参考运动，因此投影使用参考根四元数，并且不加训练噪声。
    cmd = _body_frame_cmd(motion, times, motion.root_quat_w)
    # 保持与训练相同的命令范围，但不加入训练扰动，保证回放目标可重复。
    return _clamp(cmd)

  return TaskSpec(
    name="velocity",
    goal_dim=3,
    extract_training_goal=extract,
    goal_specs=(
      ContinuousGoalSpec("Lin Vel X (m/s)", *lin_vel_x_range, step=0.1, default=0.0),
      ContinuousGoalSpec("Lin Vel Y (m/s)", *lin_vel_y_range, step=0.1, default=0.0),
      ContinuousGoalSpec("Ang Vel Z (rad/s)", *ang_vel_z_range, step=0.1, default=0.0),
    ),
    build_inference_goal_obs=lambda raw: raw,
    raw_goal_dim=3,
    precompute_replay_goal=precompute_replay_goal,
  )


@dataclass(frozen=True)
class VelocityTaskCfg:
  """保存速度任务命令范围、参考平滑窗口和训练噪声尺度。

  线速度范围单位 m/s，角速度范围单位 rad/s；window_frames 是参考帧数，
  不是传感器历史长度。command_noise 三项依次为 vx、vy、wz 的均匀噪声半宽。
  配置不创建张量，make_velocity_task 才把这些值转为闭包常量。
  """

  lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
  lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
  ang_vel_z_range: tuple[float, float] = (-3.14, 3.14)
  # 参考速度的中心平滑窗口；25 是参考帧数，不是策略观察历史长度。
  window_frames: int = 25
  """参考速度的中心平均窗口；0或1表示直接使用瞬时速度。"""
  command_noise: tuple[float, float, float] = (0.5, 0.5, 0.785)
  """每步命令的逐轴均匀噪声半宽，顺序为vx、vy、wz；叠加后再裁到范围。"""


@dataclass(frozen=True)
class TasksCfg:
  """集中保存各任务的可调参数，供 resolve_task_specs 构建具体接口。

  当前仅 velocity 有独立超参数，motion 的目标定义由状态表示决定。
  field(default_factory=VelocityTaskCfg) 保证每个配置对象独立创建子配置。
  本类字段顺序不直接分配任务槽位；当前任务返回顺序由 resolve_task_specs 的列表决定。
  """

  velocity: VelocityTaskCfg = field(default_factory=VelocityTaskCfg)


# 函数作用：将配置解析为 actor 与想象环境共享的有序任务定义。
def resolve_task_specs(
  tasks: TasksCfg,
  representation: "StateRepresentation",
  num_goals: int,
  device: str = "cpu",
) -> list[TaskSpec]:
  """将配置解析为 actor 与想象环境共享的有序任务定义。

  tasks 提供速度范围等参数，representation/num_goals 用于 motion 目标，device
  用于速度任务闭包常量。返回 [motion_spec,velocity_spec]，首任务成为潜变量对齐锚点。
  这里包含全部任务；哪些任务在本批次占用样本由 slot_mix 决定，默认 velocity 可为零。
  不创建网络、不计算目标样本，也不修改物理环境配置。
  """
  # 【源码讲解】
  # 返回顺序决定编码器槽位顺序及锚点：motion 在前，velocity 在后。
  # 任务存在不等于占用训练样本，实际占比还由 runner.slot_mix 决定。
  return [
    make_motion_task(representation, num_goals),
    make_velocity_task(
      tasks.velocity.lin_vel_x_range,
      tasks.velocity.lin_vel_y_range,
      tasks.velocity.ang_vel_z_range,
      tasks.velocity.window_frames,
      tasks.velocity.command_noise,
      device,
    ),
  ]
