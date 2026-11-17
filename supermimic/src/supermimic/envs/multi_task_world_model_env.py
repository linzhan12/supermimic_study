"""为同一世界模型轨迹组织多种目标模态的可微想象环境。

文件位置：supermimic/src/supermimic/envs/multi_task_world_model_env.py。
继承 WorldModelEnv 的重置、预测、奖励、软终止和传感器历史，仅补充每任务目标
及策略输入打包。所有槽位使用相同的动作跟踪奖励，任务差别在于编码器读取的目标。
runner 传入活跃任务定义，父类刷新目标时调用 _build_task_goals，策略读取
get_observations 返回的整批字段后，自行按 slot_mix 分配任务区间。
"""

# 【文件作用】多任务可微想象环境。
# 复用 WorldModelEnv 的动力学、奖励、软终止和传感器历史，增加各任务目标的组织。
# 所有任务仍使用动作跟踪奖励；区别是策略的不同编码器读不同的目标模态。
# 【阅读顺序】初始化任务列表 → 参考目标刷新时生成任务目标 → 打包策略与价值观察。

from __future__ import annotations

# TYPE_CHECKING 块内导入仅供静态类型检查；Any 表示物理环境依赖其接口而非固定具体包装类型。
from typing import TYPE_CHECKING, Any

# torch.Tensor 用于标注各任务目标张量，实际推进状态的运算主要在父类。
import torch
# TensorDict 将历史、不同宽度的任务目标与 critic 输入组织为共同批次 [B]。
from tensordict import TensorDict

# 来自 supermimic/src/supermimic/envs/world_model_env.py：复用可微步进、回放重置和参考时序。
from supermimic.envs.world_model_env import WorldModelEnv
# 来自 supermimic/src/supermimic/tasks/task_spec.py：统一 history/goal_任务名/critic 字段的打包约定。
from supermimic.tasks.task_spec import build_actor_obs_td

# 以下类型来自模型、经验池、状态表示和任务协议文件，运行时已有实例由 runner 注入。
# 结合 future annotations 推迟求值，避免仅为类型标注引入额外模块依赖。
if TYPE_CHECKING:
  # 来自 supermimic/src/supermimic/models.py：动力学网络类型，实际实例与世界模型学习器共享。
  from supermimic.models import Dynamics
  # 来自 supermimic/src/supermimic/replay_buffer.py：真实轨迹起点与传感器历史来源。
  from supermimic.replay_buffer import ReplayBuffer
  # 来自 supermimic/src/supermimic/representations.py：状态转换、观察合成和奖励恢复接口。
  from supermimic.representations import StateRepresentation
  # 来自 supermimic/src/supermimic/tasks/task_spec.py：每个任务的名称、目标宽度和训练目标提取函数。
  from supermimic.tasks.task_spec import TaskSpec


class MultiTaskWorldModelEnv(WorldModelEnv):
  """共享动力学与奖励、向多个目标编码器提供观察的想象环境。

  由 SuperMimicRunner._init_policy 创建，持有与模型学习器相同的 Dynamics 和表示引用，
  并从真实 ReplayBuffer 抽想象起点。父类负责动作推进及状态变化，本类只维护
  _task_goal_obs，并实现符合 MultiEncoderAE 接口的 get_observations。

  任务列表必须非空且首项为 motion，作为锚点直接复用父类已构建的 actor 目标；
  其他任务通过各自 TaskSpec.extract_training_goal 读取同一预测根状态和参考时刻。
  目标都保留完整 B 维，不在环境内切槽位；历史环也由父类管理，不为每任务独立维护。
  """

  # 函数作用：保存任务列表并初始化父类的共享模型、参考和终止配置。
  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    task_specs: list[TaskSpec],
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
  ) -> None:
    """保存任务列表并初始化父类的共享模型、参考和终止配置。

    参数 dynamics、representation 为已建立的模型对象；env 是物理环境，用来读取
    参考动作和任务阈值；batch_size=B 是想象并行数，不要求等于物理环境数。
    replay_buffer 提供真实起点，task_specs 是非空有序活跃任务定义；history_length=H
    决定策略历史长度，termination_softness 控制软终止平滑度，device 指定张量设备。

    先保存任务并断言首项名称为 motion，再把其余依赖传入父类。返回 None，构造本身
    不从经验池采样、不执行 reset 或 step；首次有效观察需在 reset 刷新目标之后获取。
    模型、表示和经验池共享引用，不在这里复制网络或创建优化器。
    """
    # 先保存任务列表，供后续父类目标刷新时调用的钩子读取。
    self.task_specs = task_specs
    # 第一个任务必须为 motion，后续复用其完整参考目标作为锚点。
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
    # 父类负责模型引用、起点采样、阈值和参考数据；batch_size 是想象并行数。
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      env=env,
      batch_size=batch_size,
      replay_buffer=replay_buffer,
      history_length=history_length,
      termination_softness=termination_softness,
      device=device,
    )

  # 函数作用：在父类刷新参考目标后构建各活跃任务的整批目标缓存。
  def _build_task_goals(self) -> None:
    """在父类刷新参考目标后构建各活跃任务的整批目标缓存。

    无显式参数，读取 task_specs、motion_times:[B]、aux_state:[B,7] 和参考动作。
    首任务 motion 直接引用父类已算好的 _actor_goal_obs，其他任务分别调用提取函数，
    将结果 clone 后保存为 _task_goal_obs[任务名]:[B,G_task]。

    返回 None，更新内部目标字典；不推进时间、不预测下一状态、不改变任务槽位。
    clone 提供独立存储但不等于 detach，任务目标若依赖预测状态仍可保留梯度。
    父类 _update_goal 在 reset 和每次推进参考后调用该钩子，之后 get_observations
    读取缓存，避免反复取观察时再次随机生成速度命令扰动。
    """
    # motion 目标已由父类 _update_goal 构造，复用缓存可避免重复计算未来窗口。
    anchor, *rest = self.task_specs
    self._task_goal_obs: dict[str, torch.Tensor] = {anchor.name: self._actor_goal_obs}
    # 每种目标均保留完整 batch，actor 再按 slot_mix 对应区间选择输入。
    # clone 将目标值保存为独立张量，避免后续状态推进改变其底层数据。
    for ts in rest:
      self._task_goal_obs[ts.name] = ts.extract_training_goal(
        self.motion, self.motion_times, self.aux_state
      ).clone()

  # 函数作用：打包多任务 actor 和 critic 的当前输入，返回批次形状为 [B] 的 TensorDict。
  def get_observations(self) -> TensorDict:
    """打包多任务 actor 和 critic 的当前输入，返回批次形状为 [B] 的 TensorDict。

    无显式参数，要求先 reset。history:[B,H,D] 来自父类传感器历史，最新帧在末尾；
    每个 goal_任务名:[B,G_task] 来自缓存；critic:[B,R+G+C] 使用完整模型表示、
    世界模型目标和仿真上下文，而不是部署传感器输入。

    调用 build_actor_obs_td 统一键名和批次元数据。函数不切分任务区间，不重新提取目标，
    不推进动力学或刷新统计；actor 随后依据自己的槽位范围选取各 goal 字段。
    """
    # history 为 [B,H,D]；goal_任务名 为 [B,G]；critic 使用完整状态和上下文。
    return build_actor_obs_td(
      self._task_goal_obs, self._sensor_history(), critic=self._critic_obs()
    )
