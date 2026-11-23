"""用学习到的 Dynamics 替代物理引擎的可微动作跟踪环境基类。

reset 从真实回放恢复状态，step 预测下一状态并依据参考动作计算奖励与软终止。
子类只组织策略观察，必要时覆盖额外任务目标钩子；共享历史、参考时刻和评分顺序在本文件。
物理仿真 env 仅用于读取任务数据与阈值，本环境的 step 不调用 MuJoCo。

调用链：reset→_on_reset→get_observations；step→_wm_step→_compute_commands→get_observations。
B 是想象并行数，可与物理环境数不同；状态成员仅在 reset 后才具备可用值。
"""

# 【文件作用】可微想象环境基类
# 以学习到的动力学代替物理引擎推进状态，维护参考动作、传感器历史、奖励和软终止。
# 【阅读顺序】从回放重置 → 动力学预测与评分 → 推进参考目标 → 合成观察；子类组织具体任务输入。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟类型注解求值，使 TYPE_CHECKING 内的类型仅用于静态检查而无需运行时导入。
from __future__ import annotations



# 抽象基类机制：要求具体想象环境实现 get_observations，同时复用本文件的状态推进模板。
from abc import ABC, abstractmethod
# TYPE_CHECKING 隔离类型依赖，Any 允许接收提供所需属性的环境包装对象而不绑定特定包装类。
from typing import TYPE_CHECKING, Any

# PyTorch 提供张量索引、批次变换、自动微分和编译；本文件的数据形状约定由这些运算实现。
import torch
# 共享 batch_size 的张量字典，统一承载状态、历史、任务目标和训练批次，不只是任意键值集合。
from tensordict import TensorDict

# 来自 supermimic/src/supermimic/replay_buffer.py：真实仿真经验池，仅用于抽想象起点和同回合传感器历史。
from supermimic.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation


class WorldModelEnv(ABC):
  """维护可微预测状态、参考动作与观察历史的模板式环境。

  依赖共享 Dynamics、StateRepresentation、真实回放和物理任务配置。基类统一实现
  重置、预测、奖励、软终止和时间推进；get_observations 为抽象接口，由单任务或多任务
  子类决定 actor 字段如何组装。_build_task_goals 是可选钩子，默认不额外生成目标。

  软终止返回连续权重，不在 step 自动重置；策略 learner 用它衰减未来收益。
  状态和动作保留跨步计算图，因此不能在 step 中随意 detach。
  """

  # 函数作用：绑定共享预测组件，并从真实环境读取参考数据与终止阈值。
  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
  ) -> None:
    """绑定共享预测组件，并从真实环境读取参考数据与终止阈值。

    dynamics/representation 用于预测与状态语义；env 提供控制步长、奖励配置和 motion 数据；
    batch_size 为想象并行数B；replay_buffer 提供起点；history_length 为策略历史帧数，
    termination_softness 控制 sigmoid 平滑尺度，device 指定张量设备。

    返回 None，计算正奖励上界、保存锚点索引和高度/姿态阈值；平滑尺度非正立即报错。
    这里尚未创建 rep_state 等运行状态，必须 reset 后才能 step。
    """
    # 【源码讲解】
    # 这里的 num_envs 等于想象 batch_size，可与物理仿真并行数不同。
    # 共享已训练动力学、表示和经验池，并读取真实任务的参考动作与终止阈值。
    # max_step_reward 由正奖励权重与 dt 推导，为 critic 价值上界等路径使用。
    self.device = device
    self.dynamics = dynamics
    self.representation = representation
    self.num_envs = batch_size
    self.replay_buffer = replay_buffer
    self.history_length = history_length

    # 每步正奖励上界由所有正权重之和乘控制步长得到，供价值上界与重置采样归一化使用。
    self.max_step_reward = env.unwrapped.step_dt * sum(
      term.weight for term in env.cfg.rewards.values() if term.weight > 0.0
    )

    # 保留参考动作数据与根锚点索引，不复制整条动作序列。
    self._num_actions = env.num_actions
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    self.motion = motion_cmd.motion
    self._anchor_index = motion_cmd.motion_anchor_body_index

    if termination_softness <= 0.0:
      raise ValueError(
        f"termination_softness must be positive, got {termination_softness}"
      )
    self._termination_softness = termination_softness
    self._configure_scoring(env)

  def _configure_scoring(self, env: Any) -> None:
    terms = env.cfg.terminations
    self._anchor_pos_threshold = terms["anchor_pos"].params["threshold"]
    self._anchor_ori_threshold = terms["anchor_ori"].params["threshold"]

  # ------------------------------------------------------------------
  # 由子类实现的观察与目标接口
  # ------------------------------------------------------------------

  # 函数作用：约定子类将当前状态组装为 actor 与 critic 观察。
  @abstractmethod
  def get_observations(self) -> TensorDict:
    """约定子类将当前状态组装为 actor 与 critic 观察。

    无参数，返回批次为 [B] 的 TensorDict，必须包含 critic 特权观察及策略实际需要字段。
    单任务可提供 actor 扁平向量，多任务可提供 history 与多个 goal_任务名。
    基类只定义接口，不在此处选择策略网络结构；未实现此抽象方法不能实例化。
    """
    ...

  # 函数作用：在完整参考目标更新后，允许子类生成额外任务目标。
  def _build_task_goals(self) -> None:  # noqa: B027
    """在完整参考目标更新后，允许子类生成额外任务目标。

    无参数、返回 None；此时 motion_times、aux_state、_actor_goal_obs 已更新。
    基类默认不执行操作，多任务子类可复用 motion 目标并提取速度命令。
    钩子由 _update_goal 调用，覆盖它时无需另行推进参考帧。
    """

  # ------------------------------------------------------------------
  # 共享的重置和推进模板
  # ------------------------------------------------------------------

  # 函数作用：暴露高层策略动作宽度。
  @property
  def num_actions(self) -> int:
    """暴露高层策略动作宽度。

    此 property 返回构造时从真实环境读取的 _num_actions，调用方式为 env.num_actions。
    它不是关节目标窗口宽度，也不是模型状态宽度；返回值用于 actor 输出尺寸。
    """
    return self._num_actions

  # 函数作用：把一批真实回放状态作为新想象轨迹起点并返回当前观察。
  def reset(self, batch: TensorDict | None = None) -> TensorDict:
    """把一批真实回放状态作为新想象轨迹起点并返回当前观察。

    batch 可为 [B] 的 TensorDict；省略时调用 sample_initial_states。恢复 rep/aux、
    关节状态、上一动作和上下文后，_on_reset 设置参考时刻、目标和传感器历史。
    返回子类 get_observations 的策略与价值输入，布局与对应 actor 一致。

    初始状态和上一动作 detach，起点不继承旧计算图；后续 step 不再切断状态梯度。
    重置不推进真实物理环境，也不清空共享经验池。
    """
    # 【源码讲解】
    # batch 来自真实回放，恢复状态、上一动作、上下文和参考时刻。
    # 由表示得到关节状态，再刷新任务目标及传感器历史；返回当前观察。
    # 初始状态 detach，使策略反向传播从本次想象轨迹开始。
    if batch is None:
      batch = self.sample_initial_states()
    self.aux_state, self.rep_state = self._unpack_initial_states(batch)
    self.joint_state = self.representation.joint_state(self.rep_state)
    # 保存未经尺度变换的策略输出；动力学内部处理缩放，奖励中的动作变化项直接使用原始值。
    # 起点的前一动作来自真实采集，用于第一想象步动作变化惩罚；之后由 step 持续更新。
    self.last_actions = batch["prev_action"].detach()
    # 整段想象沿用起点的物理上下文，不在每步重新随机化动力学条件。
    self.sim_ctx = batch["sim_ctx"]
    self._on_reset(batch)
    return self.get_observations()

  # 函数作用：执行一个想象控制步，并按固定时间顺序刷新状态与观察。
  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """执行一个想象控制步，并按固定时间顺序刷新状态与观察。

    actions:[B,A] 是原始策略输出。_wm_step 先用当前缓存参考为预测后状态评分，
    得到下一 aux/rep/关节状态、软终止和奖励；保存状态后再推进参考时刻、历史与传感器。
    返回 (next_obs,reward,term_prob,{})，reward/term_prob 通常为[B]。

    不会根据软终止自动重置，也不会把状态 detach；保持图链使多步回报能反传到 actor。
    调用前必须 reset，且动作 batch 应匹配本环境 num_envs。
    """
    # 【源码讲解】
    # 先使用当前缓存参考计算预测、奖励和软终止，再保存新状态。
    # 随后推进参考时刻并刷新目标，合成新传感器；返回下一观察及本步奖励。
    # 软终止作为回报权重交给 learner，本函数不按该值自动重置。
    next_aux_state, next_rep_state, next_joint_state, term_prob, rewards = (
      self._wm_step(
        self.rep_state,
        self.aux_state,
        actions,
        self.last_actions,
        self.sim_ctx,
      )
    )
    self.rep_state = next_rep_state
    self.aux_state = next_aux_state
    self.joint_state = next_joint_state
    # 保留当前动作的计算图，使下一步动作变化惩罚也能对本步 actor 输出求导。
    self.last_actions = actions

    # 本步已用旧缓存目标评分；现在推进参考目标，作为下一次 actor 输入和下一步评分依据。
    self._compute_commands()

    if self.history_length > 1:
      # 先把此前 latest 移入历史，丢弃最旧帧，随后才根据新状态合成新的 latest。
      self._past_sensors = torch.cat(
        [self._past_sensors[:, 1:], self._latest_sensor.unsqueeze(1)], dim=1
      )
    self._latest_sensor = self._synth_sensor()

    return self.get_observations(), rewards, term_prob, {}

  # 函数作用：从当前预测状态生成策略的单帧状态/传感器输入。
  def _synth_sensor(self) -> torch.Tensor:
    """从当前预测状态生成策略的单帧状态/传感器输入。

    无参数，向 representation 传当前 rep、aux、关节状态、上下文和上一动作。
    返回 [B,D]；rep_state 模式直接返回表示，proprio 模式合成传感器并按统一规范加噪。
    此处不拼未来目标，也不更新历史窗口，只生成一帧供调用者缓存。
    """
    # 想象环境没有物理传感器，利用预测状态、根位姿、关节状态和上一动作合成观察。
    # 统一由表示决定状态模式或传感器模式，并在需要时施加传感器噪声。
    return self.representation.compute_actor_sensor(
      self.rep_state,
      self.sim_ctx,
      aux_state=self.aux_state,
      joint_state=self.joint_state,
      last_action=self.last_actions,
      noisy=True,
    )

  # 函数作用：返回最新帧在最后的完整策略历史窗口。
  def _sensor_history(self) -> torch.Tensor:
    """返回最新帧在最后的完整策略历史窗口。

    无参数，H=1 时将 _latest_sensor:[B,D] 扩为 [B,1,D]；H>1 时与已缓存 H-1
    帧拼成 [B,H,D]。函数只读取和拼接，不推进时间、不再次加噪。
    历史缺失的填充在 reset 的预热路径处理，不在这里临时补造。
    """
    latest = self._latest_sensor.unsqueeze(1)
    if self.history_length == 1:
      return latest
    return torch.cat([self._past_sensors, latest], dim=1)

  # 函数作用：构造价值网络使用的特权观察。
  def _critic_obs(self) -> torch.Tensor:
    """构造价值网络使用的特权观察。

    无参数，沿最后维拼接当前模型表示R、完整世界模型目标G和上下文C，返回[B,R+G+C]。
    它有比部署本体传感器更多的信息，不能直接当成 actor 的实际传感器输入。
    该拼接仍保留状态梯度，以便末端价值对 actor 的动作路径求导。
    """
    # 【源码讲解】
    # critic 使用完整表示、世界模型目标及上下文，沿最后一维拼接。
    # 这是特权观察，不能直接当成部署 actor 可以获得的传感器输入。
    return torch.cat([self.rep_state, self._wm_goal_obs, self.sim_ctx], dim=-1)

  # 函数作用：把动力学、终止判据和奖励封装为一个可微评分步骤。
  def _wm_step(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把动力学、终止判据和奖励封装为一个可微评分步骤。

    输入 rep_state:[B,R]、aux_state:[B,7]、action/last_action:[B,A]、sim_ctx:[B,C]。
    先经 Dynamics 得下一状态，再恢复关节状态，以缓存的当前参考刚体状态计算软终止及奖励。
    返回顺序为 (next_aux,next_rep,next_joint,term_prob,reward)，后两者为[B]。

    action 是原始策略输出，尺度变换由 Dynamics 内部处理。此方法自身不写回环境状态，
    也不推进参考时刻；step 在收到结果后才更新成员并刷新目标，保证评分时间对齐。
    """
    # 想象环境用学习到的 Dynamics 代替物理引擎预测下一状态。
    next_aux_state, next_rep_state = self.dynamics(
      rep_state, action, aux_state, sim_ctx
    )
    # 奖励的关节限位项和下一次传感器合成都依赖关节状态；最大坐标需要逆运动学恢复。
    next_joint_state = self.representation.joint_state(next_rep_state)
    term_prob, reward = self._score_transition(
      next_aux_state, next_rep_state, next_joint_state, action, last_action
    )
    return next_aux_state, next_rep_state, next_joint_state, term_prob, reward

  def _score_transition(
    self, next_aux_state, next_rep_state, next_joint_state, action, last_action
  ) -> tuple[torch.Tensor, torch.Tensor]:
    ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w = (
      self._ref_body_state
    )
    # 先对预测后的根状态判断软终止，再对同一预测状态和同一缓存参考计算奖励。
    term_prob = self._compute_term_prob(next_aux_state, ref_body_pos_w, ref_body_quat_w)
    reward = self.representation.compute_wm_reward_from_reference(
      next_rep_state,
      next_aux_state,
      next_joint_state,
      action,
      last_action,
      ref_body_pos_w,
      ref_body_quat_w,
      ref_body_lin_vel_w,
      ref_body_ang_vel_w,
    )
    return term_prob, reward

  # 函数作用：将物理任务的高度和倾斜终止阈值平滑为可微结束权重。
  def _compute_term_prob(
    self,
    aux_state: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
  ) -> torch.Tensor:
    """将物理任务的高度和倾斜终止阈值平滑为可微结束权重。

    aux_state:[B,7] 为预测根位姿，参考刚体位置/四元数按锚点索引取参考根。
    分别计算高度差和机体系重力z分量差，除以对应阈值并取较大者，随后应用 sigmoid。
    返回[B]连续值；归一化误差为1时输出0.5，softness 越小越接近硬阈值。

    它不是学习或标定得到的跌倒概率，不包含物理环境所有可能的结束原因；
    主要用于想象回报的软存活权重，不负责自动重置。
    """
    anchor_pos_w = ref_body_pos_w[:, self._anchor_index]
    anchor_quat_w = ref_body_quat_w[:, self._anchor_index]
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    height_error = (anchor_pos_w[..., 2] - root_pos[..., 2]).abs()
    # 对单位四元数(w,x,y,z)，世界向下方向在机体系的z分量为2(x²+y²)-1。
    ref_gravity_z = 2 * (anchor_quat_w[..., 1:3] ** 2).sum(-1) - 1.0
    root_gravity_z = 2 * (root_quat[..., 1:3] ** 2).sum(-1) - 1.0
    tilt_error = (ref_gravity_z - root_gravity_z).abs()
    # 把高度误差和倾斜误差分别除以各自阈值；最大值达到 1 对应阈值附近。
    normalized_error = torch.maximum(
      height_error / self._anchor_pos_threshold,
      tilt_error / self._anchor_ori_threshold,
    )
    # sigmoid 将硬阈值变为可微的软终止近似；它不是实测跌倒概率。
    # 归一化误差等于 1 时输出 0.5；softness 越小变化越陡，用作可微结束权重。
    return torch.sigmoid((normalized_error - 1.0) / self._termination_softness)

  # 函数作用：恢复起点参考时刻并初始化目标及传感器缓存。
  def _on_reset(self, batch: TensorDict) -> None:
    """恢复起点参考时刻并初始化目标及传感器缓存。

    batch 是起点快照，motion_time:[B] 指向该状态动作前的参考帧。先更新目标，
    长历史模式再用 sensor_window 与 pad_mask 预热 H-1 帧，最后由当前预测状态合成最新帧。
    返回 None，写入 motion_times、目标缓存、_past_sensors 与 _latest_sensor。
    调用者 reset 已先恢复状态、上一动作和上下文，因此顺序不能调换。
    """
    self.motion_times = batch["motion_time"]
    self._update_goal()
    self._reset_sensors(batch)

  def _reset_sensors(self, batch: TensorDict) -> None:
    if self.history_length > 1:
      self._past_sensors = self._warmstart_past_sensors(
        batch["sensor_window"], batch["pad_mask"]
      )
    self._latest_sensor = self._synth_sensor()

  # 函数作用：把每条想象轨迹的参考时刻推进一帧，但不越过动作片段末端。
  def _compute_commands(self) -> None:
    """把每条想象轨迹的参考时刻推进一帧，但不越过动作片段末端。

    无参数，使用有序 motion_ends 检查 next_motion_times 是否恰好落到右开片段终点；
    若到达边界则保持当前最后有效帧，否则前进。随后 _update_goal 刷新评分参考和策略目标。
    返回 None，不重置状态也不切换到拼接的下一片段；软终止另由评分路径计算。
    """
    # 【源码讲解】
    # 参考帧号尝试加 1，遇到当前片段末端时保持原帧，不跨到拼接的下一片段。
    # searchsorted 在排好序的结束位置中找候选边界，再更新目标缓存。
    next_motion_times = self.motion_times + 1
    # 在有序片段终点中做成员查询，避免使用可能引入主机同步的通用集合判断。
    ends = self.motion.motion_ends
    pos = torch.searchsorted(ends, next_motion_times).clamp(max=ends.numel() - 1)
    at_end = ends[pos] == next_motion_times
    self.motion_times = torch.where(at_end, self.motion_times, next_motion_times)
    self._update_goal()

  # 函数作用：为当前参考时刻缓存身体真值及策略/模型目标。
  def _update_goal(self) -> None:
    """为当前参考时刻缓存身体真值及策略/模型目标。

    无参数，motion.body_state 提供当前帧参考刚体位姿速度；compute_wm_goal 使用当前
    aux_state 计算相对窗口，分别返回完整世界模型目标与策略目标。随后调用任务目标钩子。
    返回 None，供下一次 _wm_step 评分及当前 get_observations 组装输入。
    目标随预测根改变，不能仅推进时刻而不重算相对变换。
    """
    # 【源码讲解】
    # 按 motion_times 缓存参考刚体状态，供下一次 _wm_step 评分。
    # 同时用当前根位姿构造 wm_goal、actor_goal，并调用子类任务目标钩子。
    self._ref_body_state = self.motion.body_state(self.motion_times)
    self._wm_goal_obs, self._actor_goal_obs = self.representation.compute_wm_goal(
      self.motion, self.motion_times, self.aux_state
    )
    self._build_task_goals()

  # ------------------------------------------------------------------
  # 起点采样与历史预热辅助方法
  # ------------------------------------------------------------------

  # 函数作用：从真实经验池抽取一批想象起点及所需历史。
  def sample_initial_states(self) -> TensorDict:
    """从真实经验池抽取一批想象起点及所需历史。

    无参数，B=self.num_envs。H=1 时抽1帧窗口再去掉时间轴；H>1 时 sample_causal
    先均匀抽当前帧并向前找同回合历史，不足部分重复最早有效帧。
    返回[B]批次 TensorDict，长历史另附 sensor_window:[B,H,D] 与 pad_mask:[B,H]。
    改变H不应改变当前起点分布；本方法不直接建立环境状态，reset 才消费返回结果。
    """
    if self.history_length == 1:
      return self.replay_buffer.sample(1, self.num_envs).squeeze(1)
    # 以当前帧为末尾抽取同回合历史；不足的前驱由最早有效帧填充。
    window, padded = self.replay_buffer.sample_causal(
      self.history_length, self.num_envs
    )
    # 只把末帧作为起点状态；整段真实历史单独附加到 sensor_window，用于初始化历史环。
    batch = window[:, -1]
    batch["sensor_window"] = window["actor_obs"]
    batch["pad_mask"] = padded
    return batch

  # 函数作用：把真实历史窗口转换为想象环境的带噪前驱缓存。
  def _warmstart_past_sensors(
    self, sensor_window: torch.Tensor, pad_mask: torch.Tensor
  ) -> torch.Tensor:
    """把真实历史窗口转换为想象环境的带噪前驱缓存。

    sensor_window:[B,H,D]，pad_mask:[B,H] 标记回合开始前的重复填充槽。先去掉最后
    一帧，为剩余H-1帧加噪；填充槽共享最早可用历史样本的一次噪声实现，返回[B,H-1,D]。
    若前驱全部是填充，argmax 返回0，重复的首槽仍提供共享数值；当前最新帧另由 _synth_sensor 合成。
    这样保持与物理重置时重复同一带噪帧的初始化规则对应。
    """
    noisy = self.representation.add_proprio_noise(sensor_window[:, :-1])
    pad = pad_mask[:, :-1]  # padding is a leading run (sample_causal)
    # 重复的填充帧共用同一次噪声，与物理环境重置时的历史初始化一致。
    first_real = (~pad).int().argmax(dim=1)
    # 为每个环境取最早可用历史样本的噪声实现，广播填充此前的所有槽位。
    # 这样重复历史的数值完全一致，不会凭空制造起点之前的运动。
    shared = noisy[torch.arange(self.num_envs, device=self.device), first_real]
    return torch.where(pad.unsqueeze(-1), shared.unsqueeze(1), noisy)

  # 函数作用：将起点完整状态拆成辅助根位姿和模型表示。
  def _unpack_initial_states(
    self, batch: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """将起点完整状态拆成辅助根位姿和模型表示。

    batch['state']:[B,S]，通过 representation 的两个接口提取 aux:[B,7] 和 rep:[B,R]。
    返回顺序为 (aux,rep)，两者都 detach，防止新想象轨迹连接到起点来源的旧图。
    不处理目标、上一动作和上下文，它们由 reset 的其他步骤恢复。
    """
    full_states = batch["state"]
    aux_state = self.representation.extract_aux_state(full_states)
    rep_state = self.representation.represent(full_states)
    return aux_state.detach(), rep_state.detach()
