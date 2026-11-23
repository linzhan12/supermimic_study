
# 【文件作用】动力学网络与多任务策略网络
# 构建可微状态预测网络和多目标编码器共享解码器策略，并处理归一化、动作尺度与潜变量对齐。
# 【阅读顺序】归一化兼容层 → MLP 构造 → Dynamics → MultiEncoderAE。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。

# 推迟类型注解求值，使本文件可用稍后定义的类型并减少循环导入；不改变张量计算。
from __future__ import annotations


# TYPE_CHECKING 内的导入只服务类型检查，避免运行时循环依赖；Literal 限定观察模式取值。
from typing import TYPE_CHECKING

# PyTorch 张量、随机采样、设备和自动微分工具；计算沿最后一维组织特征，前导轴保留批次/时间。
import torch
# Module、ModuleDict、Sequential 负责注册网络与缓冲；注册后设备迁移和 state_dict 才能统一管理。
import torch.nn as nn
# EmpiricalNormalization 维护特征经验统计；本文件子类仅修正标准差更新的地址稳定性。
from rsl_rl.modules import EmpiricalNormalization
# TensorDict 携带共享批次形状和多个张量字段；回放与策略输入可统一索引、切片和堆叠。
from tensordict import TensorDict

if TYPE_CHECKING:
  # 来自 supermimic/src/supermimic/representations.py，描述网络状态与预测积分契约；只在类型检查时导入，实际对象由 runner 注入。
  from supermimic.representations import StateRepresentation
  # 来自 supermimic/src/supermimic/tasks/task_spec.py，TaskSpec 提供任务名称、目标宽度与提取接口；策略据此建立独立编码器。
  from supermimic.tasks.task_spec import TaskSpec


class _CudaGraphNormalization(EmpiricalNormalization):
  """与 CUDA graph 地址约束兼容的经验归一化器。

  继承 rsl_rl 的 EmpiricalNormalization，沿特征维维护均值和方差。
  上游 update 会重建标准差张量，本类将新值复制回旧缓冲区，使已捕获图继续
  访问同一地址。它调整的是更新机制，不改变归一化公式或新增可训练参数。
  Dynamics 和 MultiEncoderAE 都使用这个版本来支持捕获后的训练更新。
  """

  # 函数作用：积累新样本统计，同时保持标准差缓冲区地址不变。
  @torch.jit.unused
  def update(self, x: torch.Tensor) -> None:
    """积累新样本统计，同时保持标准差缓冲区地址不变。

    参数 x 为最后一维匹配特征宽度的输入样本；返回 None，改变归一化统计缓冲。
    先保存 _std 引用，调用父类计算统计，再 copy_ 回旧张量并恢复引用。
    是否实际积累样本仍遵循父类的训练模式约定；此函数不执行模型梯度优化。
    """
    # 【源码讲解】
    # 上游 update 可能替换标准差张量；先保存旧缓冲区，再将新值复制回旧地址。
    # CUDA graph 重放依赖捕获时的地址，不能仅保证数值一样。
    std_buf = self._std
    super().update(x)
    std_buf.copy_(self._std)
    self._std = std_buf


# 函数作用：构造接收普通张量的多层感知机，返回 nn.Sequential。
def build_mlp(
  input_dim: int,
  hidden_dims: tuple[int, ...],
  output_dim: int,
) -> nn.Sequential:
  """构造接收普通张量的多层感知机，返回 nn.Sequential。

  参数 input_dim/output_dim 为末轴输入输出宽度，hidden_dims 为有序隐藏层宽度。
  每个隐藏层依次使用 Linear、LayerNorm、GELU；末层只有 Linear，不额外限幅。
  任意前导批次轴被保留，例如 [B,D_in] 变为 [B,D_out]。

  此工厂不负责经验归一化；Dynamics、策略编码器与解码器在外层安排自己的统计。
  它也不同于 runner 公共工厂中接收命名观察的 rsl_rl.MLPModel。
  """
  # 【源码讲解】
  # 每个隐藏层由线性映射、LayerNorm 和 GELU 组成，最后一层仅线性映射。
  # 输入和输出的最后一维分别为 input_dim、output_dim，前面的批次维保留。
  layers: list[nn.Module] = []
  curr = input_dim
  for h in hidden_dims:
    layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.GELU()]
    curr = h
  layers.append(nn.Linear(curr, output_dim))
  return nn.Sequential(*layers)


class Dynamics(nn.Module):
  """可微的单控制步动力学模型，由真实仿真连续轨迹监督学习。

  输入包括模型表示、原始动作、辅助根位姿和仿真上下文。MLP 只读取归一化表示、
  处理后动作及上下文，输出速度级增量；表示对象负责将增量积分成下一状态。
  辅助根位姿参与积分，不作为单独输入字段直接拼入此 MLP。

  本类持有可训练网络、状态/动作统计和动作尺度 buffer。世界模型 learner 更新其
  参数；策略 learner 冻结其参数后仍可通过输出对动作的导数训练 actor。
  """

  # 函数作用：确定动力学输入输出宽度，并建立动作变换与归一化网络。
  def __init__(
    self,
    representation: StateRepresentation,
    num_actions: int,
    sim_ctx_dim: int,
    action_scale: torch.Tensor,
    action_offset: torch.Tensor,
    hidden_dims: tuple[int, ...],
  ):
    """确定动力学输入输出宽度，并建立动作变换与归一化网络。

    representation 提供 rep_state_dim、prediction_dim、dt 和 apply_prediction；
    num_actions=A、sim_ctx_dim=C；action_scale/offset 为逐动作命令映射参数，
    hidden_dims 控制隐藏层。MLP 输入宽度 R+A+C，输出为表示约定的增量宽度。

    返回 None；克隆的尺度/偏置注册为 buffer，随设备迁移和 state_dict 保存。
    _apply_prediction 保存表示的绑定方法，后续 forward 委托该方法积分。
    """
    # 【源码讲解】
    # 网络输入宽度 = 状态表示宽度 + 动作数 + 仿真上下文宽度。
    # 输出宽度由表示决定，是待积分的速度增量，不直接等于完整状态宽度。
    # 动作尺度和偏置注册为 buffer，会随模型迁移设备并进入 state_dict。
    super().__init__()

    self.num_rep_state = representation.rep_state_dim
    self.num_actions = num_actions
    self.dt = representation.dt
    self._apply_prediction = representation.apply_prediction

    self.register_buffer("action_scale", action_scale.clone())
    self.register_buffer("action_offset", action_offset.clone())

    self.state_normalizer = _CudaGraphNormalization(self.num_rep_state)
    self.action_normalizer = _CudaGraphNormalization(num_actions)

    self.mlp = build_mlp(
      self.num_rep_state + num_actions + sim_ctx_dim,
      hidden_dims,
      output_dim=representation.prediction_dim,
    )

  # 函数作用：把原始策略动作转换为模型使用的物理命令尺度。
  def proc(self, action: torch.Tensor) -> torch.Tensor:
    """把原始策略动作转换为模型使用的物理命令尺度。

    参数 action:[...,A]；返回 action*action_scale+action_offset，形状不变。
    尺度和偏置按末轴广播。回放中保存原始动作，因此前向与统计更新都要在同一位置
    应用这项变换，不能调用前先缩放一次再让此处重复缩放。
    """
    # 【源码讲解】
    # 原始策略动作映射为关节命令：a_proc=a_raw*scale+offset。
    # 经验池保存原始动作，只有此路径负责动力学输入的尺度变换，避免重复缩放。
    return action * self.action_scale + self.action_offset

  # 函数作用：预测一个可微控制步，返回 (next_aux_state,next_rep_state)。
  def forward(
    self,
    rep_state: torch.Tensor,
    action: torch.Tensor,
    aux_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """预测一个可微控制步，返回 (next_aux_state,next_rep_state)。

    参数 rep_state:[...,R]、action:[...,A]、aux_state:[...,7] 和 sim_ctx:[...,C]
    共享前导批次轴；辅助状态为根位置与四元数。先归一化表示及 proc(action)，
    与已归一化上下文拼接后经 MLP 得到增量，再以当前 dt 调用表示积分器。

    返回顺序是辅助根位姿在前、模型表示在后。此函数不访问物理引擎、不更新参考
    时刻；自动微分可沿积分与 MLP 追溯到输入动作，供策略时间反传使用。
    """
    # rep_state 是当前状态表示；action 是策略原始输出，不是参考动作片段。
    # 将动作映射到实际命令尺度后归一化，并与状态和随机化上下文拼接。
    # 【源码讲解】
    # rep_state 通常为 [B,D_rep]，action 为 [B,A]，sim_ctx 为 [B,C]。
    # 先归一化状态与处理后的动作，再沿最后一维拼接并送入 MLP。
    # aux_state 保存根位姿，传给积分器；返回顺序为下一 aux、下一 rep。
    state_norm = self.state_normalizer(rep_state)
    action_norm = self.action_normalizer(self.proc(action))
    # 网络预测速度增量，具体如何积分成下一状态由 representation 决定。
    prediction = self.mlp(torch.cat([state_norm, action_norm, sim_ctx], dim=-1))
    return self._apply_prediction(aux_state, rep_state, prediction, self.dt)

  # 函数作用：返回状态和动作经验归一化器的引用列表。
  def normalizers(self) -> list[_CudaGraphNormalization]:
    """返回状态和动作经验归一化器的引用列表。

    无参数，顺序固定为状态、动作，列表中的对象就是 forward 使用的统计模块。
    世界模型 learner 用它切换统计模式；返回引用不克隆、不刷新样本，也不改变参数。
    """
    return [self.state_normalizer, self.action_normalizer]


class MultiEncoderAE(nn.Module):
  """多种任务目标共用动作生成能力的策略网络。

  每种任务有独立目标归一化器和编码器，目标变成相同宽度的有界潜变量；
  共享解码器读取该潜变量与观察历史，输出原始动作。任务槽位是批次内连续区间，
  同一次前向可由不同编码器负责不同样本；奖励仍由外部想象环境统一计算。

  输入 TensorDict 包含 history:[B,H,D] 和 goal_任务名:[B,G_task]。
  第一个任务为锚点，辅助损失把非锚点潜变量拉向同状态下的锚点潜变量，
  不更新锚点分支。模型、归一化器均被注册为子模块，可随 state_dict 保存。
  """

  # 函数作用：为任务建立目标编码器、归一化器和共享动作解码器。
  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    task_specs: list[TaskSpec],
    slot_mix: dict[str, float],
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    latent_bc_weight: float,
    history_length: int,
    arm_swing_control: nn.Module | None = None,
  ):
    """为任务建立目标编码器、归一化器和共享动作解码器。

    obs_dim=D 是单帧宽度，history_length=H；action_dim=A；task_specs 决定
    任务顺序及各目标宽度，slot_mix 决定比例，latent_dim=Z，hidden_dims 为各 MLP
    隐藏宽度，latent_bc_weight 为对齐权重。解码器输入宽度为 Z+H*D。

    返回 None；空任务列表报错，首次调用 set_slot_mix 验证比例。任务顺序同时决定
    锚点与批次区间，改变顺序会改变路由语义。初始辅助损失为空，前向后才有逐样本值。
    """
    # 【源码讲解】
    # 每种任务有独立目标归一化器和编码器，所有任务共享动作解码器。
    # 解码器输入为 latent_dim + history_length*obs_dim，输出动作宽度。
    # 第一个任务作为潜变量对齐的锚点；slot_mix 指定批次中各任务的占比。
    super().__init__()
    if not task_specs:
      raise ValueError("task_specs must be non-empty")
    self.obs_dim = obs_dim
    self.history_length = history_length
    self.latent_dim = latent_dim
    self.latent_bc_weight = latent_bc_weight
    self.task_specs = task_specs
    self.arm_swing_control = arm_swing_control
    self._task_names = [ts.name for ts in task_specs]
    # Anchor: the first task. The aux-loss hook regularizes each non-anchor
    # encoder against the anchor encoder on the same state.
    self._anchor = self._task_names[0]
    self._aux_loss = torch.zeros(0)

    self.obs_normalizer = _CudaGraphNormalization(obs_dim)
    # ModuleDict 按任务名注册子模块，使编码器及其统计量进入 state_dict 并跟随设备迁移。
    self.goal_normalizers = nn.ModuleDict(
      {ts.name: _CudaGraphNormalization(ts.goal_dim) for ts in task_specs}
    )
    self.encoders = nn.ModuleDict(
      {
        ts.name: build_mlp(ts.goal_dim, hidden_dims, output_dim=latent_dim)
        for ts in task_specs
      }
    )
    # 解码器只接收统一宽度的潜变量和历史，因此不同目标模态可以共享动作生成参数。
    self.decoder = build_mlp(
      latent_dim + history_length * obs_dim,
      hidden_dims,
      output_dim=action_dim,
    )

    self.set_slot_mix(slot_mix)

  # 函数作用：验证并保存任务比例，供后续按当前批大小计算区间。
  def set_slot_mix(self, slot_mix: dict[str, float]) -> None:
    """验证并保存任务比例，供后续按当前批大小计算区间。

    参数 slot_mix 为任务名到非负比例的字典，缺省任务比例为零；未知名称、负值、
    总和偏离 1 超过容差时抛错。返回 None，写入 _slot_fractions。
    这里不固定某一批的整数样本数，因此同一 actor 可服务物理与想象的不同批大小。
    """
    # 【源码讲解】
    # 比例必须非负且总和约为 1；未提供的任务视为 0，未知任务名直接报错。
    # 这里只设置比例，实际每个任务分到的整数区间随 forward 的 batch 大小计算。
    unknown = set(slot_mix) - set(self._task_names)
    if unknown:
      raise ValueError(f"slot_mix has unknown tasks {sorted(unknown)}")
    fractions = [slot_mix.get(name, 0.0) for name in self._task_names]
    if any(frac < 0.0 for frac in fractions):
      raise ValueError(f"slot_mix fractions must be non-negative; got {slot_mix}")
    total = sum(fractions)
    if abs(total - 1.0) > 1e-6:
      raise ValueError(f"slot_mix must sum to 1.0; got {total}")
    self._slot_fractions = fractions

  # 函数作用：把保存的比例换成当前批次的左闭右开区间。
  def slot_ranges(self, batch_size: int) -> list[tuple[int, int]]:
    """把保存的比例换成当前批次的左闭右开区间。

    参数 batch_size=B；返回与任务顺序一致的 [(lo,hi),...]。前面任务按比例四舍五入，
    最后任务直接取 batch_size 为终点吸收余数；空区间由前向跳过。
    只计算索引，不切分或复制目标张量；forward 和统计更新都使用这套区间约定。
    """
    ranges: list[tuple[int, int]] = []
    # off 指当前任务区间的起点；区间为左闭右开 [off,hi)，后一个任务紧接前一个。
    off = 0
    last = len(self._slot_fractions) - 1
    for i, frac in enumerate(self._slot_fractions):
      # 前面任务按比例取整，最后一个任务接收剩余样本，确保覆盖整个 batch。
      hi = batch_size if i == last else off + int(round(batch_size * frac))
      ranges.append((off, hi))
      off = hi
    return ranges

  # 函数作用：从多任务观察生成 [B,A] 原始控制动作。
  def forward(self, obs_td: TensorDict) -> torch.Tensor:
    """从多任务观察生成 [B,A] 原始控制动作。

    obs_td 必须含 history:[B,H,D] 及占槽任务目标 goal_name:[B,G_name]。
    先按任务区间切目标、归一化并编码；按原批次顺序拼接 tanh 后潜变量，
    再将归一化历史展平为 [B,H*D]，与潜变量一起交共享解码器。

    除返回动作外，函数还刷新 _aux_loss 供同一步策略损失调用；不会执行优化器。
    任务切分仅作用于目标编码器，历史和解码器仍按完整批次处理，样本顺序不能打乱。
    """
    # batch 按任务分成连续区间：各任务用各自编码器，共享同一个动作解码器。
    # 【源码讲解】
    # 输入包含 history:[B,H,D] 及各任务 goal_name:[B,G_name]。
    # 批次连续分段后，各段目标经过对应编码器，按原顺序拼接潜变量。
    # 潜变量 tanh 有界化，历史归一化并展开为 [B,H*D]，共享解码器输出 [B,A]。
    # 根据本次输入批次大小重新分配区间，同一 actor 可同时用于不同并行数的采集和想象。
    slots = self.slot_ranges(obs_td.shape[0])
    enc_outs: list[torch.Tensor] = []
    for name, (lo, hi) in zip(self._task_names, slots, strict=True):
      # 占比为零或因小批次取整得到空区间时跳过，避免对空张量执行编码计算。
      if lo == hi:
        continue
      goal_n = self.goal_normalizers[name](obs_td[f"goal_{name}"][lo:hi])
      enc_outs.append(self.encoders[name](goal_n))

    # 前向时缓存逐样本辅助损失，policy learner 在本步调用 auxiliary_loss 将其纳入目标。
    self._aux_loss = self._compute_aux_loss(obs_td, slots)

    # tanh 将潜变量限制在 [-1, 1]；再与观察历史拼接，生成当前动作。
    z = torch.cat([torch.tanh(eo) for eo in enc_outs], dim=0)
    # 归一化沿最后特征轴应用，保留 [B,H,D]；flatten(start_dim=1) 再将历史展开。
    history_n = self.obs_normalizer(obs_td["history"])
    actions = self.decoder(torch.cat([z, history_n.flatten(start_dim=1)], dim=-1))
    if self.arm_swing_control is not None:
      lo, hi = slots[self._task_names.index("velocity")]
      if lo < hi:
        adjusted = self.arm_swing_control(
          actions[lo:hi], obs_td["history"][lo:hi, -1], obs_td["goal_velocity"][lo:hi]
        )
        actions = torch.cat((actions[:lo], adjusted, actions[hi:]), dim=0)
    return actions

  # 函数作用：计算非锚点任务的潜变量模仿损失，返回逐样本 [B]。
  def _compute_aux_loss(
    self, obs_td: TensorDict, slots: list[tuple[int, int]]
  ) -> torch.Tensor:
    """计算非锚点任务的潜变量模仿损失，返回逐样本 [B]。

    参数 obs_td 为整批目标，slots 是同次前向的任务区间。对每个非锚点区间，
    把本任务编码与同样本的锚点编码比较，使用 tanh 后均方差乘 latent_bc_weight。
    输入目标 detach，锚点前向 no_grad，因此该损失仅训练非锚点编码器。

    评估模式或权重为零时直接返回零；锚点样本和空区间始终贡献零。
    它不通过共享解码器，也不把目标来源的梯度反传回想象状态。
    """
    # 【源码讲解】
    # 返回逐样本辅助损失 [B]，锚点和空槽保持 0。
    # 非锚点编码器在同一状态下拟合锚点潜变量；目标输入 detach，锚点 no_grad。
    # 因此该项只更新非锚点编码器，不把梯度传到目标来源或锚点分支。
    out = torch.zeros(obs_td.shape[0], device=obs_td.device)
    # 播放评估或辅助权重为零时直接返回零向量，省去额外编码器计算。
    if not self.training or self.latent_bc_weight == 0.0:
      return out
    anchor = self._anchor
    anchor_goal_d = self.goal_normalizers[anchor](obs_td[f"goal_{anchor}"]).detach()
    for name, (lo, hi) in zip(self._task_names, slots, strict=True):
      if name == anchor or lo == hi:
        continue
      goal_n_d = self.goal_normalizers[name](obs_td[f"goal_{name}"][lo:hi]).detach()
      z = torch.tanh(self.encoders[name](goal_n_d))
      # 锚点编码器只提供目标；辅助损失的梯度只更新当前非锚点编码器。
      with torch.no_grad():
        z_anchor = torch.tanh(self.encoders[anchor](anchor_goal_d[lo:hi]))
      out[lo:hi] = self.latent_bc_weight * ((z - z_anchor) ** 2).mean(dim=-1)
    return out

  # 函数作用：取得最近一次前向缓存的逐样本辅助损失。
  def auxiliary_loss(self) -> torch.Tensor:
    """取得最近一次前向缓存的逐样本辅助损失。

    无参数，返回 _aux_loss 本身而非 detach 副本，策略 learner 可把它加入回报目标
    一起反传。应在对应 forward 后读取；调用本函数不会重新计算编码器或清空缓存。
    """
    return self._aux_loss

  # 函数作用：取得便于日志记录的辅助损失均值。
  def auxiliary_stats(self) -> dict[str, torch.Tensor]:
    """取得便于日志记录的辅助损失均值。

    无参数，前向尚未产生损失时返回空字典，否则返回 bc_loss 对应的已 detach 标量。
    与 auxiliary_loss 不同，此结果只用于记录，不保留策略梯度计算图。
    """
    if self._aux_loss.numel() == 0:
      return {}
    return {"bc_loss": self._aux_loss.mean().detach()}

  # 函数作用：用新采集输入刷新单帧观察和各任务目标的经验统计。
  @torch.jit.unused
  def update_normalization(self, obs_td: TensorDict) -> None:
    """用新采集输入刷新单帧观察和各任务目标的经验统计。

    参数 obs_td 与 forward 布局相同。观察仅取 history[:,-1]，避免重复旧历史；
    目标按同样任务区间送给对应归一化器，空区间跳过。返回 None。

    调用者应将 actor 置于训练模式，并在物理采集 inference_mode 外执行。
    本函数不训练编码器/解码器，也不对想象展开的每帧重新积累真实数据统计。
    """
    # 只用最新真实帧更新观察统计，不把重复历史再次计入同一次采集的样本量。
    obs = obs_td["history"][:, -1]
    self.obs_normalizer.update(obs)
    for name, (lo, hi) in zip(
      self._task_names, self.slot_ranges(obs.shape[0]), strict=True
    ):
      # 占比为零或因小批次取整得到空区间时跳过，避免对空张量执行编码计算。
      if lo == hi:
        continue
      # 每个目标归一化器只统计该任务实际占用的槽位，不混合其他任务的目标分布。
      self.goal_normalizers[name].update(obs_td[f"goal_{name}"][lo:hi])
