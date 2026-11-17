"""用真实物理仿真序列监督训练可微动力学世界模型。

runner 从 ReplayBuffer 抽取不跨回合的窗口，本文件把完整状态转为模型表示，
从窗口首帧自回归预测后续状态，按表示定义的特征组计算损失并更新 Dynamics。
策略网络不在本优化器中；策略更新通过模型传递梯度是 supermimic/src/supermimic/learners/policy_learner.py 的职责。

B=窗口批大小，T=每窗口帧数，S=完整状态宽度，R=模型表示宽度，C=上下文宽度。
T 帧仅提供 T-1 个下一帧标签，动作末帧不会参与这次监督展开。
"""

# 【文件作用】世界模型监督学习器
# 用经验池中的连续仿真序列训练动力学，使自回归预测与真实后续状态对齐。
# 【阅读顺序】配置损失尺度 → 更新输入统计 → 多步展开 → 分组损失 → 反向传播和日志。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟类型注解求值，使 TYPE_CHECKING 内的类型仅用于静态检查而无需运行时导入。
from __future__ import annotations



# 声明 update 可消费窗口批次迭代器，runner 因而能以生成器逐批提供回放样本。
from collections.abc import Iterable
# dataclass 自动生成训练配置对象的初始化，供入口的类型驱动命令行解析使用。
from dataclasses import dataclass
# 仅静态类型检查时导入模型/表示类型，降低运行时循环导入的可能；实际对象由调用者传入。
from typing import TYPE_CHECKING

# PyTorch 提供张量索引、批次变换、自动微分和编译；本文件的数据形状约定由这些运算实现。
import torch
# 共享 batch_size 的张量字典，统一承载状态、历史、任务目标和训练批次，不只是任意键值集合。
from tensordict import TensorDict

# 来自 supermimic/src/supermimic/learners/learner_utils.py：build_optimizers
# 创建优化器；_optimize_step 执行清梯度、反传、裁剪和更新；CapturedUpdate 与 graph_state 管理重复更新的图捕获状态。
from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
  graph_state,
)

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.representations import StateRepresentation


# 世界模型使用“监督学习”：输入仿真中的状态和动作，用下一帧作为正确答案。
# dataclass 把下面的字段组织成配置对象，调用者可用命令行覆盖默认值。
@dataclass
class WorldModelLearnerCfg:
  """世界模型监督学习的批次规模与误差尺度配置。

  lr 控制动力学优化器步长；num_batches 是一次 update 消费的窗口批数，
  batch_size 是每批序列数，horizon 是窗口帧数而非预测步数。
  nominal_error_* 为各物理特征的标称误差，权重按其平方倒数生成；它不是训练精度保证。
  不同状态表示仅使用其 LOSS_WEIGHT_CONFIG 指定的组，未使用字段仍可存在于配置中。
  """
  # lr 是学习率；2e-4 是科学计数法，即 0.0002。
  lr: float = 2e-4
  # 每次 update 处理多少批；每批多少条连续序列；每条序列包含多少帧。
  # 默认一条 8 帧窗口只能提供 7 个窗口内的下一帧监督目标。
  num_batches: int = 10
  batch_size: int = 2048
  horizon: int = 8

  # 标称误差是损失的尺度参数，不是训练后必须达到的精度；权重为 1/误差²。
  # 两种表示共用的积分根位姿标称误差，按 Bryson 规则生成权重。
  nominal_error_root_pose_pos: float = 0.01
  nominal_error_root_pose_ori: float = 0.02

  # 广义坐标特有的关节位置与速度标称误差。
  nominal_error_joint_pos: float = 0.1
  nominal_error_joint_vel: float = 3.14

  # 最大坐标特有的刚体位置、方向、线速度和角速度标称误差。
  nominal_error_body_pos: float = 0.05
  nominal_error_body_rot: float = 0.1
  nominal_error_body_vel: float = 1.0
  nominal_error_body_avel: float = 3.14


class WorldModelLearner:
  """管理 Dynamics 的优化器、输入统计和多步监督更新。

  依赖共享的 Dynamics 与 StateRepresentation：前者含可训练网络，后者定义状态转换、
  积分和损失分组。构造时建立优化器及可编译/可捕获更新包装；不从环境直接采集数据。
  runner 先调用 update_normalization 刷新新样本统计，再按更新频率调用 update。

  与策略学习交替时，策略阶段会冻结动力学参数，因此 update 会重新启用参数梯度。
  本类仅更新 dynamics 参数；返回的损失指标已从训练图 detach，供日志记录。
  """

  # 函数作用：绑定模型依赖，创建优化器、分组损失权重与更新执行包装。
  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    cfg: WorldModelLearnerCfg,
    device: str = "cpu",
  ) -> None:
    """绑定模型依赖，创建优化器、分组损失权重与更新执行包装。

    dynamics/representation 是 runner 共享对象，不会复制；cfg 决定学习率和组尺度，
    device 交给优化器构建工具。按 dynamics.parameters 收集待训练参数，再读取表示的
    LOSS_WEIGHT_CONFIG，将每个组映射为1/标称误差²。缺少对应配置字段会在此暴露。

    返回 None，保存编译后的表示、展开与损失函数，并以 CapturedUpdate 包装 _update_from。
    包装器用于降低重复训练的执行开销；实际监督算法仍由下面的方法实现。
    """
    self.dynamics = dynamics
    self.representation = representation
    self.cfg = cfg
    self.device = device

    # parameters() 返回网络中可训练的权重和偏置；转成列表供优化和裁剪使用。
    # 本 learner 的优化器只训练 dynamics，不把 representation 当作拟合参数。
    # 优化器只更新动力学参数；保存成列表可反复用于梯度裁剪和图捕获状态跟踪。
    self._all_params = list(dynamics.parameters())
    self.optimizers = build_optimizers([dynamics], self.cfg.lr, device=self.device)

    # Bryson 规则：每个特征组的权重为其标称误差平方的倒数。
    # 根据当前表示支持的损失组读取对应配置：例如关节位置、根位置等。
    # getattr(cfg, attr) 等价于按 attr 字符串指定的名字读取 cfg 的属性。
    self.loss_group_weights = {
      name: 1.0 / getattr(cfg, attr) ** 2
      for name, attr in representation.LOSS_WEIGHT_CONFIG.items()
    }

    # 编译这些张量计算以尝试减少执行开销；CapturedUpdate 还管理可用时的
    # CUDA graph 捕获/重放，实际更新逻辑仍在下面的 _update_from。
    self._compiled_rollout = torch.compile(self._rollout)
    self._compiled_compute_losses = torch.compile(self._compute_losses)
    self._compiled_represent = torch.compile(self.representation.represent)
    self._captured_update = CapturedUpdate(
      self._update_from,
      lambda: graph_state([dynamics, representation], self.optimizers),
    )

  # 函数作用：从新采集完整状态和原始动作更新动力学输入归一化统计。
  def update_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
    """从新采集完整状态和原始动作更新动力学输入归一化统计。

    states 最后一维为 S，actions 最后一维为 A，前面的样本维会合并；先将状态转为 R，
    动作经 Dynamics.proc 映射到关节命令尺度，与 forward 的输入保持一致。
    归一化器在 eval 模式不更新，因此临时设为 train，刷新后统一设回 eval。

    返回 None，只修改统计 buffer，不计算监督损失、不更新网络权重。
    runner 在物理采集 inference_mode 之外调用此方法，避免在推理作用域更新训练状态。
    """
    # 新采集的 full state 先转换为世界模型使用的表示；不是在此处训练网络。
    rep_states = self.representation.represent(states)
    # reshape(-1, 最后一维宽度) 合并前面的样本维，以便统计每个特征的尺度。
    rep_states_flat = rep_states.reshape(-1, rep_states.shape[-1])
    actions_flat = actions.reshape(-1, actions.shape[-1])

    # 经验归一化器在评估模式不会更新，先切到训练模式刷新统计，完成后统一切回评估模式。
    normalizers = self.dynamics.normalizers()
    for n in normalizers:
      n.train()

    # 输入归一化器积累状态和“缩放/偏置后的动作”统计量，与网络前向输入一致。
    self.dynamics.state_normalizer.update(rep_states_flat)
    self.dynamics.action_normalizer.update(self.dynamics.proc(actions_flat))

    for n in normalizers:
      n.eval()

  # runner 从回放池抽取连续窗口，调用 update(data)；data 逐批提供 TensorDict。
  # 函数作用：消费窗口批次，恢复动力学训练梯度并执行一次多批监督更新。
  def update(self, data: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
    """消费窗口批次，恢复动力学训练梯度并执行一次多批监督更新。

    data 为可迭代 TensorDict，每项含 state:[B,T,S]、action:[B,T,A]、sim_ctx:[B,T,C]。
    只选这三个字段，并沿新最前轴堆成 [num_batches,B,T,...] 交给捕获包装器。
    返回 Loss/state 与各分组 MSE 的标量字典。

    会把 dynamics 设为训练模式并重新开启参数梯度；实际每批 optimizer 更新发生在
    _update_from。归一化统计刷新由独立入口负责，不能把这里等同于新样本统计更新。
    """
    # 恢复模型训练模式；归一化统计的显式刷新仍由新采集数据路径负责。
    self.dynamics.train()
    # 策略更新会冻结这些参数；模型训练前重新开启其梯度，保证交替更新正确切换。
    for p in self._all_params:
      p.requires_grad_(True)
    # 只选择世界模型需要的字段，再新增最前面的批次轴并堆叠。
    # state: [B,T,D状态]；action: [B,T,D动作]；sim_ctx 是物理随机化上下文。
    batches = torch.stack([td.select("state", "action", "sim_ctx") for td in data])
    return self._captured_update(batches)

  # 函数作用：对已堆叠的每批窗口依次拟合，并返回跨批平均指标。
  def _update_from(self, batches: TensorDict) -> dict[str, torch.Tensor]:
    """对已堆叠的每批窗口依次拟合，并返回跨批平均指标。

    batches 的批次轴为 [批数,B,T]。每批先把真实完整状态转换为表示和根辅助状态，
    只从首帧取预测起点和物理上下文，然后用采集动作自回归展开 T-1 步。
    预测与真实[:,1:]对齐，计算加权状态损失并执行一次优化，梯度范数上限为1。

    返回各批 Loss/state 和原始组 MSE 的均值；统计张量 detach，避免日志保留整张训练图。
    展开期间不把预测状态换回真值，因此损失同时约束累积预测误差。
    """
    stats = []

    # unbind(0) 沿最前面的批次轴拆开；每次循环训练一个 batch。
    for td in batches.unbind(0):
      full_states = td["state"]
      actions = td["action"]

      # full_states 的常见形状为 [B, T, D]：批次、连续时间步、状态特征。
      all_rep_states = self._compiled_represent(full_states)

      # aux 保存根位置和四元数；[:, 0, :] 取每条序列的首帧作为预测起点。
      # 切片冒号表示该维全取，0 是时间下标，最后一个冒号取所有特征。
      all_aux_states = self.representation.extract_aux_state(full_states)
      init_rep_state = all_rep_states[:, 0, :]
      init_aux = all_aux_states[:, 0, :]

      # 取窗口首帧的随机化参数；本次预测展开沿用这份上下文。
      sim_ctx = td["sim_ctx"][:, 0]

      # 从首帧出发，用窗口内动作自回归地预测未来状态；形状是 [B,T-1,...]。
      pred_rep_states, pred_aux_states = self._compiled_rollout(
        init_rep_state, actions, init_aux, sim_ctx
      )

      # 预测的第 1～T-1 帧与真实序列 [:,1:] 一一对应，计算各组误差。
      # 以向量化方式计算预测与标签的分组损失，编译器可融合部分张量运算。
      loss_state, group_mse = self._compiled_compute_losses(
        pred_rep_states,
        pred_aux_states,
        all_rep_states[:, 1:],
        all_aux_states[:, 1:],
      )

      # 优化步骤包含清梯度、loss.backward()、梯度范数裁剪及参数更新；
      # 具体实现在 supermimic/src/supermimic/learners/learner_utils.py 的 _optimize_step。
      # 反向传播并更新本批次的动力学参数。
      _optimize_step(self.optimizers, loss_state, self._all_params, max_norm=1.0)

      # detach() 让日志张量脱离梯度图，避免仅为记录而保留整张计算图。
      # 缓存脱离计算图的本批损失指标。
      stats.append(torch.cat([loss_state.unsqueeze(0), group_mse]).detach())

    # 聚合各 batch 的统计值；Loss/state 是加权总损失，其余是各组原始 MSE。
    names = ["state", *self.loss_group_weights]
    means = torch.stack(stats).mean(0)
    return {f"Loss/{name}": m for name, m in zip(names, means, strict=True)}

  # 函数作用：从一个真实起点沿窗口动作自回归预测后续状态。
  def _rollout(
    self,
    init_rep_state: torch.Tensor,
    actions: torch.Tensor,
    init_aux: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """从一个真实起点沿窗口动作自回归预测后续状态。

    init_rep_state:[B,R]、init_aux:[B,7]、actions:[B,T,A]、sim_ctx:[B,C]。
    每步调用 dynamics，上一预测继续作为下一步输入；上下文在整个窗口保持不变。
    返回 (pred_rep_states,pred_aux_states)，形状 [B,T-1,R] 与 [B,T-1,7]。

    T 帧只有 T-1 个下一帧标签，最后动作的后果不在窗口内，故不计算它。
    例如四帧序列用 a0、a1、a2 预测 s1、s2、s3，整段保留梯度而不 detach。
    """
    # 【源码讲解】
    # 以四帧为例：(s0,a0)预测s1，再用预测s1与a1预测s2，最后预测s3。
    # 总计 T-1 次调用，最后一个动作没有窗口内的下一帧标签。
    # 预测状态不 detach，早期增量会通过后续预测影响总损失和梯度。
    # 想象一个四帧窗口：用 s0+a0 得到预测 s1，再用预测 s1+a1 得到预测 s2。
    # 不在每一步强行换回真实状态，因此多步预测误差会累积并参与训练。
    rep_state, aux = init_rep_state, init_aux
    pred_rep_states: list[torch.Tensor] = []
    pred_aux_states: list[torch.Tensor] = []
    # T 帧只有 T-1 个窗口内的下一帧真值；上一步预测继续作为下一步输入。
    for k in range(actions.shape[1] - 1):
      # dynamics 的返回顺序是 (下一根位姿 aux, 下一表示 rep_state)。
      aux, rep_state = self.dynamics(rep_state, actions[:, k], aux, sim_ctx)
      pred_rep_states.append(rep_state)
      pred_aux_states.append(aux)
    # 列表每项是一个预测时间步，沿 dim=1 堆叠后得到 [B,T-1,D]，与真值[:,1:]对齐。
    return torch.stack(pred_rep_states, dim=1), torch.stack(pred_aux_states, dim=1)

  # 函数作用：按物理特征组比较预测与真值，生成可优化的标量和诊断指标。
  def _compute_losses(
    self,
    pred_rep_states: torch.Tensor,
    pred_aux_states: torch.Tensor,
    gt_rep_states: torch.Tensor,
    gt_aux_states: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """按物理特征组比较预测与真值，生成可优化的标量和诊断指标。

    四个输入为相同批次/时间布局的预测与真实 rep、aux；通过表示的 loss_blocks 拆组。
    每组先计算逐通道平方误差及全体均值，记录为 group_mse；再乘1/标称误差²，
    按组通道宽度加权并除以总通道数，得到 loss_state。

    返回 (loss_state,group_mse)，后者顺序与表示的组字典一致。组宽度/权重在设备上
    用 new_full 构造，避免 CUDA graph 捕获更新过程中从主机拷贝动态标量。
    """
    # 各特征组先求平方误差，按标称尺度加权后再按通道数形成总体均值。
    # 先把预测和真实状态拆成同名组；不同表示定义各自需要比较的特征。
    # 【源码讲解】
    # 先依表示定义拆成同名特征组，对各组计算跨 batch、时间与通道的 MSE。
    # 每组乘以 1/标称误差²，再按组通道数加权，除以总通道数。
    # 返回加权标量用于优化，分组原始 MSE 用于定位哪类物理量误差较大。
    loss_blocks = self.representation.loss_blocks
    pred_blocks = loss_blocks(pred_rep_states, pred_aux_states)
    gt_blocks = loss_blocks(gt_rep_states, gt_aux_states)
    # 对每个元素求 (预测-真值)²；mean() 再对该组 batch、时间、通道求均值。
    sq_errs = {name: (pred_blocks[name] - gt_blocks[name]) ** 2 for name in pred_blocks}
    group_mse = torch.stack([sq.mean() for sq in sq_errs.values()])
    # 标量直接在相同设备上创建，避免捕获更新时从主机拷贝。
    # widths 是各组通道数；new_full 在相同设备上创建标量，兼容捕获更新。
    widths = torch.stack(
      [group_mse.new_full((), sq.shape[-1]) for sq in sq_errs.values()]
    )
    weights = torch.stack(
      [group_mse.new_full((), self.loss_group_weights[name]) for name in sq_errs]
    )
    # 每组权重为 1/标称误差²；乘通道数后除以总通道数，得到整体加权均值。
    loss_state = (group_mse * weights * widths).sum() / widths.sum()
    return loss_state, group_mse
