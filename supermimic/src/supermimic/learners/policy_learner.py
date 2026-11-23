"""通过可微世界模型展开想象轨迹，训练多任务动作策略。

BpttPolicyLearner 最大化有限步折扣回报；ActorCriticPolicyLearner 继承相同策略更新，
增加末端价值补偿和 GAE 价值拟合。动力学在策略阶段冻结参数，但其输出对动作仍可微。
物理仿真不参与反向传播，真实回放仅提供起点；模型拟合由 supermimic/src/supermimic/learners/wm_learner.py 单独执行。

B=想象轨迹批大小，L=想象步数，H=策略传感器历史长度。rollout 给 critic 使用时
采用 [L,B,...] 的时间优先布局，且保存的副本已 detach。
"""

# 【文件作用】可微策略学习器
# 在世界模型内展开轨迹并通过时间反向传播优化策略，可选 critic 提供末端价值并拟合 GAE 目标。
# 【阅读顺序】策略配置 → GAE → BPTT 轨迹优化 → actor-critic bootstrap 和价值拟合。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟类型注解求值，使 TYPE_CHECKING 内的类型仅用于静态检查而无需运行时导入。
from __future__ import annotations



# dataclass 自动生成训练配置对象的初始化，供入口的类型驱动命令行解析使用。
from dataclasses import dataclass

# PyTorch 提供张量索引、批次变换、自动微分和编译；本文件的数据形状约定由这些运算实现。
import torch
# nn.Module 是策略及价值网络接口类型，网络实现由 runner 构造后注入学习器。
import torch.nn as nn
# 共享 batch_size 的张量字典，统一承载状态、历史、任务目标和训练批次，不只是任意键值集合。
from tensordict import TensorDict

# 来自 supermimic/src/supermimic/envs/world_model_env.py：提供起点采样、可微
# reset/step、奖励及软终止，策略学习器不直接调用物理引擎。
from supermimic.envs.world_model_env import WorldModelEnv
# 来自 supermimic/src/supermimic/learners/learner_utils.py：build_optimizers
# 创建优化器；_optimize_step 执行清梯度、反传、裁剪和更新；CapturedUpdate 与 graph_state 管理重复更新的图捕获状态。
from supermimic.learners.learner_utils import (
  CapturedUpdate,
  _optimize_step,
  build_optimizers,
  graph_state,
)


# BPTT = Backpropagation Through Time，即对多步展开的计算图反向传播。
# 这里的训练轨迹由可微世界模型生成，而不是对 MuJoCo 物理引擎反传。
@dataclass
class BpttPolicyLearnerCfg:
  """纯 BPTT 策略学习配置，决定想象轨迹规模与优化稳定性。

  batch_size 是每批想象起点数，num_batches 是一次 update 处理几批，horizon 为每条
  想象轨迹的步数。noise 供 runner 在真实采集动作上加探索噪声，不在此处的想象动作上加。
  gamma 为未来回报折扣，survival_masking 控制累积软存活权重。
  beta2 调节优化器二阶矩估计；grad_clip_value 限制梯度元素，此外还有整体范数裁剪。
  """
  # 学习率为 0.0002；一批包含 batch_size 个想象轨迹起点。
  lr: float = 2e-4
  num_batches: int = 1
  batch_size: int = 4096
  # noise 供物理仿真采集时加入动作噪声；horizon 是想象展开的步数。
  noise: float = 0.1
  horizon: int = 32
  # gamma 是未来奖励折扣；survival_masking 控制是否累计软存活权重。
  gamma: float = 1.0
  survival_masking: bool = True
  beta2: float = 0.95  # 梯度突变后二阶矩估计更快适应
  grad_clip_value: float = 0.1  # 逐参数梯度元素裁剪上限


@dataclass
# 默认 actor-critic 配置继承上述字段，但覆盖 gamma 并增加 GAE 的 lam。
class ActorCriticPolicyLearnerCfg(BpttPolicyLearnerCfg):
  """带价值补偿的 BPTT 配置，继承基础批次和优化设置。

  覆盖 gamma 为0.99，并增加 GAE 的 lambda=0.95；lambda 决定价值目标中的多步误差权重。
  该配置类型还被 runner 用于选择 ActorCriticPolicyLearner 分支。
  它不是 PPO 配置，actor 仍通过世界模型奖励直接反向传播。
  """
  gamma: float = 0.99
  lam: float = 0.95


# 函数作用：逆序计算广义优势估计，并加当前状态价值得到 critic 回归目标。
def gae_returns(
  values: torch.Tensor,
  last_value: torch.Tensor,
  rewards: torch.Tensor,
  dones: torch.Tensor,
  gamma: float,
  lam: float,
) -> torch.Tensor:
  """逆序计算广义优势估计，并加当前状态价值得到 critic 回归目标。

  values/rewards/dones 采用时间优先布局 [L,B,1]，last_value:[B,1] 是展开末端价值；
  gamma 为折扣，lam 为 GAE 衰减。dones 可以是软终止值，(1-done) 缩减后续价值。
  返回与 values 同形状的目标，不在本函数内部决定哪些输入需要 detach。

  逆序递推 delta=r+gamma*(1-done)*V_next-V，adv=delta+gamma*lam*(1-done)*adv_next，
  目标为 adv+V。调用者在 no_grad 下计算价值并用 detach 轨迹提供奖励，避免目标反传。
  """
  # values/rewards/dones 的第一维是时间；返回张量形状与 values 相同。
  # 初始化逆序递推的末端：下一状态价值从 last_value 开始。
  returns = torch.empty_like(values)
  advantage = 0.0
  next_value = last_value
  # reversed 从最后一步向前计算，使每一步都能使用已算出的后续优势。
  for t in reversed(range(values.shape[0])):
    not_done = 1.0 - dones[t]
    # TD 误差 = 即时奖励 + 折扣后的下一状态价值 - 当前状态价值。
    delta = rewards[t] + not_done * gamma * next_value - values[t]
    # 从后往前累积优势；软 done 通过 (1-done) 降低后续价值的权重。
    advantage = delta + not_done * gamma * lam * advantage
    # 价值回归目标 = 当前位置估值 + GAE 优势；随后把当前值作为前一步的下一值。
    returns[t] = advantage + values[t]
    next_value = values[t]
  return returns


class BpttPolicyLearner:
  """在想象环境中通过时间反向传播直接优化 actor 的学习器。

  持有策略网络、WorldModelEnv、独立 actor 优化器和更新执行包装。update 从真实回放抽
  起点并冻结动力学参数，_train_batch 生成可微动作/状态/奖励链，最小化负回报及辅助损失。
  基类不包含 critic，末端 bootstrap 为零；子类覆盖钩子可增加价值补偿与价值拟合。

  冻结参数与 no_grad 不同：前者仍允许奖励对动作求导，后者会切断所需路径。
  构造不会开始训练，runner 调用 update 才触发参数更新。
  """

  # 函数作用：保存 actor 和想象环境，并只为 actor 参数建立优化器。
  def __init__(
    self,
    policy: nn.Module,
    wm_env: WorldModelEnv,
    cfg: BpttPolicyLearnerCfg,
    device: str = "cpu",
  ):
    """保存 actor 和想象环境，并只为 actor 参数建立优化器。

    policy 为接受观察 TensorDict 并输出 [B,A] 动作的网络；wm_env 负责可微状态推进，
    cfg 控制批次和回报，device 决定是否编译 actor 前向及优化器实现。
    返回 None，actor_params 的固定列表用于优化和裁剪；不会将动力学参数加入该优化器。

    CapturedUpdate 包装 _update_from，并由 _graph_state 提供模型与优化器状态，
    便于重复更新时捕获/重放；真实计算流程仍在本类方法中。
    """
    # actor 是待优化策略；wm_env 是用学习到的动力学进行想象的环境。
    self.actor = policy
    self.wm_env = wm_env
    self.cfg = cfg
    # 只把 actor 参数交给这个优化器；critic 有自己单独的优化器。
    # actor 优化器只接收策略参数；模型虽在计算图中传递输入梯度，但不会被本优化器更新。
    self.actor_params = [p for p in self.actor.parameters() if p.requires_grad]
    self.actor_optimizers = build_optimizers(
      [self.actor], self.cfg.lr, betas=(0.9, self.cfg.beta2), device=device
    )
    self._rollout_actor = (
      torch.compile(self.actor) if torch.device(device).type == "cuda" else self.actor
    )
    # 把批次更新交给捕获器，图状态回调提供需保持一致的参数、缓冲与优化器张量。
    self._captured_update = CapturedUpdate(self._update_from, self._graph_state)

  # 函数作用：冻结世界模型参数，抽取真实状态起点并执行多批策略更新。
  def update(self) -> dict[str, torch.Tensor]:
    """冻结世界模型参数，抽取真实状态起点并执行多批策略更新。

    无参数，从 wm_env 的经验池采样 cfg.num_batches 批起点，堆叠后交给捕获更新入口。
    actor 设为训练模式，dynamics/representation 设为 eval 且 requires_grad=False，
    但不使用 no_grad 包裹想象轨迹，因此保留对动作输入的导数。
    返回 Policy/reward、梯度范数及可选价值/辅助损失等日志字典。
    """
    # 【源码讲解】
    # 模型参数冻结但模型对输入动作的导数仍保留，actor 可以穿过模型求梯度。
    # 从真实回放抽多个起点 batch，避免只在模型自发漂移的状态分布中训练。
    self.actor.train()
    # 冻结模型参数，但不切断输出对动作的梯度；actor 仍可通过世界模型学习。
    for module in (self.wm_env.dynamics, self.wm_env.representation):
      module.eval()
      module.requires_grad_(False)
    # 从真实仿真采集的回放数据中抽起点，每个起点 batch 再向前想象 horizon 步。
    starts = [self.wm_env.sample_initial_states() for _ in range(self.cfg.num_batches)]
    return self._captured_update(torch.stack(starts))

  # 函数作用：逐批训练想象轨迹，并按批数汇总策略指标。
  def _update_from(self, starts: TensorDict) -> dict[str, torch.Tensor]:
    """逐批训练想象轨迹，并按批数汇总策略指标。

    starts 的首轴是起点批次数，每一项由 _train_batch 负责一次 actor 更新。
    将各指标相加后平均；其中 return 另除 horizon 与控制步长 dt，生成每秒尺度的
    Policy/reward。它不是物理环境完整回合的总奖励，不能与回合奖励直接等同。
    返回命名后的标量指标，不在这里保存模型或推进物理环境。
    """
    # 对所有起点 batch 依次训练，将统计量相加，最后除以批次数。
    totals: dict[str, torch.Tensor] = {}
    for start in starts.unbind(0):
      for k, v in self._train_batch(start).items():
        totals[k] = totals[k] + v if k in totals else v
    n = self.cfg.num_batches
    # 日志用回报除以步数、批次数和 dt；它不是逐次物理环境 episode 的原始总回报。
    reward = totals.pop("return") / self.cfg.horizon / n / self.wm_env.representation.dt
    return {"Policy/reward": reward} | {f"Policy/{k}": v / n for k, v in totals.items()}

  # 函数作用：从一批真实起点展开可微轨迹，更新 actor 并调用可选价值拟合。
  def _train_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
    """从一批真实起点展开可微轨迹，更新 actor 并调用可选价值拟合。

    start 是 [B] 批次的回放快照，含完整状态、上一动作、参考帧、上下文及可选历史。
    重置想象环境后，连续 horizon 步由 actor 生成动作并预测奖励，累计折扣存活回报，
    另累计 actor 的潜变量辅助损失；末端通过 _bootstrap 加入可选未来价值。

    优化目标为 mean(-return+aux_loss)，梯度跨越多步状态和动力学传回 actor。
    为 critic 保存的 rollout 副本 detach，不切断 actor 仍使用的 reward 与 next_obs。
    返回已 detach 的回报、梯度范数及可选日志；随后 _fit_critic 单独更新价值网络。
    """
    # 【源码讲解】
    # 每次迭代串起 actor→动作→动力学→状态→奖励的可微链路。
    # 累计 gamma^t*survival_t*reward_t，末端可由 critic 补充未来价值。
    # rollout 中为 critic 留下的副本 detach，actor 使用的 reward 本身不 detach。
    # 最小化负回报和辅助损失，才等价于提高回报并约束潜变量。
    # reset 用回放样本初始化想象环境；obs 包含 actor 的输入和 critic 特权观察。
    cfg = self.cfg
    obs = self.wm_env.reset(start)
    # ret=累计折扣回报；aux_loss=策略附加损失；survival=之前各步存活权重。
    ret, aux_loss, survival = 0.0, 0.0, 1.0
    rollout: list[TensorDict] = []
    # 每次循环是一段想象中的控制步：actor 根据观察产生动作，模型预测后果。
    # 因为下一次 obs 依赖这次 actions，这些步骤连成跨时间的计算图。
    for t in range(cfg.horizon):
      actions = self._rollout_actor(obs)
      # step 返回下一观察、跟踪奖励、可微软终止值和额外信息；_ 表示此处忽略。
      next_obs, reward, term_prob, _ = self.wm_env.step(actions)
      step = {"critic": obs["critic"], "reward": reward, "done": term_prob}
      # critic 训练保存独立的轨迹副本；detach 只作用于这个副本，
      # 下面直接使用的 reward、term_prob 和 next_obs 仍保留 actor 的梯度路径。
      rollout.append(TensorDict(step, batch_size=reward.shape).detach())
      # 累加折扣奖励；survival 是前面各步 (1-软终止概率) 的乘积。
      ret = ret + cfg.gamma**t * survival * reward
      # getattr 优先调用 actor 的辅助损失；没有该方法时使用返回 0 的函数。
      aux_loss = aux_loss + getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
      # 当前步之后的存活权重乘以 (1 - 当前软终止值)，供未来奖励使用。
      if cfg.survival_masking:
        survival = survival * (1 - term_prob)
      obs = next_obs
    # 展开到末端后，可由 critic 补上未来价值；纯 BPTT 基类的该项为 0。
    ret = ret + self._bootstrap(obs, survival, term_prob)
    # 优化器最小化损失，因此最大化回报等价于最小化负回报。
    loss = (-ret + aux_loss).mean()
    # 在 _optimize_step 内执行反向传播和优化器 step；梯度沿奖励→状态→
    # 世界模型→动作→actor 的多步链路返回，世界模型参数自身保持冻结。
    grad_norm = _optimize_step(
      self.actor_optimizers, loss, self.actor_params, 1.0, cfg.grad_clip_value
    )
    return {
      "return": ret.mean().detach(),
      "grad_norm": grad_norm,
      **self._fit_critic(rollout, obs),
      **getattr(self.actor, "auxiliary_stats", lambda: {})(),
    }

  # 函数作用：纯 BPTT 的末端价值钩子，默认不补偿展开范围之外的收益。
  def _bootstrap(
    self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor
  ) -> torch.Tensor | float:
    """纯 BPTT 的末端价值钩子，默认不补偿展开范围之外的收益。

    为子类统一接收末端 obs、累积 survival 和最后一步 term_prob，本实现忽略它们并返回0。
    零是标量，可广播加到 [B] 的轨迹回报。ActorCriticPolicyLearner 覆盖此方法使用 critic，
    所以主训练循环无需为有无价值网络另写一套想象展开。
    """
    del obs, survival, term_prob
    return 0.0

  # 函数作用：纯 BPTT 的价值拟合钩子，默认不执行任何更新。
  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    """纯 BPTT 的价值拟合钩子，默认不执行任何更新。

    rollout 是逐步 detach 的观察/奖励/软终止记录，last_obs 是末端观察；本实现忽略输入，
    返回空字典供日志合并。子类可覆盖并返回 value_function 等指标。
    本钩子在 actor 更新之后调用，不改变本次已经完成的 actor 优化。
    """
    del rollout, last_obs
    return {}

  # 函数作用：列出捕获更新依赖、且 Python 可能重新绑定的状态张量。
  def _graph_state(self) -> list[torch.Tensor]:
    """列出捕获更新依赖、且 Python 可能重新绑定的状态张量。

    无参数，读取 actor、动力学、表示以及 actor 优化器，委托 graph_state 汇总参数、
    缓冲与优化器张量，返回张量列表。该信息服务 CUDA graph 的捕获一致性检查，
    不生成训练样本，不改变哪些参数接收梯度。
    """
    wm = self.wm_env
    modules = [self.actor, wm.dynamics, wm.representation]
    return graph_state(modules, self.actor_optimizers)


class ActorCriticPolicyLearner(BpttPolicyLearner):
  """在 BPTT 策略更新上增加价值补偿和独立 critic 训练。

  继承 actor 与想象轨迹流程，持有额外 critic 及其优化器。末端用 critic 估计有限展开
  之后的折扣回报，之后以 detach 的想象轨迹构造 GAE 目标拟合价值网络。
  策略和价值网络分别更新；它不是通过 PPO 比率或动作对数概率来训练 actor。
  价值上界由每步最大奖励/(1-gamma) 得到，用于限制过高价值估计。
  """

  # 函数作用：初始化父类 actor 更新设施，再绑定 critic 与独立优化器。
  def __init__(
    self,
    policy: nn.Module,
    critic: nn.Module,
    wm_env: WorldModelEnv,
    cfg: ActorCriticPolicyLearnerCfg,
    device: str = "cpu",
  ):
    """初始化父类 actor 更新设施，再绑定 critic 与独立优化器。

    policy、wm_env、device 含义与基类相同；critic 将特权观察映射为 [B,1]；cfg 必须
    提供 gamma 与 lam。返回 None，计算 _v_max 作为价值裁剪上界。
    当前公式含1/(1-gamma)，因此实际配置应避免 gamma=1；默认子配置为0.99。
    """
    super().__init__(policy, wm_env, cfg, device)
    self.critic = critic
    # 价值网络有独立优化器；actor 更新阶段只借用它评估末端状态，价值拟合阶段才更新权重。
    self.critic_optimizers = build_optimizers([critic], self.cfg.lr, device=device)
    # 无限折扣和的上界：若每步都达到最大正奖励，则上界为 r_max/(1-gamma)。
    self._v_max = self.wm_env.max_step_reward / (1 - self.cfg.gamma)

  # 函数作用：评估想象末端状态，返回每条轨迹的折扣价值补偿。
  def _bootstrap(
    self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor
  ) -> torch.Tensor:
    """评估想象末端状态，返回每条轨迹的折扣价值补偿。

    obs 包含末端 critic 特权观察；survival 为累计存活权重，term_prob 为最后一步软终止。
    临时冻结 critic 参数，保留价值对输入状态的梯度，再取标量价值并裁上界。
    返回 gamma^horizon*value*alive，形状 [B]；关闭累计存活时 alive 仍取最后一步非终止权重。
    不使用 no_grad，否则 actor 无法通过末端状态价值学习。
    """
    # 冻结 critic 参数，但不对输入 obs 使用 no_grad：末端价值仍能对 actor 传梯度。
    # 末端价值作为 actor 回报的一部分：冻结 critic 权重，仍保留价值对末端状态的梯度。
    self.critic.requires_grad_(False)
    value = self.critic(obs["critic"]).clamp(max=self._v_max).squeeze(-1)
    # 累计存活模式使用整段乘积；关闭时仍用最后一步的非终止权重抑制末端价值。
    alive = survival if self.cfg.survival_masking else 1.0 - term_prob
    return self.cfg.gamma**self.cfg.horizon * value * alive

  # 函数作用：用已 detach 的想象轨迹生成固定 GAE 目标并拟合 critic。
  def _fit_critic(
    self, rollout: list[TensorDict], last_obs: TensorDict
  ) -> dict[str, torch.Tensor]:
    """用已 detach 的想象轨迹生成固定 GAE 目标并拟合 critic。

    rollout 长度 L，每项批次为 [B]；last_obs 提供末端状态。先按时间堆叠，no_grad 下
    计算当前与末端价值，再调用 gae_returns 并裁目标上界。合并时间和 batch 后，
    取长度的4倍数部分随机排列，分4个小批次，固定目标训练5轮。

    返回 value_function 的平均均方误差。此阶段重新启用 critic 参数梯度，不把价值损失
    传回 actor 轨迹；actor 本轮更新已完成。样本尾数若不足整分4份则不参与。
    """
    # 把 H 个已 detach 的逐步数据堆叠成时间优先张量 [H,B,...]。
    # 【源码讲解】
    # 输入轨迹按 [时间,B,...] 堆叠，no_grad 下计算 GAE 固定目标。
    # 合并时间与批次后用小批次拟合价值，不反传到 actor 生成的轨迹。
    # 目标和末端价值使用同一上界约束，以限制过高价值估计。
    steps = torch.stack(rollout)
    # 固定目标：此处不为计算 GAE 目标建立 critic 参数的梯度图。
    with torch.no_grad():
      values = self.critic(steps["critic"])
      last_value = self.critic(last_obs["critic"])
    # critic 使用已 detach 的想象轨迹拟合 GAE 目标，不把 critic 更新反传到轨迹。
    returns = gae_returns(
      values,
      last_value,
      steps["reward"].unsqueeze(-1),
      steps["done"].unsqueeze(-1),
      self.cfg.gamma,
      self.cfg.lam,
    ).clamp(max=self._v_max)
    # 合并时间和 batch 轴，随后打乱并分为 4 份；不能整除 4 的尾数不参与。
    obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
    # 裁为 4 的整数倍再生成随机排列，便于后面 view(4,-1)；不足四份的尾部不参与。
    perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
    # actor 回传完成后重新启用 critic 参数梯度，本阶段仅最小化价值回归误差。
    self.critic.requires_grad_(True)
    losses = []
    # 固定 GAE 目标，循环 5 轮；每轮对 4 个小批次分别做价值均方误差更新。
    for _ in range(5):
      for idx in perm.view(4, -1):
        loss = (returns[idx] - self.critic(obs[idx])).pow(2).mean()
        _optimize_step(
          self.critic_optimizers, loss, self.critic.parameters(), max_norm=1.0
        )
        losses.append(loss.detach())
    return {"value_function": torch.stack(losses).mean()}

  # 函数作用：在父类捕获状态上补充 critic 与其优化器张量。
  def _graph_state(self) -> list[torch.Tensor]:
    """在父类捕获状态上补充 critic 与其优化器张量。

    无参数，返回拼接后的张量列表，使价值拟合也处于同一更新捕获状态检查范围。
    只收集引用，不执行价值前向、不建立新的优化器，也不改变参数训练开关。
    """
    return super()._graph_state() + graph_state([self.critic], self.critic_optimizers)
