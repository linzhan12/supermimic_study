"""根据物理采集回报调整参考状态初始化（RSI）的时间桶分布。

文件位置：supermimic/src/supermimic/runners/reset_sampling.py。
参考动作帧按 motion 命令的相同规则映射到 bin_count 个桶。ReturnSampler 仅估计
各桶的回报并产生概率，实际重置由 motion 命令执行；它不采样世界模型的想象起点。
调用链为 runner.step → record → 满 horizon 时 _returns/_update → _distribution，
随后 runner 把返回概率交给 motion_cmd.set_sampling_distribution。
所有统计来自物理轨迹，不使用神经 critic；本模块也不反向传播或执行优化器更新。
"""

# 【文件作用】基于回报的重置采样
# 估计参考动作各时间桶的跟踪回报，将更多重置机会分配给低回报片段，同时保留探索和历史练习。
# 【阅读顺序】按步记录 → 短窗口回报 → 分桶价值更新 → explore/focus/keep 混合分布。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


from __future__ import annotations



# dataclass 为 ReturnSamplerCfg 生成配置初始化接口；配置本身不保存桶统计。
from dataclasses import dataclass

# PyTorch 在采集设备上完成缓存、回报逆推、unique/scatter 分组和概率归一化。
# 这里使用张量加速统计，并不构建神经网络或优化器。
import torch


@dataclass
class ReturnSamplerCfg:
  """控制桶价值估计及重置概率混合方式的配置。

  gamma 决定回报折扣与理论上界；horizon 是多少个物理控制步更新一次，并非 actor
  想象长度。memory 限制旧桶样本的有效计数，使统计能跟随正在变化的策略。
  steepness 越大越偏向当前低回报桶；explore 保留均匀覆盖，keep 保留历史重点练习。
  其余 1-explore-keep 比例分给当前低价值重点。配置对象只保存参数，不持有运行统计。
  """
  gamma: float = 0.99
  """测量回报的折扣因子；理论上界为 max_step_reward/(1-gamma)。"""
  horizon: int = 20
  """每 horizon 个物理步更新一次，窗口末端使用桶价值补足尚未观测的未来回报。"""
  memory: int = 256
  """旧桶样本的有效计数上限；未达上限时按实际样本数平均，之后限制旧样本权重。"""
  steepness: float = 25.0
  """低回报重点分布的软最大增益，作用于归一化桶价值 V/v_max。"""
  explore: float = 0.5
  """均匀分配给全部时间桶的重置概率比例。"""
  keep: float = 0.1
  """按历史重点概率平方累计形成的保留练习比例，使已重点练习的桶继续得到练习。"""


class ReturnSampler:
  """把近期物理轨迹折扣回报转换为参考时间桶的重置概率。

  每个实例持有 value/count/effort 三个 [K] 桶统计，以及 [E,H] 的帧、奖励和结束缓存；
  K=桶数，E=物理环境数，H=回报估计窗口。value 是当前桶价值，count 是有效样本量，
  effort 累计过去重点概率的平方，用于长期 keep 项，不等于访问次数。

  record 每次接收 E 个环境的一步结果，满窗口才更新；返回概率只改变后续物理环境
  重置起点，不影响本次已执行动作，也不是 ReplayBuffer 的窗口采样概率。
  终止、片段结束、普通超时分开存储，因为三者的末端价值不同。
  """
  # 函数作用：检查关键配置并分配持久统计与短窗口缓存。
  def __init__(
    self,
    cfg: ReturnSamplerCfg,
    bin_count: int,
    frame_count: int,
    num_envs: int,
    max_step_reward: float,
    device: str = "cpu",
  ) -> None:
    """检查关键配置并分配持久统计与短窗口缓存。

    cfg 提供折扣、窗口和混合权重；bin_count=K，frame_count=F 是拼接参考动作的总帧数；
    num_envs=E，max_step_reward 是每控制步奖励上界，device 决定统计张量所在设备。
    返回 None，建立 [K] 的 value/count/effort 及 [E,cfg.horizon] 的逐步缓存。

    先检查 0<gamma<1 和 explore+keep<1，再用 r_max/(1-gamma) 初始化乐观桶价值。
    尚未收到测量的桶保留该上界；effort 使用 float64 累加长期小权重，其他状态按代码
    指定类型分配。_step=0 表示下一次 record 写第零列，不在构造时推进环境或更新分布。
    这里未逐项验证所有配置边界，调用者仍须保证桶数、帧数、窗口为正及比例合法。
    """
    assert 0.0 < cfg.gamma < 1.0, f"gamma must be in (0, 1), got {cfg.gamma}"
    assert cfg.explore + cfg.keep < 1.0, "explore + keep must leave room for focus"
    self.cfg = cfg
    self.bin_count = bin_count
    self.frame_count = frame_count
    # 假设永远获得每步最大奖励，折扣无限和为该上界，用来归一化不同桶价值。
    self.v_max = max_step_reward / (1.0 - cfg.gamma)
    # 未访问桶先采用乐观上界；随后真实回报低的桶会获得更多 focus。
    self.value = torch.full((bin_count,), self.v_max, device=device)
    self.count = torch.zeros(bin_count, device=device)
    # 累计 focus 平方使用双精度，减少长时间累加小权重时的数值损失。
    self.effort = torch.zeros(bin_count, dtype=torch.float64, device=device)
    # 记录缓存按 [物理环境数,估计窗口步数] 分配，帧、奖励和结束类型共享时刻轴。
    shape = (num_envs, cfg.horizon)
    self._frames = torch.zeros(shape, dtype=torch.long, device=device)
    self._rewards = torch.zeros(shape, device=device)
    self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
    self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
    self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
    self._step = 0

  # 函数作用：将合法参考帧号映射到与 motion 命令一致的时间桶。
  def _bins(self, frames: torch.Tensor) -> torch.Tensor:
    """将合法参考帧号映射到与 motion 命令一致的时间桶。

    frames 为整数张量，可为 [E] 或 [E,H]；值应位于 [0,frame_count-1]。
    返回同形状的整数桶索引，公式为 floor(frame*bin_count/frame_count)。
    这是全时间轴的比例映射，不按动作片段重新编号；函数不更新状态，也不在此裁剪输入。
    """
    return frames * self.bin_count // self.frame_count

  # 函数作用：写入所有物理环境的一步观测，满短窗口后返回新的重置分布。
  def record(
    self,
    frames: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    clip_end: torch.Tensor,
    timed_out: torch.Tensor,
  ) -> torch.Tensor | None:
    """写入所有物理环境的一步观测，满短窗口后返回新的重置分布。

    frames/rewards/terminated/clip_end/timed_out 均为 [E]。frames 必须是执行动作前
    参考帧号，rewards 是该步得到的奖励；后三项分别表示失败、动作片段结束及普通超时。
    调用者负责处理同时发生的原因，runner 会从失败中扣除超时，并从普通超时扣除片段结束。

    按 _step 写入 [E,H] 缓存的一列，再递增列号。未满 H 返回 None；满 H 后把 _step
    置零，更新桶价值，并返回 [K] 概率张量。缓存下一窗口会逐列覆盖，value/count/effort
    则跨窗口持续保留。返回值应由调用者发布给 motion 命令，本函数不直接操作环境。
    """
    # 【源码讲解】
    # 记录的是执行动作前的参考帧和执行后获得的奖励及结束原因。
    # 累计到 horizon 步才更新各桶价值并返回新分布，其余调用返回 None。
    col = self._step
    self._frames[:, col] = frames
    self._rewards[:, col] = rewards
    self._terminated[:, col] = terminated
    self._clip_end[:, col] = clip_end
    self._timed_out[:, col] = timed_out
    self._step += 1
    if self._step < self.cfg.horizon:
      return None
    self._step = 0
    self._update()
    return self._distribution()

  # 函数作用：返回缓存内任一种回合边界的布尔掩码 [E,H]。
  def _dones(self) -> torch.Tensor:
    """返回缓存内任一种回合边界的布尔掩码 [E,H]。

    对失败、动作结束和超时做逐元素或运算，仅用于阻断跨回合回报累积与划分局部回合。
    具体末端值仍由 _returns 按原因选择，所以不能据此把三种原因合并存成一个字段。
    读取内部缓存，返回新布尔结果，不改变统计。
    """
    return self._terminated | self._clip_end | self._timed_out

  # 函数作用：逆序计算缓存每一步的折扣回报，返回与 rewards 同形状的 [E,H]。
  def _returns(self) -> torch.Tensor:
    """逆序计算缓存每一步的折扣回报，返回与 rewards 同形状的 [E,H]。

    读取 _frames、奖励、结束原因及当前桶 value，没有外部参数，不修改桶统计。
    对失败末端使用零价值，对动作片段结束使用所有桶价值均值，对超时或窗口最后一步
    使用下一参考帧所在桶的价值；下一帧索引在总时间轴末端裁剪以防越界。

    从最后列开始递推 G_t=r_t+gamma*G_next。遇到结束掩码时，把 G_next 换成该步
    按结束原因选出的 bootstrap，防止新回合奖励串入旧回合。未结束的内点沿用已算出的
    后一列回报。最末列也必须有 bootstrap，因为缓存之外的未来奖励尚未被观测。
    这是统计估值，不通过神经网络拟合，也没有 GAE 中额外的 lambda 参数。
    """
    gamma = self.cfg.gamma
    rewards = self._rewards
    next_frames = (self._frames + 1).clamp(max=self.frame_count - 1)
    # 跌倒后不 bootstrap；动作片段结束取全局平均价值，其余取下一帧所在桶。
    next_value = torch.where(
      self._terminated,
      0.0,
      torch.where(
        self._clip_end, self.value.mean(), self.value[self._bins(next_frames)]
      ),
    )
    dones = self._dones()
    returns = torch.empty_like(rewards)
    next_return = next_value[:, -1]
    for t in reversed(range(self.cfg.horizon)):
      # 回合边界处从对应的末端价值重新开始，避免把下一回合奖励串进来。
      next_return = torch.where(dones[:, t], next_value[:, t], next_return)
      returns[:, t] = rewards[:, t] + gamma * next_return
      next_return = returns[:, t]
    return returns

  # 函数作用：把一个满窗口的轨迹回报折叠到桶价值统计中。
  def _update(self) -> None:
    """把一个满窗口的轨迹回报折叠到桶价值统计中。

    无外部参数，先由 _returns 得 [E,H] 回报，按 done 生成窗口内的局部回合号。
    将环境、回合号和桶号编码为整数 key，unique 得到同组访问及次数；组内先平均回报，
    使一个环境同一回合多次停留同桶只贡献一个样本，再聚合所有环境给各桶的新样本。

    返回 None，更新 [K] 的 count 和 value；effort 在 _distribution 中更新。
    旧 count 先以 memory 封顶再加本次样本数，价值更新等价于旧有效计数和新样本的
    加权平均。没有新样本的桶保留旧值。局部回合只在本缓存窗口内编号；跨窗口的同一
    真实回合不会在这里合并成同一个组。
    """
    # 【源码讲解】
    # 把同一环境、同一回合、同一桶的多次访问先平均成一个样本。
    # 通过 unique 和 scatter_add 聚合，避免停留较久的桶仅因重复帧而过度计数。
    # 历史计数以 memory 封顶，使后续新数据能持续改变估值。
    returns = self._returns()
    dones = self._dones()
    num_envs, horizon = dones.shape
    # 回合编号在 done 的下一帧才增加；结束帧仍归属于它所结束的那一回合。
    episode = torch.cumsum(dones.long(), dim=1) - dones.long()
    envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
    # 把环境、局部回合号、时间桶编码为唯一整数键，以便聚合同回合对同桶的访问。
    key = (envs * horizon + episode) * self.bin_count + self._bins(self._frames)
    # keys 是去重后的环境/回合/桶组合；inverse 把每个 [E,H] 访问映射到组号，visits 是组内次数。
    keys, inverse, visits = torch.unique(
      key.reshape(-1), return_inverse=True, return_counts=True
    )
    samples = torch.zeros(keys.numel(), device=returns.device)
    samples.scatter_add_(0, inverse, returns.reshape(-1))
    # 一个环境同回合多次访问同桶先平均，避免速度慢或停留久的桶被重复帧过度加权。
    samples /= visits
    # key 以桶号作为最低位，取余恢复桶；后续把不同环境/回合产生的样本汇入同一个桶。
    bins = keys % self.bin_count
    new = torch.zeros_like(self.count).scatter_add_(0, bins, torch.ones_like(samples))
    # new 累计各桶新样本数，total 累计新回报和，两者共同决定加权均值更新。
    total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
    # 旧样本有效计数限制为 memory，新样本仍加入，使估值能随当前策略变化。
    self.count = self.count.clamp(max=self.cfg.memory) + new
    self.value = torch.where(
      new > 0, self.value + (total - new * self.value) / self.count, self.value
    )

  # 函数作用：由当前难度和历史重点生成 [K] 重置概率，并累计本轮重点投入。
  def _distribution(self) -> torch.Tensor:
    """由当前难度和历史重点生成 [K] 重置概率，并累计本轮重点投入。

    无显式参数。当前重点 focus=softmax(-steepness*value/v_max)，价值越低概率越高；
    effort 加 focus² 后归一化为历史 keep 分布。平方使长期集中练习的桶留下更强记录，
    不是按实际重置次数累计。均匀项给每桶 explore/K，剩余比例分给 focus 和 keep。

    返回的概率和在合法配置下约为 1。此函数有副作用：每次调用都会增加 effort，
    因此并非可随意重复查询的纯函数；正常由 record 在每个统计窗口末调用一次。
    """
    # 【源码讲解】
    # 低价值桶通过 softmax(-steepness*V/v_max) 获得更多 focus。
    # explore 保留均匀覆盖，keep 按历史 focus 平方累计维持已重点练习的片段。
    cfg = self.cfg
    # 回报越低，负号使对应 focus 权重越高；仍保留均匀探索和历史练习比例。
    focus = torch.softmax(-cfg.steepness * self.value / self.v_max, dim=0)
    self.effort += focus.double().square()
    keep = (self.effort / self.effort.sum()).float()
    explore = cfg.explore / self.bin_count
    return explore + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep
