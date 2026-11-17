
# 【文件作用】按环境存储的时序经验池
# 保存物理仿真状态、动作、参考时刻和结束标志，提供模型训练窗口及策略历史起点采样。
# 【阅读顺序】FieldSpec 定义字段 → insert 环形写入 → sample 连续窗口 → sample_causal 因果历史。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# dataclass 为字段规格生成初始化和比较方法，frozen=True 防止规格被意外改写。
from dataclasses import dataclass

# PyTorch 张量、随机采样、设备和自动微分工具；计算沿最后一维组织特征，前导轴保留批次/时间。
import torch
# TensorDict 携带共享批次形状和多个张量字段；回放与策略输入可统一索引、切片和堆叠。
from tensordict import TensorDict


@dataclass(frozen=True)
class FieldSpec:
  """一类回放字段的不可变规格，供 ReplayBuffer 在构造时分配存储。

  dim=0 表示标量字段，如帧号和结束标志；dim>0 表示最后一轴的向量宽度。
  dtype 决定存储类型，默认 float32。frozen=True 防止初始化后修改规格对象，
  但不意味着整个经验池不可变。调用 shape(E,T) 得到该字段的实际存储形状。
  """
  dim: int = 0  # 0 表示标量、没有特征尾轴；正数表示向量宽度。
  dtype: torch.dtype = torch.float32

  # 函数作用：根据环境数和时间容量计算本字段的完整存储形状。
  def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
    """根据环境数和时间容量计算本字段的完整存储形状。

    参数 num_envs=E、buffer_size=T；返回标量字段的 (E,T)，或向量字段的 (E,T,dim)。
    只返回尺寸元组，不申请张量；真正的分配发生在 ReplayBuffer.__init__。
    所有字段共享前两条批次轴，因此同一次索引可同时取出一段完整转移信息。
    """
    # 【源码讲解】
    # 标量 dim=0 的形状是 [E,T]；向量字段是 [E,T,D]。
    # 这里 E 是并行环境数，T 是每个环境的缓存容量；各字段共享前两维。
    return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
  """按物理环境分层保存时间序列的环形回放池。

  每个环境都有相同长度的时间环，insert 一次同时写入 E 个环境的一步数据。
  字段内容由 schema 指定，必须含 terminated 和 truncated，以识别回合边界。
  存储批次形状为 [E,T]，状态等向量字段在末尾附加自己的特征轴。

  sample 给世界模型抽不跨回合的连续窗口；sample_causal 给长历史策略抽以当前帧
  结束的历史并补齐回合起点。二者的目标不同：前者不能凭空填监督序列，后者允许
  重复起始帧预热观察。num_transitions 仅表示每环境已填充时间槽数，不是 E 倍总量。
  """

  # 函数作用：分配固定容量的多字段张量，并初始化环形写指针。
  def __init__(
    self,
    schema: dict[str, FieldSpec],
    buffer_size: int,
    num_envs: int,
    device: str = "cpu",
  ):
    """分配固定容量的多字段张量，并初始化环形写指针。

    参数 schema 为字段名到 FieldSpec 的映射；buffer_size=T 为每环境容量；
    num_envs=E 为并行环境数；device 为存储设备。构造时每字段一次性分配全零张量，
    统一包装为 batch_size=[E,T] 的 TensorDict。

    返回 None；_step=0 指下一写入槽，num_transitions=0 表示尚无有效记录。
    若缺少结束字段则立即报错。全零预分配不等于已有样本，采样仍受有效填充量约束。
    """
    # 【源码讲解】
    # 为每个字段一次性分配张量，减少训练循环中反复申请显存的开销。
    # _step 是下一次写入槽位，num_transitions 是每环境已填充的时间槽数，并非 E 倍总量。
    # 结束标志必须存在，因为连续窗口需要检查回合边界。
    if not {"terminated", "truncated"} <= schema.keys():
      raise ValueError("schema needs 'terminated' and 'truncated'")
    self.num_transitions = 0
    self.buffer_size = buffer_size
    self.num_envs = num_envs
    self.device = device
    self._step = 0
    self._storage = TensorDict(
      {
        name: torch.zeros(
          spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=device
        )
        for name, spec in schema.items()
      },
      batch_size=[num_envs, buffer_size],
    )

  # 函数作用：把一个物理控制步的整批记录写入各环境对应时间槽。
  def insert(self, td: TensorDict) -> None:
    """把一个物理控制步的整批记录写入各环境对应时间槽。

    参数 td 的 batch_size 必须为 [E]，键集合必须与 schema 完全一致。
    例如 state:[E,S]、action:[E,A]、terminated:[E]；函数返回 None。

    先验证接口，再写 _storage[:,_step]；写指针模 T 前进，填充量最多增长到 T。
    满池后最早数据被覆盖。这里存储的是调用者提供的转移字段，并不推断 next_state；
    后续相邻 state 能否作为标签，取决于 sample 对结束标志的检查。
    """
    # 【源码讲解】
    # td 的 batch 必须是 [E]，一次写入所有环境在一个控制步的记录。
    # 字段集合必须与 schema 完全一致；相邻时间槽中的 state 提供下一帧监督。
    # 环满后覆盖最旧记录，因此采样要先把槽位映射为时间顺序。
    if td.batch_size != torch.Size([self.num_envs]):
      raise ValueError(
        f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}"
      )
    if set(td.keys()) != set(self._storage.keys()):
      raise ValueError("schema mismatch")
    # 每个环境各有一条时间环；写满后回到开头覆盖最早的槽位。
    self._storage[:, self._step] = td
    self._step = (self._step + 1) % self.buffer_size
    self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

  # 函数作用：抽取以随机当前帧为末端的同回合历史，并返回补齐掩码。
  def sample_causal(
    self, sequence_length: int, batch_size: int
  ) -> tuple[TensorDict, torch.Tensor]:
    """抽取以随机当前帧为末端的同回合历史，并返回补齐掩码。

    参数 sequence_length=L 为历史长度，batch_size=B 为样本数。返回 window 的批次
    形状 [B,L]，各字段保留特征尾轴；padded:[B,L] 中 True 表示该槽是重复填充。
    当前帧先从已填记录中均匀抽取，故改变 L 不会改变当前帧的分布。

    从当前帧倒查：遇到更早的结束标志或缓存最旧边界就停止，之前槽位重复最早
    可达帧。当前帧即使记录 done 仍可抽到，因为 done 表示它执行动作后的结果。
    L 超容量或池为空会报错。调用方利用掩码让重复历史共用同一次传感器噪声。
    """
    # 【源码讲解】
    # 先抽当前帧，再向过去寻找连续的同回合历史，当前帧始终有效。
    # 不足 L 帧的部分重复最早有效帧，返回窗口 [B,L,...] 和填充掩码 [B,L]。
    # 当前帧的抽样不依赖历史长度，避免改变历史长度时也改变起点分布。
    # L=序列长度，B=抽样数量，E=并行环境数，T=每环境缓冲容量。
    L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
    dev, n = self.device, self.num_transitions
    if L > T:
      raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
    if n < 1:
      raise RuntimeError("sample_causal on empty buffer")
    # 缓存未满时最旧槽为 0；写满后下一写入槽 _step 正是最旧记录。
    # base + age 再模 T，把线性的时间顺序映射回物理存储槽位。
    base = self._step if n == T else 0
    # 先均匀抽环境，再抽当前帧年龄；两者组合确定每个样本的末帧。
    env_idx = torch.randint(E, (B,), device=dev)
    current_age = torch.randint(n, (B,), device=dev)
    ages = current_age[:, None] + (
      torch.arange(L, device=dev) - (L - 1)
    )  # (B, L), newest last
    ended = (self._storage["terminated"] | self._storage["truncated"])[
      env_idx[:, None], (base + ages.clamp(min=0)) % T
    ]
    # Reachable = contiguous in-episode suffix ending at the (always-valid) newest
    # frame; a stored done at an earlier slot ends the prior episode there.
    valid = (ages >= 0) & ~ended.bool()
    # 当前帧即使带 done 仍可作为起点；只有更早的结束标志会阻断其前驱历史。
    valid[:, -1] = True
    # 从当前帧向过去找连续有效部分；跨越回合起点的槽位会用最早有效帧补齐。
    reachable = torch.cummin(valid.flip(-1).int(), dim=-1).values.flip(-1).bool()
    # 把不可达位置临时替换为当前帧，再取最小值，得到最早可用历史帧。
    first_age = torch.where(reachable, ages, current_age[:, None]).amin(
      dim=-1, keepdim=True
    )
    slot = (base + torch.where(reachable, ages, first_age)) % T
    return self._storage[env_idx[:, None], slot], ~reachable

  # 函数作用：抽取真实连续窗口，供多步世界模型监督或单帧起点采样。
  def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
    """抽取真实连续窗口，供多步世界模型监督或单帧起点采样。

    参数 sequence_length=L、batch_size=B；返回批次为 [B,L] 的 TensorDict。
    先把物理槽位转换为线性 age 顺序，用结束标志前缀和排除窗口前 L-1 帧存在
    结束的起点；最后一帧允许结束，因为不需要它之后的下一帧标签。

    抽样先均匀选择存在合法窗口的环境，再在该环境合法起点内均匀选择，允许重复。
    因此它不是把所有环境的全部合法窗口合在一起全局等概率抽取。
    L 超容量或没有合法窗口会报错；仅填充到 L 帧也不保证短回合中存在合法序列。
    """
    # L=序列长度，B=抽样数量，E=并行环境数，T=每环境缓冲容量。
    # 【源码讲解】
    # 先把环形存储转换为从最旧到最新的 age 索引。
    # 结束标志前缀和之差判断窗口前 L-1 帧是否无结束，末帧允许结束。
    # 先均匀抽有合法窗口的环境，再抽该环境的合法起点，返回 [B,L,...]。
    # 这能避免把自动重置后的新状态误认为上个回合动作造成的结果。
    L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
    dev, n = self.device, self.num_transitions
    if L > T:
      raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
    # Index by age (0 = oldest filled, n-1 = newest); slot = (base + age) % T.
    # In age-space windows are linear, so wrap and warm-up collapse into one predicate.
    # 缓存未满时最旧槽为 0；写满后下一写入槽 _step 正是最旧记录。
    # base + age 再模 T，把线性的时间顺序映射回物理存储槽位。
    base = self._step if n == T else 0
    ages = torch.arange(T, device=dev)
    ended = (self._storage["terminated"] | self._storage["truncated"])[
      :, (base + ages) % T
    ].long()
    # Interior [a, a+L-2] is done-free iff c[a+L-1] == c[a] (c is cumsum with leading 0).
    c = torch.cat(
      [ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1
    )
    # 前缀和之差为零，表示窗口内部没有 done；最后一帧可以是 done。
    valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
    # 合法起点掩码前缀和，把第几个合法窗口映射为真实 age 起点。
    csum = valid.cumsum(dim=1, dtype=torch.long)
    total = csum[:, -1]
    if not (total > 0).any():
      raise RuntimeError(f"no valid sequences L={L} n={n}")
    # 先在有合法窗口的环境中均匀抽环境，再在该环境的合法起点中抽样。
    env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
    u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
    # u 是合法窗口的零起始排名；right=True 跳过前缀和的相等平台，找到对应起点。
    age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
    storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
    return self._storage[env_idx[:, None], storage_idx]
