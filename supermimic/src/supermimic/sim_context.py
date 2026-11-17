"""世界模型的逐环境物理条件读取、列布局与归一化。"""

# 【文件作用】仿真随机化上下文
# 从各环境提取摩擦、质心和编码器偏置等物理条件，按配置范围归一化，作为动力学模型的条件输入。
# 【阅读顺序】各 Spec 绑定仿真实体并读取值 → SimContext 按固定列序拼接、归一化及反归一化。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟类型注解求值，使本文件可用稍后定义的类型并减少循环导入；不改变张量计算。
from __future__ import annotations


# 正则匹配环境事件的碰撞几何体名称，用于选定足底摩擦读取对象。
import re
# ABC/abstractmethod 声明必须由具体状态表示或条件规格实现的接口，避免误用基类。
from abc import ABC, abstractmethod
# TYPE_CHECKING 内的导入只服务类型检查，避免运行时循环依赖；Literal 限定观察模式取值。
from typing import TYPE_CHECKING

# PyTorch 张量、随机采样、设备和自动微分工具；计算沿最后一维组织特征，前导轴保留批次/时间。
import torch
# Module、ModuleDict、Sequential 负责注册网络与缓冲；注册后设备迁移和 state_dict 才能统一管理。
import torch.nn as nn

if TYPE_CHECKING:
  # ManagerBasedRlEnv 是未包装物理环境类型，用于查询机器人、命令和随机化事件；这里只作类型说明。
  from mjlab.envs import ManagerBasedRlEnv


_DEGENERATE_HALF_WIDTH = 1e-3
"""半宽不大于该阈值的条件视为近常量，归一化尺度取1。

单位量级 float32 的舍入误差约1e-7，除以很小的半宽会把误差放大。
以1e-3为界可抑制近常量参数的噪声，而默认摩擦、质心和编码器偏置范围均超过此界。
"""


# 函数作用：从随机化范围得到中心和归一化尺度。
def _mean_scale(lo: float, hi: float) -> tuple[float, float]:
  """从随机化范围得到中心和归一化尺度。

  参数 lo/hi 是一列物理参数下上界，要求 hi>=lo。返回 (中心,尺度)，
  尺度通常为半宽；半宽不超过精度阈值时改为 1，避免把近常量浮点误差放大。
  它不采样随机值，配置范围只用于为已有随机化物理量建立固定单位变换。
  """
  assert hi >= lo, f"empty range: lo={lo}, hi={hi}"
  half = (hi - lo) / 2.0
  return (lo + hi) / 2.0, half if half > _DEGENERATE_HALF_WIDTH else 1.0


_ENTITY_IDS_ATTR = {"body": "body_ids", "site": "site_ids", "geom": "geom_ids"}


class SimContextSpec(ABC):
  """一种物理随机化条件的读取协议，不保存神经网络权重。

  name 对应环境事件名称，dim 是输出特征宽度。bind 在实际环境上解析资产、
  全局实体索引和随机化范围，read 在采集时读取当前各环境的原始值 [E,dim]。
  具体规格负责物理意义，SimContext 统一负责列拼接和归一化。

  抽象方法要求子类实现；调用顺序必须先 bind 后 read，不能用资产局部索引直接
  访问 MuJoCo 的全局模型数组。
  """
  name: str
  dim: int

  # 函数作用：约定具体规格如何从环境绑定实体和列范围。
  @abstractmethod
  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    """约定具体规格如何从环境绑定实体和列范围。

    参数 env 是未包装物理环境；返回长度为 dim 的 (lo,hi) 范围列表。
    子类可在此更新 dim 和缓存全局索引。基类无实现，绑定阶段不执行仿真或随机化。
    """
    ...

  # 函数作用：约定从已绑定的物理环境读取逐环境原始值。
  @abstractmethod
  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """约定从已绑定的物理环境读取逐环境原始值。

    参数 env 需与 bind 的资产布局一致；返回 [num_envs,dim] 张量。
    这里应返回物理量，不做统一中心尺度变换，后者由 SimContext.read 完成。
    """
    ...

  # 函数作用：将资产内实体编号转换为模型数组使用的全局实体编号。
  @staticmethod
  def _global_entity_id(asset, entity_type: str, local_idx: int) -> int:
    """将资产内实体编号转换为模型数组使用的全局实体编号。

    asset 是场景机器人，entity_type 选 body/site/geom，local_idx 是资产内编号。
    返回 Python int，由 indexing 中对应 id 表查询；没有随机化副作用。
    若误把局部编号直接用于模型数组，可能读到相邻资产的未随机化参数而不报尺寸错误。
    """
    # 【源码讲解】
    # 资产内索引与 MuJoCo 全局索引不一定相等。
    # 随机化写入全局模型数组，读取时必须做相同映射，否则可能读到邻近实体。
    return int(getattr(asset.indexing, _ENTITY_IDS_ATTR[entity_type])[local_idx].item())


class FootFrictionSpec(SimContextSpec):
  """足底切向摩擦条件，向上下文贡献一列。

  绑定 foot_friction 事件指定的碰撞几何体和范围。该事件在同一环境内共享随机值，
  因此取第一个匹配几何体足以代表足部条件；不同环境仍各自读取摩擦系数。
  模型使用的是 geom_friction 的第 0 分量，不是把全部摩擦参数都拼入上下文。
  """

  name = "foot_friction"
  dim = 1

  # 函数作用：把足底几何体索引初始化为未绑定状态。
  def __init__(self) -> None:
    """把足底几何体索引初始化为未绑定状态。

    无参数，返回 None。_geom_idx=None 让 read 能在错误调用顺序时明确报错；
    实际全局索引由 bind 的名称匹配与资产映射确定。
    """
    self._geom_idx: int | None = None

  # 函数作用：按足底摩擦事件选定几何体，缓存全局编号并返回一列范围。
  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    """按足底摩擦事件选定几何体，缓存全局编号并返回一列范围。

    参数 env 提供 events、scene 和资产几何体名称。geom_names 可为正则字符串或
    显式名称列表；匹配后取首项转全局编号。返回 [(lo,hi)]，供统一统计构造。
    缺事件、空选择或无匹配均报错；函数不会改变实际摩擦，只查明应该读取何处。
    """
    # 【源码讲解】
    # 从 foot_friction 事件解析随机化范围和匹配的碰撞几何体。
    # 同环境内匹配几何体共享随机值，因此选择第一个匹配即可，并转换为全局索引。
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"FootFrictionSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    ranges = params["ranges"]

    asset_cfg = params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    # geom_names 既可能是正则表达式，也可能是明确名称列表，下面分别解析两种情况。
    pattern = asset_cfg.geom_names
    if isinstance(pattern, str):
      pat = re.compile(pattern)
      matches = [i for i, name in enumerate(asset.geom_names) if pat.match(name)]
    elif isinstance(pattern, (list, tuple)) and pattern:
      name_set = set(pattern)
      matches = [i for i, name in enumerate(asset.geom_names) if name in name_set]
    else:
      raise ValueError(f"FootFrictionSpec: '{self.name}' has empty geom_names")
    # 没有匹配实体时立即报错，防止摩擦上下文读取错误碰撞体而静默污染监督数据。
    if not matches:
      raise ValueError(
        f"FootFrictionSpec: no geoms in '{asset_cfg.name}' match {pattern!r}"
      )
    self._geom_idx = self._global_entity_id(asset, "geom", matches[0])
    return [ranges[0] if isinstance(ranges, dict) else ranges]

  # 函数作用：读取已绑定足底几何体的逐环境切向摩擦。
  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取已绑定足底几何体的逐环境切向摩擦。

    参数 env 为物理环境；返回 [E,1]，保留环境与特征轴以便后续拼接。
    若尚未 bind 则报错。返回当前模型中的值，不重新抽样、不归一化。
    """
    # 【源码讲解】
    # 返回 [E,1] 的切向摩擦系数，仅取摩擦参数最后一维的第 0 项。
    # 读取必须晚于 bind，因为 bind 才确定几何体索引。
    if self._geom_idx is None:
      raise RuntimeError("FootFrictionSpec.read() called before bind()")
    return env.sim.model.geom_friction[:, self._geom_idx : self._geom_idx + 1, 0]


class CoMOffsetSpec(SimContextSpec):
  """根部惯性位置条件，对应 base_com 事件的三个坐标。

  该规格要求事件只选一个刚体，将 xyz 三列加入上下文。
  bind 缓存该刚体的全局编号，read 读取模型 body_ipos；这里得到的是当前惯性位置
  参数，具体随机化的基准与扰动方式由环境事件定义，不在此重新执行。
  """
  name = "base_com"
  dim = 3

  # 函数作用：将质心条件的刚体编号设为未绑定。
  def __init__(self) -> None:
    """将质心条件的刚体编号设为未绑定。

    无参数，返回 None。dim 固定为 3，_body_id 在绑定实际资产后才确定。
    """
    self._body_id: int | None = None

  # 函数作用：解析 base_com 事件选择的唯一刚体，并取得 xyz 随机化范围。
  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    """解析 base_com 事件选择的唯一刚体，并取得 xyz 随机化范围。

    参数 env；返回三个 (lo,hi) 范围，副作用为写入 _body_id 全局编号。
    若事件不存在或选择多于/少于一个刚体则报错。名称先映射资产内索引，再映射
    MuJoCo 全局索引，不能直接按名称列表位置读全局模型。
    """
    # 【源码讲解】
    # 质心随机化要求恰好一个 body；先由名称找资产内 body，再转全局 id。
    # 返回三个坐标的范围，供上下文构造统一尺度。
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"CoMOffsetSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    ranges = params["ranges"]
    asset_cfg = params["asset_cfg"]
    if len(asset_cfg.body_names) != 1:
      raise ValueError(
        f"CoMOffsetSpec: expected exactly 1 body_name, got {asset_cfg.body_names!r}"
      )
    body_name = asset_cfg.body_names[0]
    asset = env.scene[asset_cfg.name]
    self._body_id = self._global_entity_id(
      asset, "body", asset.body_names.index(body_name)
    )
    return [ranges[i] for i in range(self.dim)]

  # 函数作用：读取已绑定刚体的逐环境惯性位置 xyz。
  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取已绑定刚体的逐环境惯性位置 xyz。

    参数 env；返回 [E,3] 原始模型参数。尚未 bind 时抛错；不会改变质心或更新范围。
    归一化由外层 SimContext 使用 bind 阶段保存的范围执行。
    """
    if self._body_id is None:
      raise RuntimeError("CoMOffsetSpec.read() called before bind()")
    # 读取 MuJoCo 模型中该刚体当前的惯性位置参数，保留所有并行环境及 xyz 轴。
    return env.sim.model.body_ipos[:, self._body_id, :]


class EncoderBiasSpec(SimContextSpec):
  """关节编码器偏置条件，每个关节贡献一列。

  与固定宽度摩擦/质心不同，dim 要在绑定机器人后由 num_joints 决定。
  读取机器人数据中的 encoder_bias，而非 MuJoCo 全局几何体数组。
  这些条件也供最大坐标表示合成带偏置的关节观察时反归一化使用。
  """
  name = "encoder_bias"
  dim = 0  # 占位；bind 读取资产实际关节数后覆盖。

  # 函数作用：将偏置来源资产标记为未绑定。
  def __init__(self) -> None:
    """将偏置来源资产标记为未绑定。

    无参数，返回 None；_asset_name 稍后从事件 asset_cfg 获取。
    此时类级 dim=0 只是占位，不能在 bind 前用它确定最终网络输入宽度。
    """
    self._asset_name: str | None = None

  # 函数作用：绑定编码器偏置资产，并按真实关节数扩展范围。
  def bind(self, env: ManagerBasedRlEnv) -> list[tuple[float, float]]:
    """绑定编码器偏置资产，并按真实关节数扩展范围。

    参数 env；返回每个关节一份 bias_range 列表。缓存资产名称并更新 dim，
    不生成新的噪声。事件未启用会报错；同范围不表示各关节的实际随机值相同。
    """
    # 【源码讲解】
    # 编码器偏置宽度等于机器人关节数，在绑定实际资产后才能确定。
    # 每个关节沿用同一个配置范围，读取时保留逐环境、逐关节偏置。
    event_cfg = env.cfg.events.get(self.name)
    if event_cfg is None:
      raise RuntimeError(f"EncoderBiasSpec: no event '{self.name}' on env.cfg.events")
    params = event_cfg.params
    asset_cfg = params["asset_cfg"]
    self._asset_name = asset_cfg.name
    self.dim = env.scene[self._asset_name].num_joints
    # 每个关节都使用同一偏置范围，但实际 read 返回逐环境、逐关节的随机化值。
    return [params["bias_range"]] * self.dim

  # 函数作用：读取绑定机器人数据中的逐环境逐关节偏置。
  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取绑定机器人数据中的逐环境逐关节偏置。

    参数 env；返回 [E,N_joint]。调用前必须 bind 以确定资产名称。
    返回原始偏置值，统一归一化和列布局仍由 SimContext 管理。
    """
    if self._asset_name is None:
      raise RuntimeError("EncoderBiasSpec.read() called before bind()")
    return env.scene[self._asset_name].data.encoder_bias


class SimContext(nn.Module):
  """把多类物理条件拼成固定列序的模型输入，并提供单位换算。

  specs 是已绑定的有序规格；ranges 是按同序展平的逐列随机化范围。
  本类继承 nn.Module 以将 mean/scale 注册为 buffer，支持设备迁移与检查点保存，
  但不靠梯度学习这些尺度。实际每环境值在 read 时从物理模型取得。

  from_env 是通常的构建入口；read 返回 [E,C] 归一化上下文；find 查询某项列切片，
  denormalize 恢复物理单位。列顺序是动力学与经验池的接口，修改后不能直接混用旧权重。
  """
  # 函数作用：保存已绑定规格，并从列范围生成固定中心和尺度。
  def __init__(self, specs: list[SimContextSpec], ranges: list[tuple[float, float]]):
    """保存已绑定规格，并从列范围生成固定中心和尺度。

    参数 specs 按最终特征列顺序排列，ranges 每列一组上下界；返回 None。
    每列调用 _mean_scale，再注册 mean/scale:[C] buffer；不会读取当前环境数据。
    因此该构造函数通常由 from_env 在执行各规格 bind 后调用。
    """
    # 【源码讲解】
    # 按每列配置范围生成 mean 和 scale，并保存为模型 buffer。
    # 上下文不是在线经验均值归一化，而是依据随机化范围进行固定尺度变换。
    super().__init__()
    self.specs = list(specs)
    # 逐列从随机化范围确定中心和尺度，固定范围归一化不依赖当前 batch 的样本统计。
    stats = [_mean_scale(lo, hi) for lo, hi in ranges]
    self.register_buffer("mean", torch.tensor([m for m, _ in stats]))
    self.register_buffer("scale", torch.tensor([s for _, s in stats]))

  # 函数作用：发现环境启用的随机化条件并建立可读取的上下文对象。
  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv) -> SimContext:
    """发现环境启用的随机化条件并建立可读取的上下文对象。

    参数 env 是未包装环境。按候选规格固定顺序保留已启用事件，逐项 bind 解析
    宽度、索引与范围；构造 SimContext 并移动到 env.device，返回该实例。

    没有任何支持的条件时抛错，而不是返回宽度零的上下文。新增事件不会自动被读取，
    还需实现规格并加入候选序列；改变启用项可能改变模型输入维数和列语义。
    """
    # 【源码讲解】
    # 只启用环境事件中存在的规格，候选顺序就是最终上下文列顺序。
    # bind 解析索引和宽度，展平各列范围后构造对象并迁移到环境设备。
    # 列顺序改变会改变动力学输入语义，恢复旧模型时也需保持一致。
    candidates: tuple[type[SimContextSpec], ...] = (
      FootFrictionSpec,
      CoMOffsetSpec,
      EncoderBiasSpec,
    )
    # 候选顺序是上下文特征顺序；关闭某事件会改变输入宽度，旧模型不能无条件复用。
    specs = [c() for c in candidates if c.name in env.cfg.events]
    if not specs:
      raise RuntimeError(
        "No sim_ctx specs derived from env.cfg.events. "
        "Enable at least one DR event (foot_friction, base_com, encoder_bias) "
        "in the task cfg."
      )
    return cls(specs, [r for s in specs for r in s.bind(env)]).to(env.device)

  # 函数作用：返回已绑定各规格宽度之和 C。
  @property
  def output_dim(self) -> int:
    """返回已绑定各规格宽度之和 C。

    这是只读属性，使用方式为 context.output_dim。依赖规格 bind 后确定的 dim；
    不读取环境、不分配逐环境样本，用于建立网络和经验池字段宽度。
    """
    return sum(spec.dim for spec in self.specs)

  # 函数作用：读取并归一化所有环境的物理条件，返回 [E,C]。
  def read(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取并归一化所有环境的物理条件，返回 [E,C]。

    参数 env 与绑定时的实体布局一致。各 spec.read 返回 [E,D_i] 原始值，沿最后轴
    按固定序列拼接，再用 [C] 的 mean/scale 广播完成 (raw-mean)/scale。

    不改变物理参数或经验统计。调用者将结果与当前状态一起入回放，世界模型和
    想象环境据此知道样本所属物理条件，而非每个想象步重新随机化。
    """
    # 【源码讲解】
    # 各规格返回 [E,D_i] 原始值，沿特征轴拼接为 [E,C]。
    # 随后逐列执行 (raw-mean)/scale，结果存入经验池并作为动力学条件。
    # 各规格原始输出 [E,D_i] 沿最后轴拼接为 [E,C]；mean/scale 的 [C] 广播到所有环境。
    return (torch.cat([s.read(env) for s in self.specs], -1) - self.mean) / self.scale

  # 函数作用：按规格名称查询归一化上下文中的列切片。
  def find(self, name: str) -> slice | None:
    """按规格名称查询归一化上下文中的列切片。

    参数 name 对应规格/事件名称；返回 slice(start,end)，未找到返回 None。
    通过累计前面规格宽度定位，不硬编码固定列号。查询无副作用，适用于可选条件。
    """
    # 【源码讲解】
    # 累计前面规格的宽度，返回指定名称占用的列切片；不存在返回 None。
    # 消费者应使用这个查询接口，避免把某项的位置硬编码为固定下标。
    offset = 0
    for spec in self.specs:
      if spec.name == name:
        return slice(offset, offset + spec.dim)
      offset += spec.dim
    return None

  # 函数作用：将指定上下文列恢复为原始物理单位。
  def denormalize(self, sim_ctx: torch.Tensor, columns: slice) -> torch.Tensor:
    """将指定上下文列恢复为原始物理单位。

    参数 sim_ctx:[...,C]，columns 通常由 find 返回；结果形状为 [...,所选列数]。
    执行 x*scale+mean，不修改输入或统计；选择的列必须对应同一上下文实例的布局。
    最大坐标传感器合成用它恢复编码器偏置后加到相对关节位置。
    """
    # 【源码讲解】
    # 只对给定列恢复 raw=normalized*scale+mean。
    # 用于需要实际物理量的观察合成，例如恢复编码器偏置。
    return sim_ctx[..., columns] * self.scale[columns] + self.mean[columns]
