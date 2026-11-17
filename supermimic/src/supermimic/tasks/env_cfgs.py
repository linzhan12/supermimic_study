"""构造 SuperMimic 的物理仿真环境配置，以及环境观察的读取函数。

本文件不执行训练，也不创建 Dynamics。它在 mjlab 的 G1 动作跟踪配置上定义
actor 传感器、full_state 监督真值和 goal 参考窗口三组观察。广义坐标和最大坐标
共用动作、奖励与随机化设置，仅完整状态布局不同。观察字段的顺序是模型切片的协议，
必须与 supermimic/src/supermimic/representations.py 保持一致。

阅读顺序：数据读取工厂 → 参考窗口相对变换 → 观察组 → 两种配置工厂。
E 表示物理环境数，J 表示关节数，K 表示被跟踪刚体数，N 表示目标窗口帧数。
"""

# 【文件作用】物理仿真环境配置
# 在 mjlab 的 G1 跟踪任务上定义传感器观察、完整状态和参考目标，并调整随机化、奖励与终止项。
# 【阅读顺序】观察读取函数 → 观察组构造 → gencoord_env_cfg → maxcoord_env_cfg。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟类型注解求值，使 TYPE_CHECKING 内的类型仅用于静态检查而无需运行时导入。
from __future__ import annotations

# cast 只向类型检查器说明对象类型，运行时原样返回，用于 motion 命令接口提示。
from typing import cast

# mjlab 的观察读取函数集合，提供重力方向、IMU角速度、关节相对状态和上一动作等传感器项。
import mjlab.envs.mdp as mdp
# PyTorch 提供张量索引、批次变换、自动微分和编译；本文件的数据形状约定由这些运算实现。
import torch
# 物理环境及其配置类型；前者供观察回调读取场景，后者是环境配置工厂的返回对象。
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
# 观察项描述单个读取回调，观察组决定拼接顺序、是否拼接和是否启用噪声。
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
# 以名称选择场景资产/身体等实体；默认 robot 选择用于多个观察回调。
from mjlab.managers.scene_entity_config import SceneEntityCfg
# 复用 mjlab 的 G1 平地动作跟踪基础任务，本文件在其上覆盖观察、随机化和部分评分设置。
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
# 参考动作命令接口，维护参考时刻、加载的动作和被跟踪机器人刚体数据。
from mjlab.tasks.tracking.mdp.commands import MotionCommand
# 均匀噪声配置；这里只声明观察项噪声，是否执行还由观察组的开关控制。
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# 来自 supermimic/src/supermimic/representations.py：共享传感器噪声规范及目标窗口、相对根变换工具，避免物理与想象路径各自定义布局。
from supermimic.representations import (
  PROPRIO_NOISE_HALF_WIDTHS,
  assemble_anchor_window,
  motion_anchor_window,
  motion_goal_window,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# 函数作用：创建读取机器人某项资产数据的观察回调。
def _make_asset_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """创建读取机器人某项资产数据的观察回调。

  参数 attr_name 是 asset.data 上的属性名；subtract_origins 决定是否移除平铺环境原点。
  返回闭包 fn，而不是立即读取数据。工厂调用时环境还可能没有实例化，因此延迟到
  观察管理器实际调用 fn 时才访问 scene。当前根位置用此机制去除环境平移，
  根方向、速度和关节状态则直接读取。返回值的形状由指定属性决定。
  """
  # 【源码讲解】
  # 工厂返回读取指定资产属性的观察函数，attr_name 被闭包保存。
  # 位置需要减去并行环境原点，避免平铺场景的位置偏移进入训练状态。

  def fn(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
  ) -> torch.Tensor:
    """在当前物理环境读取工厂指定的机器人属性。

    参数 env 提供场景与环境原点；asset_cfg 默认选择 robot，可按观察项配置替换。
    返回对应属性张量，例如根位置 [E,3]、四元数 [E,4] 或关节值 [E,J]。
    若工厂要求 subtract_origins，则执行位置减原点；调用方必须只对合适的位置属性启用。
    只读取数据，不推进环境、不修改机器人状态。
    """
    asset = env.scene[asset_cfg.name]
    # getattr 根据工厂捕获的属性名读取数据，同一逻辑可复用到根位置、速度和关节状态。
    val = getattr(asset.data, attr_name)
    return val - env.scene.env_origins if subtract_origins else val

  return fn


root_pos = _make_asset_obs_fn("root_link_pos_w", subtract_origins=True)
root_ori = _make_asset_obs_fn("root_link_quat_w")
root_lin_vel = _make_asset_obs_fn("root_link_lin_vel_w")
root_ang_vel = _make_asset_obs_fn("root_link_ang_vel_w")
joint_pos = _make_asset_obs_fn("joint_pos")
joint_vel = _make_asset_obs_fn("joint_vel")


# ---------------------------------------------------------------------------
# 最大坐标的逐刚体观察项
# ---------------------------------------------------------------------------


# 函数作用：创建读取 motion 命令所缓存机器人刚体数据的观察回调。
def _make_body_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """创建读取 motion 命令所缓存机器人刚体数据的观察回调。

  attr_name 指向 robot_body_pos_w 等属性，subtract_origins 用于刚体位置原点修正。
  返回闭包 fn，运行时才取得当前环境的 MotionCommand。与资产读取工厂不同，
  此处处理被跟踪刚体序列，最终把 [E,K,单体宽度] 展平为 [E,K×单体宽度]。
  刚体顺序来自 motion 命令，最大坐标表示和运动学计算必须沿用相同顺序。
  """
  # 【源码讲解】
  # 从 motion 命令读取被跟踪刚体数据，再展开为 [E,body_count*单体宽度]。
  # 位置减环境原点时增加刚体轴，使同一环境的所有刚体减去同一原点。

  def fn(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    """读取当前跟踪刚体的指定数据，并整理成扁平完整状态的一段。

    参数 env 是物理环境，command_name 默认取 motion 命令。原始位置/速度通常为
    [E,K,3]，四元数为 [E,K,4]。位置原点 [E,3] 先加刚体广播轴再相减，
    随后保留环境轴、合并刚体与坐标轴返回。cast 仅帮助类型检查，不执行运行时转换。
    此回调只读缓存，不推进参考动作。
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    val = getattr(command, attr_name)
    if subtract_origins:
      val = val - env.scene.env_origins.unsqueeze(1)
    # 保留环境轴，其余刚体轴和坐标轴展平成特征；排列顺序必须与状态表示的切片一致。
    return val.reshape(env.num_envs, -1)

  return fn


body_pos_w = _make_body_obs_fn("robot_body_pos_w", subtract_origins=True)
body_quat_w = _make_body_obs_fn("robot_body_quat_w")
body_lin_vel_w = _make_body_obs_fn("robot_body_lin_vel_w")
body_ang_vel_w = _make_body_obs_fn("robot_body_ang_vel_w")


# 函数作用：读取当前参考时刻对应的未来关节目标窗口。
def motion_command_window(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """读取当前参考时刻对应的未来关节目标窗口。

  参数 env 和 command_name 用于找到 MotionCommand，时间索引由该命令维护。
  返回 [E,N×2J]，每帧内部依次为关节位置与关节速度；未来窗口索引和片段边界处理
  委托 supermimic/src/supermimic/representations.py 的 motion_goal_window。这里取的是参考动作，不是机器人
  当前关节状态，也不是策略输出控制命令。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return motion_goal_window(command.motion, command.time_steps)


# 函数作用：把未来参考根位姿窗口表达在当前机器人根坐标系中。
def _motion_anchor_rel(
  env: ManagerBasedRlEnv, command_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """把未来参考根位姿窗口表达在当前机器人根坐标系中。

  先取参考根位置 [E,N,3] 和方向 [E,N,4]，参考位置加 env_origins 后与实际机器人
  根位姿处于同一世界系，再调用 assemble_anchor_window 做相对变换。
  返回 (rel_pos,rel_ori)，形状分别为 [E,N×3] 与 [E,N×6]。
  环境原点平移在相对计算中抵消；若漏加原点，远离场景原点的环境会出现虚假目标误差。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  target_pos, target_quat = motion_anchor_window(command.motion, command.time_steps)
  # 参考动作以共享原点存储，机器人位于平铺环境；为参考根加环境原点，统一世界系。
  target_pos = target_pos + env.scene.env_origins[:, None, :]
  # 统一到同一世界系后再做相对变换，环境原点会抵消，不会把平铺偏移误当作目标误差。
  return assemble_anchor_window(
    command.robot_anchor_pos_w, command.robot_anchor_quat_w, target_pos, target_quat
  )


# 函数作用：提供目标观察组中的相对根位置字段。
def motion_anchor_pos_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """提供目标观察组中的相对根位置字段。

  参数选择物理环境及 motion 命令；调用统一相对变换后返回第一项 [E,N×3]。
  这一字段保留在完整世界模型目标中，部署策略目标会移除它，因此位置和方向不能
  在观察配置中随意合成一个无法区分的字段。函数只读目标，不修改参考时间。
  """
  return _motion_anchor_rel(env, command_name)[0]


# 函数作用：提供目标观察组中的相对根方向字段。
def motion_anchor_ori_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """提供目标观察组中的相对根方向字段。

  参数选择环境和 motion 命令；复用 _motion_anchor_rel 的第二项 [E,N×6]。
  每帧方向由六维旋转表示编码，与策略及世界模型的目标构造保持一致。
  它与位置字段共享窗口定义和当前根位姿，不能使用独立的参考时刻。
  """
  return _motion_anchor_rel(env, command_name)[1]


# ---------------------------------------------------------------------------
# 环境配置构造函数
# ---------------------------------------------------------------------------


# 函数作用：向观察项字典原地插入一个本体传感器项。
def _add_term(terms: dict[str, ObservationTermCfg], name: str, func, **params) -> None:
  """向观察项字典原地插入一个本体传感器项。

  terms 是待修改的字典；name 同时是观察项键和共享噪声规范的查询键；func 是读取回调，
  params 作为回调参数转交 ObservationTermCfg。噪声半宽大于零时附带均匀噪声配置。
  返回 None。是否实际施加这些噪声还取决于观察组的 enable_corruption；
  本文件 actor 组默认关闭组级扰动，由状态表示路径统一添加训练噪声。
  """
  # 按观察项名称查询统一噪声半宽，避免环境观察与想象合成采用不同噪声尺度。
  half = PROPRIO_NOISE_HALF_WIDTHS[name]
  kwargs: dict = {"func": func}
  if params:
    kwargs["params"] = params
  # 只有非零噪声项才附带均匀噪声配置；观察组是否实际启用噪声由组级配置决定。
  if half > 0:
    kwargs["noise"] = Unoise(n_min=-half, n_max=half)
  terms[name] = ObservationTermCfg(**kwargs)


# 函数作用：构造策略能够从机器人传感器获得的观察配置。
def _sensor_actor_group() -> ObservationGroupCfg:
  """构造策略能够从机器人传感器获得的观察配置。

  无参数，返回 ObservationGroupCfg。拼接顺序为重力方向3、机体角速度3、
  相对默认姿态的关节位置J、关节速度J、上一动作A，总宽度为 6+2J+A。
  目标不混入此组，而放在 goal 组。joint_pos 的 biased=True 使读取包含编码器偏置。
  关闭组级噪声，避免物理采集与世界模型合成传感器各自使用不同规则或重复加噪。
  """
  # 【源码讲解】
  # 按固定顺序拼接重力方向、角速度、相对关节位置、关节速度和上一动作。
  # 此处关闭观察组噪声，由 representation 统一加噪，保持物理与想象输入一致。
  terms: dict[str, ObservationTermCfg] = {}
  _add_term(terms, "projected_gravity", mdp.projected_gravity)
  _add_term(terms, "base_ang_vel", mdp.builtin_sensor, sensor_name="robot/imu_ang_vel")
  _add_term(terms, "joint_pos", mdp.joint_pos_rel, biased=True)
  _add_term(terms, "joint_vel", mdp.joint_vel_rel)
  _add_term(terms, "actions", mdp.last_action)
  return ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )


# 函数作用：构造完整世界模型目标的观察配置。
def _goal_obs_group() -> ObservationGroupCfg:
  """构造完整世界模型目标的观察配置。

  返回顺序固定的 command、motion_anchor_pos_b、motion_anchor_ori_b 三段，
  分别是未来关节窗口、相对根平移和相对根方向。concatenate_terms=True 将它们
  按字典插入顺序拼成最后一维，enable_corruption=False 保持参考目标确定。
  策略使用的目标会排除相对平移部分，因此字段名称及顺序也是导出与目标提取协议。
  """
  # 【源码讲解】
  # 目标依次包含关节参考窗口、相对位置窗口和相对方向窗口。
  # 位置与方向保持独立命名项，部署 actor 可排除仅世界模型使用的相对位置。
  return ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(func=motion_command_window),
      "motion_anchor_pos_b": ObservationTermCfg(func=motion_anchor_pos_b),
      "motion_anchor_ori_b": ObservationTermCfg(func=motion_anchor_ori_b),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


# 函数作用：在 mjlab 的 G1 跟踪任务上构建广义坐标训练或播放配置。
def gencoord_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """在 mjlab 的 G1 跟踪任务上构建广义坐标训练或播放配置。

  play 决定基础任务采用训练还是播放设置，返回可继续修改的 ManagerBasedRlEnvCfg。
  本函数把参考锚点设为 pelvis，覆盖 actor/full_state/goal 三组观察；完整状态依次为
  根位置3、四元数4、关节位置J、世界系根线速度3、角速度3、关节速度J，共13+2J维。
  随后只保留上下文支持的三类随机化，移除自碰撞奖励及末端位置终止，调整关节限位权重。

  这些改动是为物理评分与可微想象评分保持对应；这里返回配置而非环境实例，
  真正创建仿真发生在训练入口的 create_env。full_state 不加噪，作为监督标签。
  """
  # 【源码讲解】
  # 先复用 G1 跟踪环境，再用广义坐标完整状态覆盖观察布局。
  # full_state 关闭噪声，作为监督真值；actor 传感器单独组织。
  # 仅保留上下文支持的随机化事件，并调整奖励与终止，使想象路径能够对应。
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)

  cfg.commands["motion"].anchor_body_name = "pelvis"

  cfg.observations["actor"] = _sensor_actor_group()

  # 添加不加噪的完整状态观察组，作为世界模型监督真值。
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "root_pos": ObservationTermCfg(func=root_pos),
      "root_ori": ObservationTermCfg(func=root_ori),
      "joint_pos": ObservationTermCfg(func=joint_pos),
      "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
      "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
      "joint_vel": ObservationTermCfg(func=joint_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # 添加完整目标组，未来窗口内容与相对根目标使用同一参考时刻。
  cfg.observations["goal"] = _goal_obs_group()

  # 过滤随机化事件，使物理训练条件处于 SimContext 能显式读取并提供给动力学的范围内。
  cfg.events = {
    name: term
    for name, term in cfg.events.items()
    if name in {"foot_friction", "base_com", "encoder_bias"}
  }

  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["joint_limit"].weight = -2.0
  # cfg.rewards.pop("motion_global_root_pos", None)
  # cfg.rewards.pop("motion_global_root_ori", None)

  # 移除末端身体位置终止，None 使该项不存在时也安全；保留的阈值供世界模型软终止读取。
  cfg.terminations.pop("ee_body_pos", None)

  return cfg


# 函数作用：复用公共任务设置，改为逐刚体最大坐标完整状态。
def maxcoord_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """复用公共任务设置，改为逐刚体最大坐标完整状态。

  参数 play 原样传给 gencoord_env_cfg；返回 ManagerBasedRlEnvCfg。
  完整状态四段依次是所有刚体位置、所有刚体四元数、所有线速度、所有角速度，
  宽度分别3K、4K、3K、3K，共13K。它不是把每个刚体的13维交错排列。
  actor 传感器和目标仍沿用公共布局；最大坐标表示根据这些段宽度切片，顺序不能调换。
  """
  # 【源码讲解】
  # 继承通用设置后，仅替换完整状态为逐刚体位置、四元数、线速度和角速度。
  # 各项拼接顺序必须与 MaxCoordRepresentation 的切片一致，每刚体总计13维。
  cfg = gencoord_env_cfg(play=play)

  # 将完整状态替换为逐刚体最大坐标布局。
  # 布局为全部位置3K、全部四元数4K、全部线速度3K、全部角速度3K。
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "body_pos_w": ObservationTermCfg(
        func=body_pos_w,
        params={"command_name": "motion"},
      ),
      "body_quat_w": ObservationTermCfg(
        func=body_quat_w,
        params={"command_name": "motion"},
      ),
      "body_lin_vel_w": ObservationTermCfg(
        func=body_lin_vel_w,
        params={"command_name": "motion"},
      ),
      "body_ang_vel_w": ObservationTermCfg(
        func=body_ang_vel_w,
        params={"command_name": "motion"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.observations["goal"] = _goal_obs_group()

  return cfg
