"""世界模型状态语义、未来动作目标、坐标积分、传感器与奖励接口。"""

# 【文件作用】状态语义与坐标表示
# 定义完整状态如何转为网络表示、预测如何积分、目标如何构造，以及奖励和损失需要哪些状态量。
# 【阅读顺序】目标窗口工具 → StateRepresentation 公共接口 → GenCoordRepresentation → MaxCoordRepresentation。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟类型注解求值，使本文件可用稍后定义的类型并减少循环导入；不改变张量计算。
from __future__ import annotations


# 提供 sqrt 等标量数学常量；根方向损失使用 sqrt(0.5) 调整旋转矩阵误差尺度。
import math
# ABC/abstractmethod 声明必须由具体状态表示或条件规格实现的接口，避免误用基类。
from abc import ABC, abstractmethod
# TYPE_CHECKING 内的导入只服务类型检查，避免运行时循环依赖；Literal 限定观察模式取值。
from typing import TYPE_CHECKING, Literal

# PyTorch 张量、随机采样、设备和自动微分工具；计算沿最后一维组织特征，前导轴保留批次/时间。
import torch
# nn.Module 为状态表示提供模块与 buffer 注册机制，让固定重力、噪声等张量随设备迁移。
import torch.nn as nn
# matrix_from_quat 把四元数转旋转矩阵；quat_apply/quat_apply_inverse 在世界系与根系间旋转向量；
# subtract_frame_transforms 计算当前根到未来参考根的相对位姿。
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  # ManagerBasedRlEnv 是未包装物理环境类型，用于查询机器人、命令和随机化事件；这里只作类型说明。
  from mjlab.envs import ManagerBasedRlEnv
  # MotionLoader 描述参考动作查表接口，提供帧边界、关节状态和根位姿；此导入只供类型检查。
  from mjlab.tasks.tracking.mdp.commands import MotionLoader
  # ForwardKinematics 的类型用于广义坐标身体状态恢复；实例在 bind 时取自参考动作加载器。
  from mjlab.utils.forward_kinematics import ForwardKinematics

  # 来自 supermimic/src/supermimic/sim_context.py，SimContext 定义物理条件列布局；最大坐标传感器合成按列恢复编码器偏置。
  from supermimic.sim_context import SimContext

# 来自 supermimic/src/supermimic/utils/analytic_ik.py，AnalyticIK 把身体旋转/角速度转换为关节位置/速度。
from supermimic.utils.analytic_ik import AnalyticIK
# 来自 supermimic/src/supermimic/utils/gencoord_helpers.py：compute_gencoord_state 转换局部状态，
# decompose_rep_state 拆各语义段，infer_num_joints_from_state_dim 与 rep_state_dim_from_num_joints 推导布局尺寸。
from supermimic.utils.gencoord_helpers import (
  compute_gencoord_state,
  decompose_rep_state,
  infer_num_joints_from_state_dim,
  rep_state_dim_from_num_joints,
)
# 来自 supermimic/src/supermimic/utils/helpers.py：quat_integrate 积分根姿态；quat_to_rot6d/rot6d_to_matrix 转换旋转表示；
# rodrigues_precompute 预计算轴角旋转项，rodrigues_apply 将其施加到向量或旋转列。
from supermimic.utils.helpers import (
  quat_integrate,
  quat_to_rot6d,
  rodrigues_apply,
  rodrigues_precompute,
  rot6d_to_matrix,
)
# 来自 supermimic/src/supermimic/utils/maxcoord_helpers.py：compute_maxcoord_rep_state/decompose_maxcoord_rep_state 转换与拆解最大坐标；
# 三个 maxcoord_*_dim 工具计算宽度；reconstruct_full_from_aux_and_rep 与 world_frame_bodies 恢复世界系状态。
from supermimic.utils.maxcoord_helpers import (
  compute_maxcoord_rep_state,
  decompose_maxcoord_rep_state,
  maxcoord_full_state_dim,
  maxcoord_prediction_dim,
  maxcoord_rep_state_dim,
  reconstruct_full_from_aux_and_rep,
  world_frame_bodies,
)
# 来自 supermimic/src/supermimic/utils/tracking_reward.py，tracking_reward 统一计算身体跟踪与动作/关节惩罚，使两种表示共享奖励标准。
from supermimic.utils.tracking_reward import tracking_reward


# 函数作用：按表示名称创建状态语义对象，并设置观察和积分所需配置。
def create_representation(
  name: str,
  full_state_dim: int,
  num_actions: int,
  dt: float,
  decimation: int,
  default_joint_pos: torch.Tensor,
  actor_obs_mode: str,
) -> StateRepresentation:
  """按表示名称创建状态语义对象，并设置观察和积分所需配置。

  name 选择 gencoord/maxcoord；full_state_dim 用于推断关节或刚体数；num_actions
  用于传感器噪声宽度；dt 为控制步长，decimation 为每控制步物理子步数；
  default_joint_pos 为默认关节姿态；actor_obs_mode 选择状态或传感器输入。
  返回 StateRepresentation 子类实例，尚需 bind 绑定实际环境。

  proprio 模式注册非持久噪声半宽 buffer；子步系数 (n+1)/(2n) 用于匹配常加速度下
  半隐式子步合成的位置积分。未知表示或最大坐标宽度不能被13整除时直接报错。
  """
  # 两种表示对完整状态的布局假设不同，必须与环境配置保持一致。
  if name == "gencoord":
    num_joints = infer_num_joints_from_state_dim(full_state_dim)
    rep: StateRepresentation = GenCoordRepresentation(num_joints, default_joint_pos)
  elif name == "maxcoord":
    if full_state_dim % 13 != 0:
      raise ValueError(
        f"Maxcoord full_state_dim ({full_state_dim}) must be divisible by 13"
      )
    rep = MaxCoordRepresentation(full_state_dim // 13, default_joint_pos)
  else:
    raise ValueError(
      f"Unknown representation: {name}. Must be 'gencoord' or 'maxcoord'."
    )
  rep.actor_obs_mode = actor_obs_mode
  if actor_obs_mode == "proprio":
    rep.register_buffer(
      "_proprio_noise_bounds",
      rep.proprio_noise_bounds(num_actions, default_joint_pos.device),
      persistent=False,
    )
  rep.dt = dt
  # Semi-implicit composition of `decimation` substeps under constant accel.
  # 控制步由多个物理子步组成；此系数用于预测增量的半隐式位置积分。
  rep.substep_coeff = (decimation + 1) / (2 * decimation)
  return rep


# ---------------------------------------------------------------------------
# 未来目标布局常量
# ---------------------------------------------------------------------------
# 目标窗口取 NUM_GOAL_FRAMES 帧，每帧相隔 GOAL_FRAME_STRIDE 个参考控制步。
# 以50Hz参考为例，步距5即0.1秒，10帧包含当前帧及约0.9秒未来信息。
NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5

# wm_goal  = [content, rel_pos_b(3*N), rel_ori_b(6*N)]
# actor_goal = [content, rel_ori_b(6*N)]
# content = NUM_GOAL_FRAMES joint targets: per frame [joint_pos, joint_vel].
GOAL_REL_POS_DIM = 3 * NUM_GOAL_FRAMES  # rel_pos_b: in wm_goal only
GOAL_REL_ORI_DIM = 6 * NUM_GOAL_FRAMES  # rel_ori_b: in both wm_goal and actor_goal


# 函数作用：计算策略侧目标宽度。
def actor_goal_dim(num_goals: int) -> int:
  """计算策略侧目标宽度。

  参数 num_goals 是完整世界模型目标宽度，返回扣除未来根相对位置后的宽度。
  该公式依赖统一目标布局 [动作内容,相对位置,相对方向]，不适用于任意命令向量。
  速度任务的三维宽度由其 TaskSpec 独立声明。
  """
  return num_goals - GOAL_REL_POS_DIM


PROPRIO_NOISE_HALF_WIDTHS: dict[str, float] = {
  "projected_gravity": 0.05,
  "base_ang_vel": 0.2,
  "joint_pos": 0.01,
  "joint_vel": 0.5,
  "actions": 0.0,
}


# 函数作用：把按观察项排列的历史转换为按时间帧排列的历史。
def term_major_to_temporal(
  flat: torch.Tensor, term_widths: list[int], history_length: int
) -> torch.Tensor:
  """把按观察项排列的历史转换为按时间帧排列的历史。

  参数 flat:[B,H*sum(widths)]、term_widths 为各观察项单帧宽度、history_length=H。
  输入先排列某项全部历史再排下一项；逐项切片重塑后沿特征轴拼成 [B,H,sum(widths)]。

  返回新组织的张量，不修改原始历史。导出边界用它对齐部署观察与训练解码器布局；
  两者特征总数相同也可能顺序不同，不能只 reshape 整个 flat 来替代本步骤。
  """
  parts = []
  # off 跟踪扁平历史中当前观察项的起始列；每项占 H×w 列，拆出后重塑为 [B,H,w]。
  off = 0
  for w in term_widths:
    chunk = flat[..., off : off + history_length * w]
    parts.append(chunk.unflatten(-1, (history_length, w)))
    off += history_length * w
  # 各项已对齐到相同历史帧轴，沿特征轴拼回 [B,H,各项宽度之和]。
  return torch.cat(parts, dim=-1)


# 函数作用：从完整世界模型目标提取策略可见部分。
def extract_actor_goal(wm_goal: torch.Tensor) -> torch.Tensor:
  """从完整世界模型目标提取策略可见部分。

  参数 wm_goal:[...,G]；返回 [...,G-GOAL_REL_POS_DIM]。保留前面的关节窗口和
  末尾相对方向窗口，移除中间的相对位置。只做切片拼接，不重新查询参考动作。
  """
  tail = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
  return torch.cat([wm_goal[..., :-tail], wm_goal[..., -GOAL_REL_ORI_DIM:]], dim=-1)


# 函数作用：把根位姿转换为可按平方误差监督的命名通道。
def _root_loss_blocks(aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
  """把根位姿转换为可按平方误差监督的命名通道。

  参数 aux_state:[...,7] 为位置与四元数；返回 root_pos:[...,3]、root_ori:[...,9]。
  四元数转旋转矩阵后乘 sqrt(1/2)，使矩阵平方距离在小角度时近似旋转角平方，
  从而方向标称误差可按弧度理解。函数保留输入梯度，供多步模型损失使用。
  """
  # 四元数转 3×3 旋转矩阵并展开为 9 通道；乘 sqrt(0.5) 后，
  # 小角度时矩阵平方误差近似角度平方，使方向标称误差可按弧度解释。
  rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
  return {"root_pos": aux_state[..., :3], "root_ori": rot * math.sqrt(0.5)}


# 函数作用：统一拼接世界模型目标和策略目标。
def _assemble_goal(
  content: torch.Tensor,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """统一拼接世界模型目标和策略目标。

  参数 content 是已展开关节参考窗口；root_pos/quat 是当前根状态；target_pos/quat
  是未来根窗口。先算每帧相对位姿，再分别返回 (wm_goal,actor_goal)。
  前者为 [内容,相对位置,相对方向]，后者为 [内容,相对方向]。

  目标的坐标基准依赖当前根，因此想象状态变化也会改变相对目标。这条共用路径
  用于物理训练、想象和播放，避免三个位置独立拼接而产生错位。
  """
  rel_pos, rel_ori = assemble_anchor_window(
    root_pos, root_quat, target_pos, target_quat
  )
  # 世界模型目标保留相对平移以支持特权观察；actor 只取得动作内容和相对方向。
  # 两个返回值不能互换，否则会破坏网络宽度和部署输入约定。
  wm_goal = torch.cat([content, rel_pos, rel_ori], dim=-1)
  actor_goal = torch.cat([content, rel_ori], dim=-1)
  return wm_goal, actor_goal


# 函数作用：取得当前参考片段内的未来目标窗口索引。
def goal_window_indices(
  motion: MotionLoader, motion_times: torch.Tensor
) -> torch.Tensor:
  """取得当前参考片段内的未来目标窗口索引。

  参数 motion 提供各片段右开结束位置，motion_times:[E] 是当前帧；返回 [E,N]，
  候选为 t+i*GOAL_FRAME_STRIDE，N=NUM_GOAL_FRAMES。先查当前片段，再将
  超出末端的未来帧夹到该片段最后有效帧，因此尾部目标会重复而不串入下一片段。

  只构造索引，不推进 motion_times。关节窗口与根窗口必须复用同一规则。
  """
  # 【源码讲解】
  # 按固定步距取未来参考帧，边界限制应遵循当前片段，防止目标跨片段。
  # 输出索引用于关节目标与根目标，二者必须取相同时间窗口。
  # motion_ends 保存各片段的右开结束位置；未来目标不得越过当前片段末帧。
  clip = torch.searchsorted(motion.motion_ends, motion_times, right=True)
  # Past-the-end guard: searchsorted returns numel() at the final frame.
  clip = clip.clamp(max=motion.motion_ends.numel() - 1)
  last_valid = motion.motion_ends[clip] - 1
  offsets = (
    torch.arange(NUM_GOAL_FRAMES, device=motion_times.device) * GOAL_FRAME_STRIDE
  )
  return torch.minimum(motion_times[:, None] + offsets, last_valid[:, None])


# 函数作用：读取未来关节位置/速度，构成目标的动作内容部分。
def motion_goal_window(
  motion: MotionLoader, motion_times: torch.Tensor
) -> torch.Tensor:
  """读取未来关节位置/速度，构成目标的动作内容部分。

  参数 motion 为参考加载器，motion_times:[E]；返回 [E,N*2J]。
  索引得到 [E,N,J] 位置与速度，每帧先位置后速度拼为 [E,N,2J]，再展平未来帧。
  不加入根相对位姿，也不把参考动作当作策略输出的控制命令。
  """
  idx = goal_window_indices(motion, motion_times)
  # idx 为 [E,N]；查表得到 [E,N,J]，位置与速度合成 [E,N,2J]，再展平未来帧。
  frames = torch.cat([motion.joint_pos[idx], motion.joint_vel[idx]], dim=-1)
  # 仅保留环境轴，把时间与关节特征合并；每帧内部仍按位置后速度排列。
  return frames.reshape(frames.shape[0], -1)


# 函数作用：读取与关节窗口同一组未来帧的参考根位姿。
def motion_anchor_window(
  motion: MotionLoader, motion_times: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """读取与关节窗口同一组未来帧的参考根位姿。

  参数 motion、motion_times:[E]；返回位置 [E,N,3] 和四元数 [E,N,4]。
  此处仍为世界系参考数据，后续 assemble_anchor_window 才转到当前根相对系。
  """
  idx = goal_window_indices(motion, motion_times)
  return motion.root_pos_w[idx], motion.root_quat_w[idx]


# 函数作用：把未来参考根窗口表达为相对于当前机器人根的位姿。
def assemble_anchor_window(
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """把未来参考根窗口表达为相对于当前机器人根的位姿。

  参数 root_pos:[E,3]、root_quat:[E,4]；target_pos:[E,N,3]、target_quat:[E,N,4]。
  广播当前根到每个未来帧，合并 E/N 两轴做批量相对变换，再将方向转六维旋转。
  返回展平的 rel_pos:[E,N*3]、rel_ori:[E,N*6]。

  当前根和参考根必须使用同一世界原点；统一的环境平移在相减时抵消。
  函数不重新采样参考、不推进时间，所得相对目标仍对当前根状态可微。
  """
  # 当前根每环境只有一份 [E,3/4]，目标窗口每环境有 N 帧。
  # 将当前根广播到每个未来帧，再合并 E、N 两轴调用批量坐标变换。
  num_envs, num_frames = target_pos.shape[:2]
  root_pos = root_pos[:, None, :].expand(num_envs, num_frames, 3).reshape(-1, 3)
  root_quat = root_quat[:, None, :].expand(num_envs, num_frames, 4).reshape(-1, 4)
  # 相对位置相当于 R_root^T×(target_pos-root_pos)，方向为当前根到目标根的旋转。
  # 两者必须处于同一世界原点约定，环境平移偏移才会抵消。
  rel_pos, rel_ori_w = subtract_frame_transforms(
    root_pos, root_quat, target_pos.reshape(-1, 3), target_quat.reshape(-1, 4)
  )
  rel_pos = rel_pos.reshape(num_envs, -1)
  # 四元数转换为六维旋转表示，再把 N 帧展开为 [E,N×6]；位置相应为 [E,N×3]。
  rel_ori = quat_to_rot6d(rel_ori_w).reshape(num_envs, -1)
  return rel_pos, rel_ori


_compiled_assemble_goal = torch.compile(_assemble_goal)


class StateRepresentation(nn.Module, ABC):
  """广义坐标与最大坐标的统一状态语义接口，也是奖励和观察合成的桥梁。

  完整物理状态、模型表示、辅助根状态、预测增量是不同布局，子类必须声明各自宽度，
  并实现 represent、apply_prediction、reconstruct_full_state 等接口。
  基类负责共享传感器噪声规范、目标窗口、策略输入组织及统一身体跟踪奖励。

  它继承 nn.Module 以支持 buffer 与设备迁移，但不是动力学 MLP 本身。
  Dynamics 调用其积分接口，WorldModelLearner 调用 loss_blocks，想象环境调用
  传感器和奖励接口。新增表示时这几条路径必须对同一状态布局保持一致。
  """

  full_state_dim: int
  rep_state_dim: int
  prediction_dim: int
  num_joints: int
  actor_obs_mode: Literal["rep_state", "proprio"] = "rep_state"

  # Each channel group's nominal-error field on ``WorldModelLearnerCfg``. Every
  # representation scores the integrated root pose; subclasses add their own.
  LOSS_WEIGHT_CONFIG: dict[str, str] = {
    "root_pos": "nominal_error_root_pose_pos",
    "root_ori": "nominal_error_root_pose_ori",
  }

  substep_coeff: float = 1.0

  # ------------------------------------------------------------------
  # Abstract methods (must be implemented by subclasses)
  # ------------------------------------------------------------------

  # 函数作用：约定把完整状态 [...,S] 转换为模型输入表示 [...,R]。
  @abstractmethod
  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    """约定把完整状态 [...,S] 转换为模型输入表示 [...,R]。

    子类决定局部坐标、姿态参数化与特征顺序；保留所有前导批次/时间轴。
    基类为抽象接口，不执行转换。完整状态里的世界根位姿由 extract_aux_state 单独保存。
    """
    ...

  # 函数作用：约定从完整状态提取根位置和四元数。
  @abstractmethod
  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    """约定从完整状态提取根位置和四元数。

    输入 [...,S]，输出 [...,7]=[位置xyz,四元数wxyz]。两种完整状态的切片位置不同，
    所以由子类实现，不能假定所有布局的前七列都是根位姿。
    """
    ...

  # 函数作用：约定把网络预测增量积分为下一状态。
  @abstractmethod
  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """约定把网络预测增量积分为下一状态。

    参数 aux_state:[...,7]、curr_rep_state:[...,R]、prediction:[...,P] 与控制步长 dt；
    返回 (next_aux_state,next_rep_state)，辅助根状态在前。子类决定增量分组、坐标系、
    姿态积分与子步修正，必须保留对动作经过 prediction 的梯度。
    """
    ...

  # 函数作用：约定从模型状态恢复完整物理状态。
  @abstractmethod
  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    """约定从模型状态恢复完整物理状态。

    参数 aux_state:[...,7] 与 rep_state:[...,R]；返回 [...,S]，用于可视化等路径。
    重建必须恢复世界系速度和该表示的完整排列，不能直接将局部特征拼接后当作完整状态。
    """
    ...

  # 函数作用：绑定奖励需要的机器人关节软限位。
  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    """绑定奖励需要的机器人关节软限位。

    参数 env 为未包装环境，sim_context 为已构造上下文；基类从机器人取第零环境
    共享的关节限位。返回 None，子类可继续绑定正运动学或逆运动学和上下文切片。
    这一步依赖实际资产索引，应在环境及表示创建后执行。
    """
    self._soft_joint_pos_limits = env.scene["robot"].data.soft_joint_pos_limits[0]

  # 函数作用：约定返回按机器人关节顺序排列的位置与速度。
  @abstractmethod
  def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    """约定返回按机器人关节顺序排列的位置与速度。

    输入 rep_state:[...,R]；返回 [...,2J]=[关节位置,关节速度]。
    广义坐标可直接切片，最大坐标需逆运动学恢复；用于传感器及关节限位奖励。
    """
    ...

  # 函数作用：约定从想象状态合成不含上一动作的本体传感器。
  @abstractmethod
  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    joint_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    """约定从想象状态合成不含上一动作的本体传感器。

    参数 rep_state、aux_state、joint_state 与 sim_ctx；返回重力方向、根角速度、
    相对关节位置和关节速度的拼接。子类负责坐标与偏置细节，上一动作由外层追加。
    此接口使策略在物理采集与可微想象中读取相同语义的传感器特征。
    """
    ...

  # 函数作用：计算含上一动作的完整本体传感器宽度。
  def proprio_obs_dim(self, num_actions: int) -> int:
    """计算含上一动作的完整本体传感器宽度。

    参数 num_actions=A；返回 proprio_term_widths(A) 的和，即 6+2J+A。
    只是尺寸查询，不读取环境或为网络生成传感器样本。
    """
    return sum(self.proprio_term_widths(num_actions))

  # 函数作用：返回传感器各项单帧特征宽度及其拼接顺序。
  def proprio_term_widths(self, num_actions: int) -> list[int]:
    """返回传感器各项单帧特征宽度及其拼接顺序。

    参数 num_actions=A；返回 [3,3,J,J,A]，对应重力方向、根角速度、关节位置、
    关节速度与上一动作。环境观察、噪声展开与 ONNX 历史重排都依赖这一顺序。
    """
    return [3, 3, self.num_joints, self.num_joints, num_actions]

  # 函数作用：将逐观察项噪声半宽展开成逐特征向量。
  def proprio_noise_bounds(
    self, num_actions: int, device: torch.device | str
  ) -> torch.Tensor:
    """将逐观察项噪声半宽展开成逐特征向量。

    参数 num_actions 与 device；返回 [6+2J+A]。每项按照其宽度重复相应半宽，
    顺序与传感器布局一致。这里只定义尺度，不抽随机数；动作项默认不加噪声。
    """
    widths = self.proprio_term_widths(num_actions)
    halves = PROPRIO_NOISE_HALF_WIDTHS.values()
    return torch.cat(
      [torch.full((w,), h, device=device) for w, h in zip(widths, halves, strict=True)]
    )

  # 函数作用：计算历史观察与目标展开后的扁平策略输入宽度。
  def actor_obs_dim(self, num_actions: int, num_goals: int, history_length: int) -> int:
    """计算历史观察与目标展开后的扁平策略输入宽度。

    参数动作数、完整目标宽度及 history_length=H；返回 H*单帧宽度+actor_goal_dim。
    单帧宽度依观察模式为 R 或传感器宽度；未来动作目标已有自己的窗口，不再乘 H。
    这是接口尺寸查询，多编码器策略仍分别接收 history 和各任务目标字典。
    """
    # 观察模式决定每帧宽度，历史长度乘在每帧宽度上；参考目标只追加一次。
    sensor_or_state = (
      self.rep_state_dim
      if self.actor_obs_mode == "rep_state"
      else self.proprio_obs_dim(num_actions)
    )
    # 历史只重复状态/传感器部分，未来动作目标已经自带窗口结构，不再乘历史长度。
    return history_length * sensor_or_state + actor_goal_dim(num_goals)

  # 函数作用：按统一规范为本体传感器增加逐特征均匀扰动。
  def add_proprio_noise(self, obs: torch.Tensor) -> torch.Tensor:
    """按统一规范为本体传感器增加逐特征均匀扰动。

    参数 obs:[...,D]；返回同形状张量。随机值先映射到 [-1,1)，乘注册的半宽后相加。
    噪声项 detach，不对随机抽样求导；输入 obs 的梯度仍能通过加法传递。
    仅适用于已由工厂为 proprio 模式注册噪声范围的表示对象。
    """
    return obs + ((torch.rand_like(obs) * 2 - 1) * self._proprio_noise_bounds).detach()

  # 函数作用：选择策略的状态输入来源，并按需添加传感器噪声。
  def compute_actor_sensor(
    self,
    rep_state: torch.Tensor,
    sim_ctx: torch.Tensor,
    aux_state: torch.Tensor | None = None,
    joint_state: torch.Tensor | None = None,
    last_action: torch.Tensor | None = None,
    sim_actor_obs: torch.Tensor | None = None,
    noisy: bool = False,
  ) -> torch.Tensor:
    """选择策略的状态输入来源，并按需添加传感器噪声。

    参数 rep_state、sim_ctx 为必需；proprio 模式优先使用 sim_actor_obs 真实传感器，
    未提供时要求 aux_state、joint_state、last_action，先 compute_wm_obs 再追加动作。
    rep_state 模式直接返回表示，不走传感器合成与噪声分支。

    返回 [B,D] 单帧输入，不含目标，也不组织历史。noisy 控制是否加统一噪声；
    长历史窗口的维护和重置填充由 runner/想象环境各自负责。
    """
    # 【源码讲解】
    # rep_state 模式直接使用模型表示；proprio 模式优先使用实际仿真传感器。
    # 想象环境无真实传感器时，用预测状态合成并追加上一动作，再按统一幅度加噪。
    # 状态模式不合成传感器，直接返回模型表示；proprio 模式才需要后续读取或合成。
    if self.actor_obs_mode == "rep_state":
      return rep_state
    if sim_actor_obs is not None:
      obs = sim_actor_obs
    else:
      if aux_state is None or joint_state is None or last_action is None:
        raise ValueError(
          "aux_state, joint_state and last_action are required to synthesize "
          "proprio obs (no sim_actor_obs provided)"
        )
      # 先得到重力、角速度及关节信息，再追加上一动作，保持与真实传感器布局一致。
      obs = self.compute_wm_obs(rep_state, aux_state, joint_state, sim_ctx)
      obs = torch.cat([obs, last_action], dim=-1)
    if noisy:
      obs = self.add_proprio_noise(obs)
    return obs

  # 函数作用：展开传感器历史并与策略目标拼成单张量输入。
  def build_actor_obs(
    self, sensor_history: torch.Tensor, actor_goal: torch.Tensor
  ) -> torch.Tensor:
    """展开传感器历史并与策略目标拼成单张量输入。

    参数 sensor_history:[B,H,D]、actor_goal:[B,G_actor]；返回 [B,H*D+G_actor]。
    保留样本顺序，历史最新帧原本位于末尾。此工具用于单张量观察路径，多任务 actor
    使用 TensorDict 路径按任务分别传目标。
    """
    return torch.cat([sensor_history.flatten(start_dim=1), actor_goal], dim=-1)

  # ------------------------------------------------------------------
  # Default implementations (may be overridden)
  # ------------------------------------------------------------------

  # 函数作用：按当前参考时刻构建世界模型与策略两套目标。
  def compute_wm_goal(
    self,
    motion: MotionLoader,
    motion_times: torch.Tensor,
    aux_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """按当前参考时刻构建世界模型与策略两套目标。

    参数 motion 提供参考动作，motion_times:[E]，aux_state:[E,7] 是当前根位姿。
    读取未来关节和根窗口，再通过共享拼接函数返回 (wm_goal,actor_goal)。

    目标相对方向/位置依赖当前根状态，会随想象状态变化；参考窗口的选择只依赖时间。
    共享函数被编译用于减少重复张量运算，布局规则仍与物理环境观察保持一致。
    """
    # 【源码讲解】
    # 同一参考窗口既生成模型目标也生成 actor 目标，避免各路径拼接顺序不一致。
    # 当前根位姿参与相对位置与方向计算，因此目标随预测根状态变化。
    content = motion_goal_window(motion, motion_times)
    target_pos, target_quat = motion_anchor_window(motion, motion_times)
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    return _compiled_assemble_goal(
      content, root_pos, root_quat, target_pos, target_quat
    )

  # 函数作用：约定恢复跟踪刚体的世界系状态以计算统一奖励。
  @abstractmethod
  def world_frame_bodies(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """约定恢复跟踪刚体的世界系状态以计算统一奖励。

    输入 aux_state、rep_state；返回位置、旋转矩阵、线速度和角速度四元组。
    位置/速度末轴为3，旋转末两轴为3×3，额外保留批次和刚体轴。
    子类可通过关节正运动学或最大坐标恢复，不能混用机体系速度与世界系目标。
    """
    ...

  # 函数作用：将预测状态恢复为身体状态并计算逐样本跟踪奖励。
  def compute_wm_reward_from_reference(
    self,
    next_rep_state: torch.Tensor,
    next_aux_state: torch.Tensor,
    next_joint_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    target_body_pos_w: torch.Tensor,
    target_body_quat_w: torch.Tensor,
    target_body_lin_vel_w: torch.Tensor,
    target_body_ang_vel_w: torch.Tensor,
  ) -> torch.Tensor:
    """将预测状态恢复为身体状态并计算逐样本跟踪奖励。

    参数 next_rep/aux/joint_state 为动作后的预测；action/last_action 为原始策略输出；
    后四个 target_body 字段为同一参考时刻的世界系身体数据。返回 [... ] 标量奖励，
    调用统一 tracking_reward 处理重锚定、误差核、动作变化、关节超限和 dt 缩放。

    这里不推进参考时间，也不再次缩放动作；目标时刻由 WorldModelEnv 缓存管理。
    状态恢复路径保留梯度，使奖励可通过动力学预测回传到策略。
    """
    # 先把当前表示恢复为世界系刚体信息，再传给统一奖励函数。
    # 动作仍为原始策略输出，动作变化惩罚不使用动力学内部缩放后的命令。
    return tracking_reward(
      next_aux_state,
      *self.world_frame_bodies(next_aux_state, next_rep_state),
      next_joint_state,
      action,
      last_action,
      target_body_pos_w,
      target_body_quat_w,
      target_body_lin_vel_w,
      target_body_ang_vel_w,
      self._soft_joint_pos_limits,
      self.dt,
    )

  # 函数作用：约定将状态拆成世界模型监督使用的命名通道组。
  @abstractmethod
  def loss_blocks(
    self, rep_state: torch.Tensor, aux_state: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """约定将状态拆成世界模型监督使用的命名通道组。

    参数 rep_state、aux_state；返回组名到张量的字典。键必须与子类
    LOSS_WEIGHT_CONFIG 对应，未返回的通道不会直接加入状态损失。
    每组保留批次/时间轴，由 learner 计算均方误差及按标称尺度加权。
    """
    ...


class GenCoordRepresentation(StateRepresentation):
  """以根与关节为核心的广义坐标表示。

  完整状态为根位置3、四元数4、关节位置J、世界线速度3、世界角速度3、关节速度J。
  模型表示为高度1、机体系重力3、根线速度3、角速度3、关节位置J、关节速度J；
  辅助状态保存世界根位置与四元数，网络预测 [根线增量3,根角增量3,关节速度增量J]。

  积分解析更新位置、姿态和速度，避免网络直接输出互相矛盾的高度/重力方向。
  奖励通过绑定的正运动学恢复世界系身体状态；监督损失比较根位姿与关节状态。
  """

  LOSS_WEIGHT_CONFIG = {
    **StateRepresentation.LOSS_WEIGHT_CONFIG,
    "joint_pos": "nominal_error_joint_pos",
    "joint_vel": "nominal_error_joint_vel",
  }

  # 函数作用：设置关节数、三类状态宽度与默认关节姿态。
  def __init__(self, num_joints: int, default_joint_pos: torch.Tensor):
    """设置关节数、三类状态宽度与默认关节姿态。

    参数 num_joints=J、default_joint_pos:[J]；完整宽度13+2J、表示宽度10+2J、
    预测宽度6+J。返回 None，fk 此时仅作类型声明，必须等 bind 获取环境运动学对象。
    默认关节姿态用于从绝对关节位置生成相对传感器观察，不是当前参考目标。
    """
    super().__init__()
    self.num_joints = num_joints
    self.full_state_dim = 13 + 2 * num_joints
    self.rep_state_dim = rep_state_dim_from_num_joints(num_joints)
    self.prediction_dim = 6 + num_joints  # [Δv_b(3), Δw_b(3), Δvj(N)]
    self._default_joint_pos = default_joint_pos
    self.fk: ForwardKinematics  # set by bind

  # 函数作用：绑定公共限位与参考动作加载器的正运动学对象。
  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    """绑定公共限位与参考动作加载器的正运动学对象。

    参数 env、sim_context；先执行基类绑定，再从 motion 命令取 fk。返回 None。
    绑定后 world_frame_bodies 能将关节空间预测转为身体空间奖励所需的位姿与速度。
    """
    super().bind(env, sim_context)
    self.fk = env.command_manager.get_term("motion").motion.fk

  # 函数作用：把完整世界系关节状态转成机体系模型表示。
  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    """把完整世界系关节状态转成机体系模型表示。

    输入 full_state:[...,13+2J]；先合并前导批次轴，调用广义坐标工具，再恢复原前导轴，
    输出 [...,10+2J]。这允许同一函数处理单步 [B,S] 或训练窗口 [B,T,S]。
    不修改输入，根世界位姿另由 extract_aux_state 保存。
    """
    flat = full_state.reshape(-1, self.full_state_dim)
    rep = compute_gencoord_state(flat)
    return rep.reshape(*full_state.shape[:-1], -1)

  # 函数作用：提取广义坐标完整状态开头的根位置和四元数。
  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    """提取广义坐标完整状态开头的根位置和四元数。

    输入 [...,13+2J]，返回前七列 [...,7]。这里直接切片而不克隆，调用者若需要
    独立快照或切断梯度，应在自己的边界处理。
    """
    return full_state[..., :7]

  # 函数作用：以速度增量积分广义坐标，返回下一根位姿和下一表示。
  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """以速度增量积分广义坐标，返回下一根位姿和下一表示。

    输入 aux:[...,7]、curr_rep:[...,10+2J]、prediction:[...,6+J] 与 dt。
    先分出当前机体系根速度/角速度和关节速度增量；速度加全增量，关节位置用
    当前速度加 substep_coeff 倍增量积分。根平移与姿态使用同样修正后的速度。

    根姿态变化后，将下一世界速度旋回新根坐标系，并重新从新姿态推导重力方向、
    从新根位置取高度。输出布局与输入一致，返回顺序 (next_aux,next_rep)。
    这样同一预测中根姿态、高度和重力保持一致，梯度可贯穿整个解析积分过程。
    """
    # 【源码讲解】
    # 网络预测当前根坐标系下的线/角速度增量和关节速度增量。
    # 速度加完整增量；位置用子步系数加权后的速度积分，姿态需同步旋转更新。
    # 最终将速度重新表达于下一根坐标系，并由积分根姿态推导高度与重力方向。
    N = self.num_joints
    # c=(n+1)/(2n)：将总速度增量均分到 n 个半隐式物理子步后的平均系数。
    # 网络预测的是整控制步的速度增量，而不是直接预测关节位置。
    c = self.substep_coeff
    _, _, v_b, w_b, qj, vj = decompose_rep_state(curr_rep_state)
    dv_b = prediction[..., 0:3]
    dw_b = prediction[..., 3:6]
    dvj = prediction[..., 6 : 6 + N]

    next_v_cur = v_b + dv_b  # next twist, expressed in the current root frame
    next_w_cur = w_b + dw_b
    next_vj = vj + dvj
    # 位置 = 原位置 + 时间间隔 × 加权速度；速度则加完整增量。
    next_qj = qj + (vj + c * dvj) * dt

    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]
    v_global = quat_apply(root_quat, next_v_cur)
    w_global = quat_apply(root_quat, next_w_cur)
    v_pose = quat_apply(root_quat, v_b + c * dv_b)
    w_pose = quat_apply(root_quat, w_b + c * dw_b)
    next_root_pos = root_pos + v_pose * dt
    next_root_quat = quat_integrate(root_quat, w_pose, dt)

    # 根姿态已改变，必须将世界速度重新转到“新的”根坐标系。
    next_v_b = quat_apply_inverse(next_root_quat, v_global)
    next_w_b = quat_apply_inverse(next_root_quat, w_global)
    world_down = torch.zeros_like(v_global)
    world_down[..., 2] = -1.0
    next_g_b = quat_apply_inverse(next_root_quat, world_down)

    next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
    next_rep = torch.cat(
      [next_root_pos[..., 2:3], next_g_b, next_v_b, next_w_b, next_qj, next_vj],
      dim=-1,
    )
    return next_aux, next_rep

  # 函数作用：由根辅助状态与局部表示恢复完整世界系状态。
  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    """由根辅助状态与局部表示恢复完整世界系状态。

    输入 aux:[...,7] 与 rep:[...,10+2J]，返回 [...,13+2J]。根 xy/四元数取 aux，
    高度以 rep 中 z 为准；机体系线/角速度旋转回世界系，再按完整状态约定拼接。
    用于可视化或完整状态消费者，不推进动力学。
    """
    z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]
    # Override root Z with rep_state's z
    # 完整状态的高度以 rep 中 z 为准；xy 和四元数来自 aux，再把机体系速度旋回世界系。
    root_pos = torch.cat([root_pos[..., :2], z], dim=-1)
    v_global = quat_apply(root_quat, v_b)
    w_global = quat_apply(root_quat, w_b)
    parts = [root_pos, root_quat, qj, v_global, w_global, vj]
    return torch.cat(parts, dim=-1)

  # 函数作用：拆出根位姿、关节位置与关节速度监督组。
  def loss_blocks(
    self, rep_state: torch.Tensor, aux_state: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """拆出根位姿、关节位置与关节速度监督组。

    输入 rep、aux 保留所有批次/时间轴；返回 root_pos、root_ori、joint_pos、joint_vel。
    高度、重力等状态不另列独立组，避免对根积分派生量重复计同类误差。
    具体权重和组宽度归一由 WorldModelLearner 处理。
    """
    # z, g_b and the root twist all follow from the integrated root pose.
    # 广义坐标的监督组使用根位姿、关节位置和关节速度；未列出的状态通道不直接计入损失。
    _, _, _, _, joint_pos, joint_vel = decompose_rep_state(rep_state)
    return {
      **_root_loss_blocks(aux_state),
      "joint_pos": joint_pos,
      "joint_vel": joint_vel,
    }

  # 函数作用：用正运动学把根与关节状态恢复为跟踪身体的世界状态。
  def world_frame_bodies(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """用正运动学把根与关节状态恢复为跟踪身体的世界状态。

    输入 aux/rep；先把机体系根线/角速度旋转至世界系，与根位姿、关节位置/速度
    一起输入绑定的 fk，返回 (pos,rot_matrix,lin_vel,ang_vel)。
    旋转输出从四元数转矩阵，以匹配统一奖励接口；函数保留可微运动学链路。
    """
    # 【源码讲解】
    # 由关节状态通过正运动学恢复跟踪刚体，再用于统一世界坐标奖励。
    # 关节空间动力学与身体空间奖励通过这条可微路径连接。
    _, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
    root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
    pos, quat, lin_vel, ang_vel = self.fk(
      root_pos,
      root_quat,
      qj,
      quat_apply(root_quat, v_b),
      quat_apply(root_quat, w_b),
      vj,
    )
    return pos, matrix_from_quat(quat), lin_vel, ang_vel

  # 函数作用：直接从广义坐标表示取关节位置和速度。
  def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    """直接从广义坐标表示取关节位置和速度。

    输入 [...,10+2J]，返回 [...,2J]=[qj,vj]。不需逆运动学，顺序与机器人关节列一致，
    不在此扣默认姿态；只有传感器合成路径需要相对关节位置。
    """
    _, _, _, _, qj, vj = decompose_rep_state(rep_state)
    return torch.cat([qj, vj], dim=-1)

  # 函数作用：从广义坐标预测合成本体传感器的状态部分。
  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    joint_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    """从广义坐标预测合成本体传感器的状态部分。

    参数 rep、joint_state 提供 g_b、w_b、关节位置和速度；aux/sim_ctx 保留统一接口
    但本实现不使用。返回 [重力3,角速度3,关节位置减默认姿态J,关节速度J]。
    上一动作和噪声在 compute_actor_sensor 中追加，不在这里重复处理。
    """
    # 广义坐标传感器合成直接从 rep 取重力与角速度，关节状态提供位置和速度。
    # 统一接口仍接收 aux/context，但本实现不使用；上一动作由调用方另行拼接。
    del aux_state, sim_ctx
    _, g_b, _, w_b, _, _ = decompose_rep_state(rep_state)
    qj, vj = joint_state.chunk(2, dim=-1)
    return torch.cat([g_b, w_b, qj - self._default_joint_pos, vj], dim=-1)


class MaxCoordRepresentation(StateRepresentation):
  """以跟踪刚体为核心的最大坐标表示。

  完整状态按位置3K、四元数4K、世界线速度3K、世界角速度3K排列，K 为刚体数。
  表示保存根系局部位置3K、六维旋转6K、线速度3K、角速度3K及根高度1、重力方向3；
  辅助状态是世界根位置与四元数，预测为每刚体线/角速度增量共6K。

  网络可以直接学习身体运动，关节观察与限位奖励则通过 AnalyticIK 恢复。
  绑定阶段还查询编码器偏置列，以使想象合成传感器与物理随机化条件一致。
  """

  LOSS_WEIGHT_CONFIG = {
    **StateRepresentation.LOSS_WEIGHT_CONFIG,
    "body_pos": "nominal_error_body_pos",
    "body_rot": "nominal_error_body_rot",
    "body_vel": "nominal_error_body_vel",
    "body_avel": "nominal_error_body_avel",
  }

  # 函数作用：声明刚体表示尺寸并注册传感器合成常量。
  def __init__(self, num_bodies: int, default_joint_pos: torch.Tensor):
    """声明刚体表示尺寸并注册传感器合成常量。

    参数 num_bodies=K、default_joint_pos:[J]；完整/表示/预测宽度分别由最大坐标
    工具推导，默认姿态与世界向下单位向量注册为非持久 buffer。
    返回 None；逆运动学在 bind 后才可用，编码器偏置映射初始为空。
    """
    super().__init__()
    # 这里 B 是刚体数，不是 batch；完整状态每刚体 13 维，预测每刚体 6 个速度增量。
    # 表示维数还包含六维旋转和根高度、重力方向，因此三种维数不能混用。
    self.num_bodies = num_bodies
    self.full_state_dim = maxcoord_full_state_dim(num_bodies)
    self.rep_state_dim = maxcoord_rep_state_dim(num_bodies)
    self.prediction_dim = maxcoord_prediction_dim(num_bodies)
    self.num_joints = default_joint_pos.shape[0]
    self.register_buffer("_default_joint_pos", default_joint_pos, persistent=False)
    self.register_buffer("_gravity_w", torch.tensor([0.0, 0.0, -1.0]), persistent=False)
    self.analytic_ik: AnalyticIK  # set by bind
    self._encoder_bias: tuple[slice, SimContext] | None = None

  # 函数作用：把最大坐标表示绑定到机器人逆运动学与可选编码器偏置。
  def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
    """把最大坐标表示绑定到机器人逆运动学与可选编码器偏置。

    参数 env、sim_context；先绑定软关节限位，再将 MuJoCo 模型、资产刚体索引、
    关节地址与跟踪刚体顺序交给 AnalyticIK。查询上下文 encoder_bias 的列切片，
    启用时缓存切片和上下文对象，供传感器合成反归一化使用。
    返回 None；这一步必须在计算 joint_state 前完成。
    """
    super().bind(env, sim_context)
    robot = env.scene["robot"]
    motion = env.command_manager.get_term("motion")
    # 绑定 MuJoCo 模型、刚体与关节地址以及跟踪刚体顺序，供最大坐标恢复关节状态。
    # 若调整跟踪刚体列表，逆运动学输入顺序也必须一致。
    self.analytic_ik = AnalyticIK(
      env.sim.mj_model,
      robot.indexing.body_ids,
      robot.indexing.joint_q_adr,
      motion.body_indexes,
      self._default_joint_pos.device,
      self._default_joint_pos.dtype,
    )
    # 查询编码器偏置是否存在及其列切片，后续用反归一化恢复物理偏置值。
    columns = sim_context.find("encoder_bias")
    self._encoder_bias = None if columns is None else (columns, sim_context)

  # 函数作用：把完整世界系刚体状态转换为根系模型表示。
  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    """把完整世界系刚体状态转换为根系模型表示。

    输入 full_state:[...,13K]，合并前导轴后调用最大坐标转换工具，恢复为
    [...,15K+4]。保留训练时间轴和批次轴，不修改原始完整状态。
    根世界位姿另存 aux，局部特征与辅助状态共同定义后续积分。
    """
    flat = full_state.reshape(-1, self.full_state_dim)
    rep = compute_maxcoord_rep_state(flat, self.num_bodies)
    return rep.reshape(*full_state.shape[:-1], -1)

  # 函数作用：从按字段排列的最大坐标完整状态提取根位姿。
  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    """从按字段排列的最大坐标完整状态提取根位姿。

    输入 [...,13K]；根位置是前三列，根四元数位于全部位置3K列之后的前四列。
    返回 [...,7]。不能直接取前七列，因为那里还包含其他刚体的位置。
    """
    B = self.num_bodies
    return torch.cat([full_state[..., :3], full_state[..., B * 3 : B * 3 + 4]], dim=-1)

  # 函数作用：积分逐刚体速度增量，并把结果转换到下一根坐标系。
  def apply_prediction(
    self,
    aux_state: torch.Tensor,
    curr_rep_state: torch.Tensor,
    prediction: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """积分逐刚体速度增量，并把结果转换到下一根坐标系。

    输入 aux:[...,7]、rep:[...,15K+4]、prediction:[...,6K]、dt；返回
    (next_aux,next_rep)。在当前根系更新线/角速度，采用子步系数修正的速度积分
    相对位置和旋转；根的整体位姿单独在世界系推进。

    最后施加旧根到新根的逆向坐标旋转到所有局部位置、旋转列与速度，再重算
    根高度和重力方向。局部根位置漂移先被扣除，避免积累到相对身体布局中。
    旋转的 Rodrigues 预计算复用于六维表示两列，整个链路保持可微。
    """
    # 【源码讲解】
    # 每个刚体在线/角速度空间更新，再积分局部位置与旋转。
    # 所有预测先以当前根坐标系表达，最后转换到下一根坐标系。
    # 返回辅助根位姿与模型表示，保持下次调用的坐标约定一致。
    B = self.num_bodies
    c = self.substep_coeff
    root_pos = aux_state[..., :3]
    root_quat = aux_state[..., 3:7]

    local_pos, local_rot_6d, local_vel, local_avel, _, _ = decompose_maxcoord_rep_state(
      curr_rep_state, B
    )

    # MaxCoord 对每个 body 预测线速度/角速度增量，而非逐关节增量。
    # Prediction deltas (in root body frame)
    dv_local = prediction[..., : B * 3].unflatten(-1, (B, 3))
    dw_local = prediction[..., B * 3 : B * 6].unflatten(-1, (B, 3))

    # --- Step 1: Apply deltas in local frame ---
    next_vel = local_vel + dv_local
    next_avel = local_avel + dw_local
    pose_vel = local_vel + c * dv_local
    pose_avel = local_avel + c * dw_local

    # --- Step 2: Integrate in current root frame ---
    # Positions (with drift correction: subtract root_pos_local)
    # 局部位置可能存在数值漂移；先减去根局部位置，根的相对位置重新回到零。
    # 身体平移只积分相对根线速度，根的整体平移单独在世界系更新。
    root_pos_local = local_pos[..., 0:1, :]
    rel_vel = pose_vel - pose_vel[..., 0:1, :]
    next_pos = (local_pos - root_pos_local) + rel_vel * dt
    # Rotations (per-body omega, shared trig for 2 rot6d cols)
    # 角速度乘 dt 得旋转向量，预计算 Rodrigues 旋转项并复用于六维旋转的两列。
    k_body, cos_body, sin_body = rodrigues_precompute(pose_avel * dt)
    cols = local_rot_6d.unflatten(-1, (3, 2))
    c1_mid = rodrigues_apply(cols[..., 0], k_body, cos_body, sin_body)
    c2_mid = rodrigues_apply(cols[..., 1], k_body, cos_body, sin_body)

    # --- Step 3: Root pose in world frame (with drift correction) ---
    next_root_pos = root_pos + quat_apply(
      root_quat,
      root_pos_local.squeeze(-2) + pose_vel[..., 0, :] * dt,
    )
    next_root_quat = quat_integrate(
      root_quat, quat_apply(root_quat, pose_avel[..., 0, :]), dt
    )

    # --- Step 4: Frame change (old root → new root) ---
    # 坐标轴随根旋转后，所有局部向量要施加相反旋转，才仍表达于新的根坐标系。
    frame_omega = -pose_avel[..., 0:1, :] * dt
    k_f, cos_f, sin_f = rodrigues_precompute(frame_omega)

    next_local_pos = rodrigues_apply(next_pos, k_f, cos_f, sin_f)
    next_local_vel = rodrigues_apply(next_vel, k_f, cos_f, sin_f)
    next_local_avel = rodrigues_apply(next_avel, k_f, cos_f, sin_f)
    next_local_rot_6d = torch.stack(
      [
        rodrigues_apply(c1_mid, k_f, cos_f, sin_f),
        rodrigues_apply(c2_mid, k_f, cos_f, sin_f),
      ],
      dim=-1,
    ).flatten(-2, -1)

    # --- Step 5: Aux state & trailing fields ---
    world_down = torch.zeros_like(next_root_pos)
    world_down[..., 2] = -1.0
    next_up_dir = quat_apply_inverse(next_root_quat, world_down)

    next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
    next_rep = torch.cat(
      [
        next_local_pos.flatten(-2, -1),
        next_local_rot_6d.flatten(-2, -1),
        next_local_vel.flatten(-2, -1),
        next_local_avel.flatten(-2, -1),
        next_root_pos[..., 2:3],  # root_height
        next_up_dir,
      ],
      dim=-1,
    )
    return next_aux, next_rep

  # 函数作用：委托最大坐标工具恢复世界系完整身体状态。
  def reconstruct_full_state(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> torch.Tensor:
    """委托最大坐标工具恢复世界系完整身体状态。

    参数 aux、rep 及实例保存的 K；返回 [...,13K]，遵循完整状态字段顺序。
    本包装器不调用动力学、不更新参考时刻，常供可视化完整状态路径使用。
    """
    # 委托最大坐标工具恢复世界系完整状态；传入刚体数决定各段的切分宽度。
    return reconstruct_full_from_aux_and_rep(aux_state, rep_state, self.num_bodies)

  # 函数作用：生成根位姿及非根刚体的位置、旋转和速度监督组。
  def loss_blocks(
    self, rep_state: torch.Tensor, aux_state: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """生成根位姿及非根刚体的位置、旋转和速度监督组。

    输入 rep、aux；返回 root_pos/root_ori 与 body_pos/body_rot/body_vel/body_avel。
    body 0 是根，其局部位姿及根派生字段不在身体组重复计入；身体各组取1号以后
    并展平刚体与通道轴。具体均方误差和尺度权重由 learner 统一计算。
    """
    # Body 0 is the root: its pose is identically zero/identity and its twist is
    # the root twist. root_height/up_dir also follow from the root pose. All of
    # them are scored by root_pos/root_ori instead.
    # 最大坐标损失从逐刚体通道取组，下面切掉 body 0。
    # 根相对位姿为零/单位旋转，高度和方向由积分根位姿导出，根位姿使用独立误差组。
    pos, rot_6d, vel, avel, _, _ = decompose_maxcoord_rep_state(
      rep_state, self.num_bodies
    )
    return {
      **_root_loss_blocks(aux_state),
      # 跳过根刚体，展平其余刚体轴和坐标轴；同样规则应用于方向及速度组。
      "body_pos": pos[..., 1:, :].flatten(-2, -1),
      "body_rot": rot_6d[..., 1:, :].flatten(-2, -1),
      "body_vel": vel[..., 1:, :].flatten(-2, -1),
      "body_avel": avel[..., 1:, :].flatten(-2, -1),
    }

  # 函数作用：将根系身体表示恢复为统一奖励接口的世界系状态。
  def world_frame_bodies(
    self, aux_state: torch.Tensor, rep_state: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将根系身体表示恢复为统一奖励接口的世界系状态。

    输入 aux/rep；调用最大坐标工具并传入实例 K，返回位置、旋转矩阵、线速度、角速度。
    这是坐标恢复而非正运动学拟合，输出身体顺序保持跟踪列表顺序。
    """
    return world_frame_bodies(aux_state, rep_state, self.num_bodies)

  # 函数作用：通过解析逆运动学恢复关节位置和速度。
  def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
    """通过解析逆运动学恢复关节位置和速度。

    输入 rep:[...,15K+4]，取各身体六维旋转和角速度，旋转转矩阵后交给已绑定
    AnalyticIK，拼接其两项输出为 [...,2J]。用于传感器合成及关节限位奖励。
    调用前必须 bind；身体顺序与索引改变会影响恢复语义。
    """
    # 【源码讲解】
    # 最大坐标表示不直接存关节状态，需要经绑定的解析逆运动学恢复。
    # 恢复结果用于关节限位和合成传感器，因此刚体顺序和机器人索引必须匹配。
    _, rot6d, _, avel, _, _ = decompose_maxcoord_rep_state(rep_state, self.num_bodies)
    return torch.cat(self.analytic_ik(rot6d_to_matrix(rot6d), avel), dim=-1)

  # 函数作用：从预测身体状态合成含随机化偏置的关节传感器部分。
  def compute_wm_obs(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    joint_state: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> torch.Tensor:
    """从预测身体状态合成含随机化偏置的关节传感器部分。

    参数 rep、aux、joint_state、sim_ctx；先把世界向下向量旋到根系取得重力，
    读取根角速度，以关节位置减默认姿态生成相对位置。若启用 encoder_bias，
    从上下文反归一化偏置并叠加，再拼接关节速度。

    返回 [...,6+2J]，不含上一动作和额外噪声；外层 compute_actor_sensor 负责两者。
    这条路径使最大坐标模型内部不存关节状态，也能向同一 actor 提供传感器接口。
    """
    root_quat = aux_state[..., 3:7]
    # 将单位向下向量广播到 batch，再旋转到当前根坐标系，合成重力方向观察。
    g_w = self._gravity_w.expand(*aux_state.shape[:-1], 3)
    projected_gravity = quat_apply_inverse(root_quat, g_w)
    _, _, _, avel, _, _ = decompose_maxcoord_rep_state(rep_state, self.num_bodies)
    joint_pos, joint_vel = joint_state.chunk(2, dim=-1)
    joint_pos_rel = joint_pos - self._default_joint_pos
    # 只有启用编码器偏置随机化时才恢复并叠加实际偏置，使合成关节观察匹配传感器条件。
    if self._encoder_bias is not None:
      columns, sim_context = self._encoder_bias
      joint_pos_rel = joint_pos_rel + sim_context.denormalize(sim_ctx, columns)
    return torch.cat([projected_gravity, avel[..., 0, :], joint_pos_rel, joint_vel], -1)
