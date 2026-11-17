"""可微世界模型轨迹的身体动作跟踪奖励。

文件位置：supermimic/src/supermimic/utils/tracking_reward.py。
输入统一为世界系预测/参考身体状态；通过重锚定处理身体相对姿态跟踪，同时保留
独立根位置、根方向项。六项正奖励采用 weight*exp(-err_sq/std²)，再减动作变化
及关节超限惩罚，最终乘控制步长 dt。奖励保持对预测状态和动作可微，供 BPTT 使用。
本模块不推进动力学、不判定回合结束、不更新归一化或模型参数。
"""

# 【文件作用】想象轨迹的动作跟踪奖励
# 比较预测身体状态与参考动作，组合位置、方向、速度跟踪项以及动作变化和关节限位惩罚。
# 【阅读顺序】重锚定参考身体布局 → 计算各组误差 → 高斯核奖励 → 惩罚与时间步缩放。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


from __future__ import annotations



# PyTorch 提供可微的误差、指数核与限位惩罚计算，使梯度可经奖励回传策略。
import torch
# mjlab 旋转工具：四元数转矩阵、求逆、复合及提取偏航分量，用于参考身体重锚定。
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul, yaw_quat

# 来自 supermimic/src/supermimic/utils/helpers.py：rotation_error_magnitude 计算弧度角距离；
# small_matmul 执行小矩阵乘法，避免普通 TF32 路径的舍入并利于编译融合。
from supermimic.utils.helpers import rotation_error_magnitude, small_matmul


# 函数作用：按预测根的平面位置和相对偏航重新放置参考身体布局。
def _reanchor_targets(
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  pred_root_pos: torch.Tensor,
  pred_root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """按预测根的平面位置和相对偏航重新放置参考身体布局。

  target_pos:[...,K,3]、target_quat:[...,K,4] 是参考世界位姿，K 为跟踪刚体数且
  第 0 个是根；pred_root_pos:[...,3]、pred_root_quat:[...,4] 是预测根位姿。
  四元数沿用 mjlab 的 (w,x,y,z) 约定。批次前缀可广播，当前调用通常为单批次轴 B。

  先从参考身体减去参考根位置，再用“预测根相对参考根旋转的偏航分量”旋转布局。
  新平移锚点采用预测根 xy 与参考根 z；方向也左乘该偏航差旋转。
  返回 tgt_pos:[...,K,3] 及 tgt_mat:[...,K,3,3]，方向由四元数转为旋转矩阵。
  不改输入张量，不重锚定速度；根的全局误差在 tracking_reward 中另行评分。
  """
  # 【源码讲解】
  # 相对刚体布局围绕参考根展开，但平移 xy 和偏航采用预测根。
  # 参考高度及侧倾/俯仰信息仍保留，返回目标位置与旋转矩阵供局部身体项评分。
  anchor_pos = target_pos[..., 0:1, :]

  # 目标刚体相对布局跟随预测根部的平面位置和偏航，高度仍来自参考根部。
  delta_pos = torch.cat([pred_root_pos[..., :2], target_pos[..., 0, 2:3]], dim=-1)
  delta_mat = matrix_from_quat(
    yaw_quat(quat_mul(pred_root_quat, quat_inv(target_quat[..., 0, :])))
  ).unsqueeze(-3)

  # 先移除参考根平移，再施加偏航差旋转并加回新锚点位置，保留身体相对布局。
  rel_pos = target_pos - anchor_pos
  tgt_pos = (
    delta_pos.unsqueeze(-2) + small_matmul(delta_mat, rel_pos.unsqueeze(-1))[..., 0]
  )
  tgt_mat = small_matmul(delta_mat, matrix_from_quat(target_quat))
  return tgt_pos, tgt_mat


# 函数作用：比较预测状态与参考身体运动，返回逐样本可微跟踪奖励。
def tracking_reward(
  aux_state: torch.Tensor,
  pred_pos: torch.Tensor,
  pred_rot_mat: torch.Tensor,
  pred_lin_vel: torch.Tensor,
  pred_ang_vel: torch.Tensor,
  joint_state: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  target_pos: torch.Tensor,
  target_quat: torch.Tensor,
  target_lin_vel: torch.Tensor,
  target_ang_vel: torch.Tensor,
  soft_joint_pos_limits: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  """比较预测状态与参考身体运动，返回逐样本可微跟踪奖励。

  设 B 为批大小、K 为跟踪刚体数、J 为关节数、A 为动作宽度。aux_state:[B,7]
  包含预测根位置和四元数；pred_pos/pred_lin_vel/pred_ang_vel:[B,K,3]，
  pred_rot_mat:[B,K,3,3]。target_pos/lin_vel/ang_vel 同为 [B,K,3]，
  target_quat:[B,K,4]；参考应与本步预测评分时刻对齐。
  joint_state:[B,2J] 按关节位置、速度拼接；action 与 last_action:[B,A] 是本步及
  上一原始策略输出。soft_joint_pos_limits:[J,2] 给软下上界，dt 是控制步秒数。

  先重锚定身体参考，随后计算六组平方误差：全局根位置、全局根方向、身体位置、
  身体方向、世界线速度、世界角速度。身体项对刚体取均值，根项单独保留权重。
  各项变成正的高斯核分数后相加，再减 0.1*动作差平方和与 2.0*关节超限距离和，
  最后乘 dt，返回 [B]。关节速度未用于限位惩罚，但 joint_state 仍按统一接口传入。

  函数仅计算奖励，不修改状态或保存统计；保持张量计算图，让策略梯度能穿过奖励
  和身体状态回到动作。该值是每控制步奖励，不是整个回合回报。
  """
  # 【源码讲解】
  # 所有输入对应预测后的身体状态及缓存参考，返回逐环境标量 [B]。
  # 根项直接对参考根评分，身体位置方向项对重锚定目标评分。
  # 指数跟踪项为正；动作差平方和与关节软限外距离为负惩罚，最后乘 dt。
  tgt_pos, tgt_mat = _reanchor_targets(
    target_pos, target_quat, aux_state[..., :3], aux_state[..., 3:7]
  )

  # 每个元组依次为平方误差、奖励权重、误差尺度 std。
  # 根项直接对原参考评分，身体位姿项用重锚定目标，速度项使用参考世界系速度。
  terms: list[tuple[torch.Tensor, float, float]] = [
    (torch.sum((pred_pos[..., 0, :] - target_pos[..., 0, :]) ** 2, dim=-1), 0.5, 0.3),
    (
      rotation_error_magnitude(
        pred_rot_mat[..., 0, :, :], matrix_from_quat(target_quat[..., 0, :])
      )
      ** 2,
      0.5,
      0.4,
    ),
    (torch.sum((pred_pos - tgt_pos) ** 2, dim=-1).mean(-1), 1.0, 0.3),
    ((rotation_error_magnitude(pred_rot_mat, tgt_mat) ** 2).mean(-1), 1.0, 0.4),
    (torch.sum((pred_lin_vel - target_lin_vel) ** 2, dim=-1).mean(-1), 1.0, 1.0),
    (torch.sum((pred_ang_vel - target_ang_vel) ** 2, dim=-1).mean(-1), 1.0, 3.14),
  ]
  # 误差越小，指数项越接近 1；加权后再扣除动作突变和关节超限。
  reward = sum(w * torch.exp(-err / std**2) for err, w, std in terms)
  # 对全部动作通道平方差求和得到 [B]，强调连续控制平滑性；不是用 dt 计算显式导数。
  action_rate = torch.sum((action - last_action) ** 2, dim=-1)
  # 统一状态布局是 [位置J,速度J]；限位项只比较位置，速度部分在这里不使用。
  joint_pos, _ = joint_state.chunk(2, dim=-1)
  # 沿最后轴拆出关节下限和上限；clamp(min=0) 只保留超出范围的距离，范围内为零。
  lo, hi = soft_joint_pos_limits.unbind(-1)
  joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
  # dt 是一个控制步的秒数，使每步奖励随控制时间尺度缩放。
  return dt * (reward - 0.1 * action_rate - 2.0 * joint_limit)
