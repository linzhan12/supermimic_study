# F2 支撑期定足的全身斜走参考

## 假设与方法

直接跟踪已有低摆臂直走周期时，改用持续的斜走落脚点会补充原动作库稀缺的斜走候选。原 G1 策略和世界模型权重固定。选用已实测的 `f2-sonic-bank-low-arm-55/slow-a.npz`，保留骨盆姿态、躯干和全部上肢关节。骨盆增加横向速度 (v_y)，每只脚在支撑时保持相对原轨迹的世界坐标横移恒定；离地后以 (10u^3-15u^4+6u^5) 把横向落点前移 (v_yT)，其中 (T) 为步态周期。每帧通过 MuJoCo 双脚 IK 匹配落点与原脚底朝向，使用周期中心差分更新腿部关节速度。筛除足间距低于 8 cm、IK 不收敛或关节改动超过 0.65 rad 的候选。

在线对照中的 `--early-steering-handoff True` 默认关闭。启用时允许已经运行至少 0.28 s 的方向启动片段在测得支撑且实际速度已经沿指令方向增长后，提前尝试交给实测支撑接缝。候选仍以原基线和实测接触状态排序。只改变 F2 规划入口。

## 同一冻结策略的固定参考测量

5 s 回放取末 4 s，速度在机器人航向坐标系下统计。摆臂为左右肘关节第 5 至第 95 百分位差。

| 参考横速 m/s | 骨盆降低 m | 实测前向 m/s | 实测横向 m/s | 实测偏航 rad/s | 完整摆步 左/右 | 肘幅 左/右 |
| --- | --- | --- | --- | --- | --- | --- |
| +0.08 | 0 | 0.212 | +0.109 | −0.023 | 2/5 | 9.2°/9.7° |
| +0.14 | 0 | 0.202 | +0.150 | −0.026 | 2/3 | 9.6°/12.6° |
| −0.08 | 0 | 0.219 | −0.022 | −0.015 | 1/3 | 10.8°/7.2° |
| −0.14 | 0 | 0.216 | −0.080 | −0.010 | 2/4 | 11.8°/7.9° |
| −0.20 | 0 | 0.214 | −0.113 | −0.006 | 2/3 | 12.2°/7.4° |
| +0.20 | 0.02 | 0.168 | +0.200 | −0.037 | | |
| −0.23 | 0.02 | 0.166 | −0.148 | −0.011 | | |

左侧 +0.20 m/s 固定周期的横速已达到命令值，前速与完整摆步仍需验收。参考和固定跟踪的右侧响应不对称；左右偏差不能从参考速度单独判断。未降低骨盆的 +0.20 m/s 候选在第 38 帧不收敛，中档直走生成的 ±0.14 m/s 候选足间距不足。

## 在线换向与末次松杆

侧向命令目标为 ((0.25,\pm0.20,0))，分别取主场景第 325–374、375–424 控制步。留出使用另一受试者的物理初态。末次松杆记录最后 25 步平面速度模长均值。

| 参考库 / 初态 | 左斜走 `vx, vy, wz` | 右斜走 `vx, vy, wz` | 末次松杆 m/s |
| --- | --- | --- | --- |
| 原方向片段加新周期 / 主 | 0.157, +0.062, +0.155 | 0.276, −0.247, −0.133 | 0.061 |
| 原方向片段加新周期 / 留出 | 0.154, +0.062, +0.162 | 0.271, −0.246, −0.140 | 0.097 |
| 用新周期替换旧斜走稳态 / 主 | 0.157, +0.062, +0.155 | 0.229, −0.187, −0.069 | 0.061 |
| 用新周期替换旧斜走稳态 / 留出 | 0.150, +0.040, +0.170 | 0.248, −0.141, −0.069 | 0.057 |
| 混合库、提前衔接 / 主 | 0.163, +0.069, +0.131 | 0.256, −0.248, −0.128 | 0.091 |
| 混合库、提前衔接 / 留出 | 0.150, +0.040, +0.170 | 0.302, −0.231, −0.144 | 0.048 |
| 新周期加降骨盆候选、提前衔接 / 主 | 0.163, +0.069, +0.131 | 0.181, −0.178, −0.045 | 0.060 |
| 新周期加降骨盆候选、提前衔接 / 留出 | 0.150, +0.040, +0.170 | 0.224, −0.166, −0.071 | 0.062 |

右侧候选确实被选中。在主初态替换旧稳态后，横速接近目标，偏航绝对值也较混合库更小。左侧主初态仅末 5 步进入 `cycle-left`，留出连续 50 步使用 `start-left-b`，加大稳态横速没有解决入口。应固定相同初态和命令比较如何交接左侧支撑分支。F2 继续判定为未通过。

## 复现

从 `supermimic` 目录运行，示例生成 +0.20 m/s 参考、执行原策略和打包真实轨迹。其他候选按表替换速度和骨盆高度。每条候选均先在物理中回放，再用对应轨迹打包参考与跟踪偏差。

```bash
export OMP_NUM_THREADS=2 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/supermimic-f2-mpl
checkpoint=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
.venv/bin/python scripts/retarget_f2_diagonal_cycle.py \
  outputs/f2-sonic-bank-low-arm-55/slow-a.npz \
  outputs/f2-sonic-retargeted-diagonal/slow-left-20d.npz \
  --lateral-speed 0.20 --pelvis-drop 0.02
.venv/bin/python scripts/eval_f2.py --checkpoint-dir "$checkpoint" \
  --motion-file outputs/f2-sonic-retargeted-diagonal/slow-left-20d.npz \
  --motion-only True --motion-window-s 5 --record-posture True \
  --output-dir outputs/f2-sonic-diag-cycle-slow-left-20d
.venv/bin/python scripts/pack_f2_bank.py --output outputs/f2-sonic-pack-left-20d \
  --entry cycle-left-strong walk \
  outputs/f2-sonic-diag-cycle-slow-left-20d/reference_0_motion.jsonl 0 250
```

在线库基于低摆臂实验库，主复现参数为 `--trackability-weight 2 --prediction-steps 20 --pace-margin 0.25 --candidate-entries 3 --restart-from-rest True --directional-selection True --steering-measured-seam True --project-contact-seam True --rank-contact-seam True --adapt-ranking True --adapter-steering-only True --diagonal-heading-lock True --contact-stop-capture True --direction-cost-weight 4`。适配器使用 `outputs/f2-sonic-adapter-06/contact_adapter.pt`。启动启用 `--start-upper-blend 0.48 --start-search-time 0.54 --contact-launch-duration 0.60 --rank-contact-launch True --contact-launch-forward-lead 0.08`；停步使用 `--contact-stop-duration 0.40 --contact-stop-max-foot-speed 0.25`。原始结果在 `outputs/f2-sonic-retargeted-diagonal-{online,early,strong}-*`，取 `stand_start_planned.jsonl`。留出使用 `walk1_subject2.npz` 的第 50 和 231 帧作为站立和行走初态。
