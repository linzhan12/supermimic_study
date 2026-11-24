# F2.4 镜像全身参考的方向覆盖验证

将完整参考关于矢状面反射，左右关节交换，各轴符号由 G1 模型在零关节状态的转轴确定。设 $M=\operatorname{diag}(1,-1,1)$，根位置与线速度方向按 $M$ 反射，旋转按 $R'=MRM$，角速度按 $-M$ 反射。两侧脚的正运动学反射误差小于 $5\times10^{-8}$ m。每段作为独立完整参考闭环执行。

| 镜像参考 | 原动作 / 起帧 | 实测前向 / 横向 / 偏航 |
| --- | --- | ---: |
| 左斜走 A | walk1_subject2 / 3915 | +0.309 / +0.260 / −0.086 |
| 左斜走 B | walk3_subject2 / 9253 | +0.287 / +0.207 / −0.072 |
| 右斜走 | walk2_subject4 / 8997 | +0.264 / −0.101 / −0.132 |
| 左转 | walk1_subject1 / 2326 | +0.436 / −0.012 / +0.431 |
| 左起步末 0.4 s | walk2_subject3 / 2153 | +0.220 / +0.130 / +0.146 |
| 右起步末 0.4 s | walk3_subject3 / 132 | +0.420 / −0.155 / −0.040 |

速度单位为 m/s、m/s、rad/s。两条镜像左斜走偏航绝对值的第 90 百分位为 1.77、1.04 rad/s，均值掩盖了转头和回摆。保留两条起步和较稳定的右斜走做联合对照。主初态未选新候选，动作和速度与原航向组相同。留出初态选中镜像左起步，左斜走侧向由 −0.021 转为 +0.065 m/s，但前向为 −0.027 m/s，换右时仍延续该起步。末次松杆末半秒从 0.025 恶化至 0.553 m/s。增加几何对称片段可以改变方向，控制仍缺少可接续的后续支撑阶段。**F2 未通过。**

## 复现

从 `supermimic/` 运行。各行原片段从表中帧前 10 帧到帧后 160 帧截取为镜像。固定闭环从镜像文件第 10 帧开始，起步 1.5 s，其他 2 s。下例为右斜走。

~~~bash
.venv/bin/python scripts/mirror_f2_motion.py ../data/lafan1_official/walk2_subject4.npz \
  outputs/f2-sonic-mirrored-motion/diagonal-right.npz --first 8987 --stop 9157
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python -u scripts/eval_f2.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file outputs/f2-sonic-mirrored-motion/diagonal-right.npz \
  --motion-only True --motion-window-s 2 --motion-frames '(10,)' --record-posture True \
  --output-dir outputs/f2-sonic-mirror-replay-diagonal-right
~~~

新库 `outputs/f2-sonic-bank-mirror-screened` 复制原联合库，并按 `pack_f2_bank.py` 的 `0 100` 和 `0 75` 分别增加镜像右斜走及两条镜像起步。沿用 [斜走航向实验](2026-11-25-f2-sonic-command-scoped-tracking.md) 适配预演，换新库，结果为 `outputs/f2-sonic-mirror-{main,holdout}-adapt`。
