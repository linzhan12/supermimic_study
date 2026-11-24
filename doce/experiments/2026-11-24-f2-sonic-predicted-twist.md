# F2.3 对实际指令误差评分的世界模型预演

保持冻结 checkpoint、同一短起步库及 0.48 s 上身接缝，禁用上轮方向类别过滤。世界模型仍比较预测骨盆、姿态和参考。增加一个单独可消融的指令损失，从预测骨盆位姿差计算指令时刻的体轴平移和世界竖直轴偏航。先对预演后 2/3 时间平均，减少换重造成的瞬时速度摆动，再将相对真实指令的残差按 `0.18 m/s、0.16 m/s、0.50 rad/s` 缩放，求平滑 L1 和。控制模型的实际指令响应与参考跟踪误差共同参与排序。每条参考可选多个入点，排除距离已选入点 0.10 s 内的相邻点。

| 指令损失权重 / 每条入点 | 左转实际偏航 | 右转实际偏航 | 右转偏航 MAE | 首次松杆前向均值 |
| --- | ---: | ---: | ---: | ---: |
| 0 / 1 | +0.049 | −0.101 | 0.407 | 0.070 |
| 0.8 / 1 | +0.024 | −0.053 | 0.510 | 0.337 |
| 3.0 / 1 | +0.024 | −0.053 | 0.510 | 0.337 |
| 1.2 / 3 | +0.010 | −0.246 | 0.532 | 0.055 |
| 3.0 / 3 | +0.072 | −0.356 | 0.465 | 0.052 |

偏航单位 rad/s，前向速度单位 m/s。入点只有一个时，增加对指令的权重让控制一直走过旧转向。三个入点提供了可选的过渡相，在此例改善右转的实际均值，并重新接到录制停步。右转偏航 MAE 仍升高，左转和斜走也没有通过。加入指令误差的效果依赖候选覆盖，不能用改善的右转均值代替完整 F2。

收步后的左斜走接缝是主要缺口。末次静止时前行速度约为零，双脚均接触，但左斜走原片段最佳下肢姿态差约 0.60 rad，右斜走同一幅度约 0.26 rad。下一轮需补足方向转换的起步相位，保持直接镜像来源和实际仿真校验的区别。

## 复现

从 `supermimic/` 运行，使用 [短起步库](2026-11-24-f2-sonic-start-support.md)。

~~~bash
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
for item in '08 0.8 1' '3 3 1' '12-e3 1.2 3' '3-e3 3 3'; do
  read -r name weight entries <<< "$item"
  .venv/bin/python -u scripts/eval_f2.py \
    --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
    --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
    --cases '("stand_start",)' --reference-bank outputs/f2-sonic-bank-short-start \
    --planned-only True --trackability-weight 2 --prediction-steps 20 --pace-margin 0.25 \
    --start-upper-blend 0.48 --command-cost-weight "$weight" --candidate-entries "$entries" \
    --output-dir "outputs/f2-sonic-command-$name"
done
~~~
