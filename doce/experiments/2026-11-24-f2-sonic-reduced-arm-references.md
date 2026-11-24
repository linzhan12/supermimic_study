# F2.3 保留肩肘节律的参考摆幅对照

在原单周期库中，快档规划闭环右肘的 5–95 百分位摆幅为 35.0°，其相近实测速率的完整参考约 34.9°。本实验仅改参考库的肩肘周期分量。令 $\bar q_j$ 为首个周期的关节均值，使用 $q'_j=\bar q_j+s(q_j-\bar q_j)$、$\dot q'_j=s\dot q_j$，保持关节节律和均值。冻结模型、足位、骨盆及起步和停步文件保持一致。

| 参考幅度比例 | 稳态慢 / 中 / 快（m/s） | 快档左 / 右肩俯仰摆幅 | 快档左 / 右肘摆幅 | 直走和换速筛查失败数 |
| --- | ---: | ---: | ---: | ---: |
| 1.00 | 0.242 / 0.358 / 0.475 | 11.9° / 20.7° | 22.8° / 35.0° | 0 |
| 0.55 | 0.238 / 0.364 / 0.465 | 9.8° / 12.7° | 16.9° / 26.9° | 0 |
| 0.30 | 0.236 / 0.366 / 0.462 | 7.0° / 8.7° | 14.3° / 23.6° | 2 |

0.55 组保留双脚交替和连续三档数值筛查，快档右肘摆幅降低 8.1°。正、侧面对照仍能看到右肘的周期活动。0.30 组进一步减小参考幅度，实际肘关节少动的量开始收窄，同时快档左右完整摆步为 1 / 2，快档换速末半秒侧向达到 +0.112 m/s。仅降低肩肘的参考振幅不能独立控制实际摆臂及平衡。保留 0.55 组作为联合对照，**本实验未完成 F2**。

## 复现

从 `supermimic/` 运行，第二组把 `n=55`、`scale=0.55` 改为 30 和 0.30。只评估此次受影响的四个直走用例。

~~~bash
n=55
scale=0.55
.venv/bin/python scripts/scale_f2_reference_arms.py \
  outputs/f2-sonic-bank-single-pace "outputs/f2-sonic-bank-low-arm-$n" --scale "$scale"
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python -u scripts/eval_f2.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --record-posture True --planned-only True \
  --cases "('steady_forward_0p220','steady_forward','steady_forward_0p500','multispeed_profile')" \
  --reference-bank "outputs/f2-sonic-bank-low-arm-$n" --trackability-weight 2 \
  --prediction-steps 20 --pace-margin 0.25 --steady-forward True \
  --extra-forward-speeds '(0.22,0.50)' --multispeed-profile True \
  --output-dir "outputs/f2-sonic-low-arm-$n"
~~~

筛查用 `check_f2_gait.check_gait`，统计留在各目录的 `gait-only.json`。仅使用 `audit_forward_gait.py` 离线回放 0.55 组，与[原单周期同速回放](2026-11-24-f2-sonic-axial-regression.md)用相同完整参考和区间。三个 GIF 与同名 JSON 保存在 `outputs/f2-sonic-low-arm-55/paired-{slow,medium,fast}.*`。
