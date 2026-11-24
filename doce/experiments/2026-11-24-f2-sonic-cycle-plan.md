# F2.2 周期库在线跟踪与世界模型消融

## 对照

将已用冻结 actor 跑完 7 s 的周期加入参考库。候选的步态速度按整个周期取均值，结合固定回放的实际偏差估计。在一个周期内选择入点，避免只因局部加速就把慢档判断成快档。起步与停步继续沿用已测片段。世界模型组在同一批候选上展开 20 步物理模型预测，覆盖 160 ms 接缝及其后约 240 ms。跟踪器、起始参考和命令序列均相同。

## 现象及测量

固定速度各持续 6 s，取后 4 s。速度列单位为 m/s、m/s、rad/s。

| 指令 | 运动学组前向 / 侧向 / 偏航 | 预演组前向 / 侧向 / 偏航 | 左右有效摆动，运动学 → 预演 |
| ---: | --- | --- | --- |
| 0.22 | 0.200 / +0.069 / −0.023 | 0.225 / +0.048 / −0.016 | 3 / 2 → 2 / 1 |
| 0.35 | 0.350 / +0.008 / +0.017 | 0.363 / +0.018 / −0.021 | 2 / 2 → 2 / 2 |
| 0.50 | 0.472 / +0.041 / +0.017 | 0.483 / +0.037 / +0.008 | 3 / 3 → 2 / 2 |

相对短片段版，快档运动学组支撑膝由 `31.7° / 41.0°` 降至 `20.4° / 27.3°`，摆动事件由 `10 / 7` 降至 `3 / 3`，偏航降低。世界模型排序这次改变了候选和实际运动。慢档侧向均值降低 `0.021 m/s`，同时右脚完整摆动减少至一次，所以两组各有失败项目。

| 连续段（末半秒） | 运动学组前向 / 侧向 / 偏航 | 预演组前向 / 侧向 / 偏航 |
| --- | --- | --- |
| 0.22 | 0.155 / +0.111 / −0.112 | 0.159 / +0.087 / −0.045 |
| 0.50 | 0.482 / +0.136 / −0.014 | 0.443 / +0.081 / −0.024 |
| 0.35 | 0.365 / −0.045 / +0.325 | 0.460 / −0.052 / +0.103 |
| 松杆 | 0.001 / −0.005 / +0.009 | −0.001 / −0.023 / −0.046 |

世界模型组降低了连续换速时的侧向与偏航峰值，但中速段末前向误差由 `0.015` 升到 `0.110 m/s`。预演权重从 `2.0` 改到 `0.8` 的该用例走出相同轨迹。两组都从 0.22 起步，在末段停住。**F2 仍未通过。** 当前跟踪会在一次 160 ms 接缝完成前再次重规划。保持目标速度时也有数次只用 5 步就换回的片段插入。这些事件破坏了步态周期，下一次限定接缝执行时机并保留同片段的精确时间推进。

三档同速正侧面对照在 `outputs/f2-sonic-cycle-plan-wm/paired-{slow,medium,fast}.gif`。三列分别为运动学组、预演组和固定完整参考，使用上表末四秒。

## 复现

先按 [周期参考验证](2026-11-24-f2-sonic-cycle-references.md) 准备周期的物理轨迹。在 `supermimic/` 目录生成周期库。

~~~bash
P=outputs/f2-sonic-cycle-physical
R=outputs/f2-sonic-cycle-retimed-physical
T=outputs/f2-sonic-transitions
.venv/bin/python scripts/pack_f2_bank.py --output outputs/f2-sonic-bank-cycles \
  --entry slow-a walk "$R/slow-a/reference_0_motion.jsonl" 0 350 \
  --entry slow-native walk "$P/slow-a/reference_0_motion.jsonl" 0 350 \
  --entry medium-a walk "$R/medium-a/reference_0_motion.jsonl" 0 350 \
  --entry fast-a walk "$R/fast-a/reference_0_motion.jsonl" 0 350 \
  --entry fast-b walk "$R/fast-b/reference_0_motion.jsonl" 0 350 \
  --entry fast-native walk "$P/fast-a/reference_0_motion.jsonl" 0 350 \
  --entry start-a start "$T/walk1_subject5/reference_15_motion.jsonl" 80 150 \
  --entry start-b start "$T/walk1_subject1/reference_69_motion.jsonl" 80 150 \
  --entry stop-a stop "$T/walk4_subject1/reference_7118_motion.jsonl" 0 115 \
  --entry stop-b stop "$T/walk2_subject3/reference_1434_motion.jsonl" 0 135 \
  --entry stop-c stop "$T/walk3_subject4/reference_12169_motion.jsonl" 0 134

export OMP_NUM_THREADS=2 WANDB_MODE=disabled
CKPT=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
for item in 'kin 0 10' 'wm 2 20'; do
  read -r name weight horizon <<< "$item"
  .venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
    --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
    --reference-bank outputs/f2-sonic-bank-cycles --planned-only True \
    --trackability-weight "$weight" --prediction-steps "$horizon" \
    --steady-forward True --extra-forward-speeds '(0.22, 0.50)' --multispeed-profile True \
    --cases '("steady_forward", "steady_forward_0p220", "steady_forward_0p500", "multispeed_profile")' \
    --output-dir "outputs/f2-sonic-cycle-plan-$name"
done
~~~
