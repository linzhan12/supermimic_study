# F2 速度、停步姿态与自然摆臂

2026-11-23。阶段保持在 F2，沿用 [F2 原验收门槛](2026-11-23-f2-velocity-posture.md)。

## 固定预算的权重对照

权重 `0.125`、2,000 次迭代的运行目录为 `2026-11-23_16-57-24`，训练代码为 `bdb7eed`。六条原回放均跑满。本轮在行走初态松杆的关节 RMS 由 F1 的 `0.359` 降至 `0.304 rad`，降幅 `15.4%`，低于 `20%` 门槛。右斜移侧向 MAE 比 F1 增加 `0.03070 m/s`，高于 `0.030 m/s` 容差。停步躯干倾斜降至 `2.27°`。整体未通过。

## 持续直走的摆臂补测

从原近静止初态开始，保持零速 1 秒，再持续给 `0.35 m/s` 前进指令 6 秒。用直走最后 4 秒衡量稳态，避免把起步时的一次肩姿态变化计作摆臂。原 F2 速度、停步姿态、回放完成和完整参考一致性门槛继续使用。

| 策略 | 左肩俯仰 P95−P05（°） | 右肩俯仰 P95−P05（°） | 左右肩相关系数 | 肩差与髋差相关系数 |
| --- | ---: | ---: | ---: | ---: |
| F1 | 1.20 | 3.81 | −0.682 | −0.324 |
| F2，权重 0.25，2,000 次 | 2.11 | 2.62 | +0.246 | +0.202 |
| F2，权重 0.125，2,000 次 | 1.75 | 2.81 | −0.422 | −0.549 |

同一留出片段的完整参考回放中，抓取近静止用例中前向命令 `0.22–0.55 m/s`、侧向绝对值小于 `0.15 m/s`、偏航绝对值小于 `0.20 rad/s` 的 35 个采样，左右肩幅度分别为 `12.3°` 和 `14.3°`，左右相关系数为 `−0.834`。对整条留出动捕按相近速度进行筛选时，左右肩的幅度约 `31°`，其中混有姿势变化和转向过渡。

新增工程筛查门槛为左右肩各 `8–40°`、较小侧幅度至少为较大侧一半、两肩相关系数不大于 `−0.25`、肩差与髋差相关系数不大于 `−0.30`。同时要求稳态前向 MAE 至多 `0.10 m/s` 且全程无跌倒。上述指标检查是否有适度、对侧协调的摆动，整体自然度仍由实际回放观察。

## 训练调整

定量结果表明降低统一姿态权重会削弱停步站姿，手臂几乎静止的问题在 F1 也存在。新的对照保留 `0.25` 的零速姿态权重，把非零命令的姿态倍率设为 `0.5`，有效权重为 `0.125`。独立的前进摆臂项根据实际髋关节俯仰差给出限幅的对侧肩俯仰目标。摆臂项随侧向、转向指令减弱，在零速时为零。原共享 decoder 与动作 encoder 保持冻结。

```bash
cd /home/hi/SuperMimic-Official/supermimic
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 \
  --motion-file '../data/lafan1_official/walk2_subject3.npz,../data/lafan1_official/walk3_subject3.npz' \
  --runner.independent-velocity True --runner.posture-weight 0.25 \
  --runner.moving-posture-scale 0.5 --runner.arm-swing-weight 0.15 \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --runner.actor-obs rep_state --runner.history-length 1 \
  --runner.run-name f2-phase-arm-swing-s0 \
  --runner.max-iterations 2000 --runner.log-interval 50 --runner.save-interval 500 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 \
  --runner.wm.horizon 8 --runner.policy.batch-size 128 --runner.policy.horizon 32
```

量化回放使用 `scripts/eval_p0.py --record-posture True --steady-forward True`。基线记录位于本机 `supermimic/outputs/f1-arm-check/`、`f2-w025-arm-check/` 和 `f2-w0125-arm-check/`。

### 仅用摆臂奖励的结果

运行 `2026-11-23_17-04-07` 的停步站姿 RMS 为 `0.204 rad`，躯干倾斜为 `3.46°`。持续前进 MAE 为 `0.0232 m/s`，但左右肩摆幅只有 `2.42°`、`2.81°`，肩差与髋差相关系数为 `−0.169`。右转侧向和偏航 MAE 分别比 F1 增加 `0.03029 m/s`、`0.07102 rad/s`，均超限。F2 和摆臂筛查都未通过。

### 用步态相位引导肩部动作

固定速度对应速度 encoder 的固定目标输入。上一轮新增奖励能影响这个目标编码，但持续直走时肩部仍近乎静止。下一轮在速度分支加入由当前髋俯仰及其角速度驱动的限幅对侧肩动作。它只混合两个肩俯仰动作列，随直走指令生效，训练和回放共用同一个 actor 前向。完整参考分支继续使用原解码输出。该轮另加入转向指令上的窄偏航误差项，以处理前一轮右转超调。

首轮参数为零速姿态权重 `0.25`、移动姿态倍率 `0.5`、摆臂奖励 `0.10`、步态肩动作混合系数 `0.90`、转向偏航奖励 `0.20`，预算仍为 2,000 次，种子为 0，从同一预训练 checkpoint 新起跑。肩部相位控制和转向奖励同时引入，本轮用于联合验收，两者的单独贡献需要额外对照。
