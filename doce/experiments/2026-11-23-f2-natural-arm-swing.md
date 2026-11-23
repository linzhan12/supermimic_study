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

运行 `2026-11-23_17-10-30` 达到停步姿态门槛，肩差与髋差相关系数达到 `−0.467`，左右肩高度反相 `−0.986`。左肩幅度为 `25.63°`，右肩只有 `7.29°`，摆幅和平衡性未通过。稳态中左、右肩外展中位数约 `25°` 和 `7°`，为对称留出摆动空间提供了依据。速度门槛另有五项超差，仍留在 F2。

### 固定移动 encoder，单独训练停步

用通过原速度门槛的 F1 encoder 初始化移动分支，用已通过停步门槛的 F2 `0.25` encoder 初始化近零速分支。根据指令接近零的程度平滑混合二者的潜变量，固定移动 encoder 和共享 decoder，只微调停步 encoder。持续直走时，步态肩控同时给左右肩对称的 `±0.28 rad` 外展目标。该配置改变了速度分支结构，所有原闭环门槛及摆臂门槛保持相同，作为 F2 内的工程对照记录。

初始化探针的摆幅为 `39.33° / 22.04°`，左右相关 `−0.983`，肩差与髋差相关 `−0.833`。但稳态直走 MAE 为 `0.1541 m/s`，初始化探针也有四项原速度超差。联合训练改用 `±0.26 rad` 外展、`0.55` 的肩部混合系数，并让肩控从实测前向速度 `0.12 m/s` 起在 `0.16 m/s` 的区间内逐步介入。移动与停步 encoder 都以较小的学习率微调，从而让身体适应肩部动作，共享 decoder 和完整参考 encoder 保持冻结。

运行 `2026-11-23_17-18-49` 用 `4×10⁻⁵` 学习率训练 1,800 次，移动姿态倍率 `0.10`、转向奖励 `0.08`。左右肩 `10.04° / 12.86°`，持续前进 MAE `0.0344 m/s`，但肩差与髋差相关 `−0.239`，并有五项速度超差。停步关节 RMS 改善 `16.0%`，未过门槛。

较弱肩控的初始化探针将混合系数降至 `0.45`，保留 `±0.26 rad` 和 `0.12 m/s` 渐入，摆幅 `11.28° / 11.11°`，稳态前进 MAE `0.0530 m/s`，首次通过原 F2 与摆臂筛查。该组合使用已有的两个训练 encoder。复验运行 `2026-11-23_17-22-01` 以 `1×10⁻⁶` 学习率执行 200 次外层迭代，固定 F1 移动 encoder，只微调停步 encoder，世界模型继续用实际执行动作训练。该轮设置不会从两个旧运行恢复优化器。

## 速度、停步与摆臂筛查结果

训练代码 `7d94d4a`，训练目录 `supermimic/logs/SuperMimic-GenCoord-G1/2026-11-23_17-22-01/`。200 次外层迭代采集 25,600 条新物理转移。对比权重确认 F1 原有的 actor 张量均逐项相同，新增停步 encoder 有 18 个参数张量相对停步初始化更新。最终 actor SHA-256 为 `4c4554ff41073cc14781a148fd461f559ddd7cd4c53ab8be90edbc081ec8c47e`。

| 验收项 | F1 | 最终 F2 | 门槛 |
| --- | ---: | ---: | ---: |
| 行走中松杆，末 0.5 s 站姿关节 RMS（rad） | 0.3595 | 0.2025（−43.7%） | 改善 ≥ 20% |
| 行走中松杆，平均躯干倾斜（°） | 13.65 | 3.15（−76.9%） | 改善 ≥ 20% |
| 行走中松杆，末 0.5 s 残余平移速率（m/s） | 0.0033 | 0.0090 | < 0.10 |
| 持续直走，左 / 右肩俯仰摆幅（°） | 1.20 / 3.81 | 11.21 / 11.19 | 各 8–40，较小侧 ≥ 较大侧的一半 |
| 持续直走，左右肩相关系数 | −0.682 | −0.887 | ≤ −0.25 |
| 持续直走，肩差与髋差相关系数 | −0.324 | −0.533 | ≤ −0.30 |
| 持续直走，前向 MAE（m/s） | 0.0429 | 0.0538 | ≤ 0.10 |

六条原回放和新增持续直走均跑满，两个完整参考轨迹与 F1 逐字节一致。用最终 checkpoint 重做 CPU 回放，七条轨迹与摘要均逐字节相同。近静止起步平均前向 `0.315 m/s`，左右转平均偏航分别为 `+0.299` 和 `−0.354 rad/s`。移动阶段逐轴相对 F1 的最大 MAE 增量是前向 `0.0114 m/s`、侧向 `0.0169 m/s` 和偏航 `0.0284 rad/s`，均在原容差内。

**该 checkpoint 通过速度、停步姿态和摆臂筛查，整套步态复查未通过。** 用户在本地回放发现前进动作已失去完整参考的自然步态。随后复查确认跨步间距、髋膝踝摆幅及左右对称性存在明显差距，阶段仍留在 F2。该筛查使用一个训练种子、一条留出片段和两个固定初态，且该片段也用于挑选肩控参数。最终 CPU 回放记录了 31 条求解器迭代上限警告。见 [F2 步态与训练流程复查](2026-11-23-f2-forward-gait-audit.md)。

### 复现最终训练和本地验收

```bash
cd /home/hi/SuperMimic-Official/supermimic
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 \
  --motion-file '../data/lafan1_official/walk2_subject3.npz,../data/lafan1_official/walk3_subject3.npz' \
  --runner.independent-velocity True --runner.posture-weight 0.25 \
  --runner.moving-posture-scale 0 --runner.arm-swing-weight 0 \
  --runner.arm-swing-control 0.45 --runner.turn-tracking-weight 0 \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-07-37 \
  --runner.stop-encoder-init-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-35-09 \
  --runner.arm-swing-roll 0.26 --runner.arm-swing-speed-gate 0.12 \
  --runner.freeze-moving-encoder True --runner.policy.lr 0.000001 \
  --runner.actor-obs rep_state --runner.history-length 1 \
  --runner.run-name f2-phase-anchor-finetune-s0 \
  --runner.max-iterations 200 --runner.log-interval 50 --runner.save-interval 500 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 \
  --runner.wm.horizon 8 --runner.policy.batch-size 128 --runner.policy.horizon 32
```

```bash
cd /home/hi/SuperMimic-Official/supermimic
F2_CKPT=logs/SuperMimic-GenCoord-G1/2026-11-23_17-22-01
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
  --checkpoint-dir "$F2_CKPT" --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --record-posture True --steady-forward True --output-dir outputs/f2-final-replay
.venv/bin/python scripts/check_f2.py outputs/f1-arm-check outputs/f2-final-replay

OMP_NUM_THREADS=2 .venv/bin/python -u scripts/play_speed_test.py \
  --checkpoint-dir "$F2_CKPT" --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --device cpu --start-paused True --num-steps-rollout 32
```

本机量化结果与对照动画见 `supermimic/outputs/f2-phase-anchor-finetune-s0/`。Viser 回放通过终端打印的本机地址打开，点击 **Play** 后在 **Velocity Command** 中设置 `Lin Vel X = 0.35 m/s` 可以观察持续摆臂。
