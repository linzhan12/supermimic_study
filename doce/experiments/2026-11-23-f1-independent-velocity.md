# F1 独立速度基线试跑

2026-11-23。训练运行 `2026-11-23_16-07-37`，训练代码版本 `58ea380`。

## 问题与试跑设置

[P0](2026-11-23-p0-g1-replay.md) 的近静止用例中，给定 `0.35 m/s` 前向摇杆命令，平均实际速度为 `0.023 m/s`。F1 检验独立速度目标能否让速度 encoder 从近静止状态带出步态，同时保留松杆减速和原有完整参考能力。

- 用 P0 的本地 checkpoint `2026-11-17_17-55-41` 初始化 actor 和世界模型。只更新 actor 的速度 encoder，冻结共享 decoder、动作 encoder 和 actor 归一化统计。世界模型继续用实际执行动作与连续物理状态拟合，critic 在试跑中重新初始化。
- 现有 checkpoint 使用 `rep_state`，历史长度为 1。该权重用于匹配 F0 起点的内部试跑。正式 F1/F2 对照需要兼容本体历史观测的预训练权重。
- 每个命令保持 50–100 步，对应 50 Hz 下的 1–2 s。命令类别为精确零速、纯纵向、纯侧向、纯转向、斜移和走中转向，精确零速占采样的 `2/7`。纵向范围为 `[-0.20, 0.55] m/s`，侧向为 `[-0.25, 0.25] m/s`，偏航为 `[-0.55, 0.55] rad/s`。
- 下游复位从 `walk2_subject3`、`walk3_subject3` 取近静止和行走帧，近静止抽样概率为 25%。两段共 24,240 帧、约 8.08 分钟，只用于复位起姿。`walk1_subject1` 留给下游回放评测。旧 checkpoint 的预训练动作覆盖尚未核实。
- 物理和想象使用同一速度奖励。平移先按骨盆航向投影，偏航取世界竖直轴分量。误差核的平移、偏航尺度分别为 `0.25 m/s` 和 `0.50 rad/s`，平移与偏航权重之比为 `2:1`。动作后状态用这一步的原命令评分，再刷新命令。回放记录命令、保持步数、类别和探索后的有效动作。跌倒判据使用 `0.45 m` 骨盆高度与 `60°` 倾斜阈值，物理回合另有 10 s 时间截断。

## 开跑前的现象与结论

| 验证项 | 观察 |
| --- | --- |
| 命令时序与零速 | 真实仿真跨过保持期边界时，采集保存原命令，想象从该回放起点恢复原命令和倒计时。精确零速一直保持到最后一个评分步。 |
| 同状态评分 | 4 个 CPU 仿真实例分别使用零速、斜移和左右转向相关命令。把同一批物理步末状态送入想象评分，最大奖励差为 `1.51×10⁻⁹`，四条转移均未触发跌倒。 |
| 动作标签 | 人工设置动作裁剪上限时，回放动作与仿真动作管理器收到的动作完全一致。 |
| 梯度 | 两步想象回传的速度 encoder 最大绝对梯度为 `0.03595`，共享 decoder 和动作 encoder 无参数梯度。decoder 对历史输入仍有非零梯度。 |
| P0 回归 | 接入 F1 后复跑 P0，六条轨迹逐字节相同。仓库的 220 个测试通过。 |

结果支持进入手动训练，用闭环回放检验跟速收益。开跑前检查使用了旧权重和 CPU，只执行采集、评分和梯度检查。CUDA 完整训练、训练后跟速表现及模型在更长闭环窗口内的误差仍需由试跑验证。

## 手动训练

在可用 NVIDIA GPU 的终端执行。种子为 0，计划采集 256,000 条物理转移。训练终端显示进度，W&B 离线模式只在本机保存指标。终端打印的 `Logging to` 目录会保存 `local_config.json` 和最终权重。

```bash
cd /home/hi/SuperMimic-Official/supermimic
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 \
  --motion-file '../data/lafan1_official/walk2_subject3.npz,../data/lafan1_official/walk3_subject3.npz' \
  --runner.independent-velocity True \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --runner.actor-obs rep_state --runner.history-length 1 \
  --runner.run-name f1-independent-velocity-s0 \
  --runner.max-iterations 2000 --runner.log-interval 50 --runner.save-interval 500 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 \
  --runner.wm.horizon 8 --runner.policy.batch-size 128 --runner.policy.horizon 32
```

## 本地可视化回放

使用本地权重和动作文件。打开终端打印的 `Local playback` 地址，点击 **Play**。在 **Velocity Command** 中将 **Lin Vel X (m/s)** 从 `0` 调到 `0.4` 可以观察起步，再调回 `0` 观察松杆。结束时在终端按 Ctrl+C。

```bash
cd /home/hi/SuperMimic-Official/supermimic
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/play_speed_test.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-07-37 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --device cpu --start-paused True --num-steps-rollout 32
```

已用这次 F1 权重访问本地 Viser 页面并验证速度与完整参考模式均能步进。测试拦截了 W&B API 和初始化调用。

## 训练后量化回放与判定

量化回放沿用 P0 的同一评测片段、两类初态和固定摇杆序列。从 `summary.json` 和逐步轨迹读取结果。

```bash
cd /home/hi/SuperMimic-Official/supermimic
F1_CKPT=logs/SuperMimic-GenCoord-G1/2026-11-23_16-07-37
F1_OUT=outputs/f1-independent-velocity-s0-copy
OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_p0.py \
  --checkpoint-dir "$F1_CKPT" \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --output-dir "$F1_OUT" && git -C .. rev-parse HEAD > "$F1_OUT/git_commit.txt"
```

## 训练后的现象与结论

训练于 16:07:37 启动，16:09:35 正常退出并保存本地权重。训练前后逐项比较 actor 张量，共 18 个张量发生变化，均属速度 encoder。完整参考 A 路在两个初态下的轨迹均与 P0 逐字节一致。F1 六条轨迹都完成了计划窗口，未触发骨盆跌倒条件。

| 固定摇杆指标 | P0 | F1 | 预设门槛 |
| --- | ---: | ---: | ---: |
| 近静止起步，平均前向速度（m/s） | 0.023 | **0.331** | ≥ 0.20 |
| 左转段，平均偏航速率（rad/s） | +0.024 | **+0.377** | ≥ +0.15 |
| 右转段，平均偏航速率（rad/s） | −0.063 | **−0.361** | ≤ −0.15 |
| 行走中松杆，最后 0.5 s 平均平移速率（m/s） | 0.034 | **0.003** | < 0.10 |

**F1 通过预设的单种子试跑门槛。** 在这组配对用例中，独立速度基线改善了近静止起步、走中转向和松杆后的残余速度。右斜移的侧向目标为 `−0.20 m/s`，实际均值为 `−0.042 m/s`，且伴随 `−0.161 rad/s` 的非指令偏航。参考速度 B 路的偏航误差也有所增加。后续对照应把右斜移和偏航串扰纳入评测。

本轮使用一份 `rep_state` checkpoint、一个训练种子和一条下游留出片段。CPU 回放过程中记录了 31 条求解器迭代上限警告。结论适用于本轮用例，正式 F1/F2 对照仍需匹配本体历史的预训练起点和更多配对测试。原始轨迹位于本机 `supermimic/outputs/f1-independent-velocity-s0-20261123/`。
