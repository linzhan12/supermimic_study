# F2 平地速度与姿态试跑

2026-11-23。训练代码 `1397537`，F2 运行目录 `2026-11-23_16-35-09`。

## F1 现象

用户在本地回放发现，F1 的速度更接近期望，走动和停后的姿态仍不自然。用 `walk1_subject1.npz`、相同的两个初态和固定摇杆序列重新回放 F0 与 F1，补记实际关节角、躯干倾斜及触地采样时的足部 site 平移速度。

| F1 用例与阶段 | 速度现象 | 姿态现象 |
| --- | --- | --- |
| 近静止起步，目标 `0.35 m/s` | 平均前向 `0.331 m/s` | 腰部偏航绝对角均值 `6.28°`，触地足部 site 平均平移速率 `0.127 m/s` |
| 近静止用例左转，目标 `+0.40 rad/s` | 平均偏航 `+0.377 rad/s` | 腰部偏航绝对角均值 `8.42°`，躯干平均倾斜 `9.49°` |
| 近静止用例走动后松杆 | 末 0.5 s 平移速率 `0.013 m/s` | 平均腰部俯仰绝对角 `9.60°`，躯干平均倾斜 `11.77°` |
| 行走初态松杆 | 末 0.5 s 平移速率 `0.003 m/s` | 平均腰部俯仰绝对角 `10.97°`，躯干平均倾斜 `13.65°`，末 0.5 s 名义站姿关节 RMS `0.359 rad` |

F1 通过了原先的速度筛查。姿态指标说明速度达到门槛后仍有前倾、转向时扭腰和支撑脚运动。触地足部 site 速率是单步滑动代理，包含足部滚动。F0 在近静止用例基本没有起步，因此它在同一阶段的低足速不能用来证明步态更好。原始记录在本机 `supermimic/outputs/f0-posture-s0-20261123/` 和 `supermimic/outputs/f1-posture-s0-20261123/`。

## F2 对照设置

- F2 和 F1 从同一份预训练 checkpoint 起跑，沿用 `rep_state`、相同动捕片段、种子 0、命令分布以及 256,000 条新物理转移预算。actor 继续只更新速度 encoder。
- 速度奖励保留原尺度。先以 `0.25` 作为姿态项权重。上半身项对躯干倾斜、腰部转动和过量手臂偏移评分。手臂保留 `0.4 rad` 的活动余量。
- 有非零命令时，下半身项用几何支撑和支撑脚滑动评分。精确零速下，当身体实际速度和偏航速率降下来且双足接近地面时，名义站姿的得分逐渐升高。收步过渡继续由速度目标驱动。
- 物理仿真与想象使用同一套 G1 正向运动学和步末关节状态。训练支撑信号来自足部 site 与平地的几何距离，量化回放另用 MuJoCo 实际触地标志计算足速。

## 开跑前验证

4 个 CPU 仿真实例覆盖零速和非零命令。用 MuJoCo 的实际足部位置和速度核对了正向运动学。在同一物理步末状态和步前命令上，物理与想象总奖励的最大绝对差为 `1.51×10⁻⁹`，姿态奖励对动作的最大绝对梯度约 `9.14×10⁻⁴`。两步想象速度 encoder 有梯度，共享 decoder 与动作 encoder 保持冻结。仓库 222 个测试通过。GPU 完整训练和闭环姿态收益待本轮手动试跑验证。

## 手动训练

```bash
cd /home/hi/SuperMimic-Official/supermimic
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 \
  --motion-file '../data/lafan1_official/walk2_subject3.npz,../data/lafan1_official/walk3_subject3.npz' \
  --runner.independent-velocity True --runner.posture-weight 0.25 \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --runner.actor-obs rep_state --runner.history-length 1 \
  --runner.run-name f2-velocity-posture-s0 \
  --runner.max-iterations 2000 --runner.log-interval 50 --runner.save-interval 500 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 \
  --runner.wm.horizon 8 --runner.policy.batch-size 128 --runner.policy.horizon 32
```

后续权重对照与速度、停步和摆臂筛查见 [F2 速度、停步姿态与自然摆臂](2026-11-23-f2-natural-arm-swing.md)。整套步态仍未通过，见 [F2 步态与训练流程复查](2026-11-23-f2-forward-gait-audit.md)。

## 训练后验收

量化回放输出为 JSONL 和 `summary.json`。浏览器回放打开终端打印的本地地址，点击 **Play**，用速度滑块检查前进、转向和松杆。

```bash
cd /home/hi/SuperMimic-Official/supermimic
F2_CKPT=logs/SuperMimic-GenCoord-G1/2026-11-23_16-35-09
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
  --checkpoint-dir "$F2_CKPT" --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --record-posture True --output-dir outputs/f2-posture-s0-copy

OMP_NUM_THREADS=2 .venv/bin/python -u scripts/play_speed_test.py \
  --checkpoint-dir "$F2_CKPT" --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --device cpu --start-paused True --num-steps-rollout 32
```

试跑优先检查两个指标。行走初态松杆的末 0.5 s 站姿关节 RMS 与松杆段的平均躯干倾角，分别比 F1 下降至少 20%。固定摇杆的移动阶段逐轴比较 MAE，容许前向和侧向各增加 `0.03 m/s`，偏航增加 `0.05 rad/s`。仍须完成起步、同向转向及松杆的速度门槛，整段无跌倒，完整参考两条轨迹与 F1 一致。走动时另外记录支撑脚足速和腰部角度，用配对本地回放检查动作是否更自然。达到这些条件才能把速度和姿态都纳入下一阶段的判断。

## F2 实测现象

训练于 16:35:09 开始，16:36:54 正常结束并保存最终权重。核对了 F1 和 F2 的本地配置，初始 checkpoint、种子、训练动作、仿真频率和交互预算一致。相对共同起点，F2 actor 的 18 个变动张量均属于速度 encoder。F2 在两类初态下的六条回放全部跑满，完整参考两条轨迹与 F1 逐字节一致。

| 固定摇杆姿态指标 | F1 | F2 | 变化 |
| --- | ---: | ---: | ---: |
| 行走初态松杆，末 0.5 s 名义站姿关节 RMS（rad） | 0.359 | 0.200 | −44.3% |
| 行走初态松杆，躯干平均倾斜（°） | 13.65 | 3.32 | −75.7% |
| 行走初态松杆，腰部俯仰绝对角均值（°） | 10.97 | 1.53 | −86.0% |
| 近静止用例左转，腰部偏航绝对角均值（°） | 8.42 | 2.61 | −69.0% |
| 近静止用例左转，躯干平均倾斜（°） | 9.49 | 3.55 | −62.6% |

行走初态松杆的末 0.5 s，F1 与 F2 的平均残余平移速率分别为 `0.0033` 和 `0.0037 m/s`。近静止用例中，F2 的起步平均前向速度为 `0.272 m/s`，左右转向的平均偏航速率分别为 `+0.323` 和 `−0.404 rad/s`，均保留了 F1 的绝对速度门槛。右斜移目标侧向速度为 `−0.20 m/s`，实际侧向均值从 `−0.042` 改善到 `−0.119 m/s`。

| 超过容差的移动阶段 | 指标 | F1 MAE | F2 MAE | 增量 | 容许增量 |
| --- | --- | ---: | ---: | ---: | ---: |
| 近静止用例起步 | 前向（m/s） | 0.0490 | 0.0824 | +0.0334 | 0.0300 |
| 近静止用例左转 | 前向（m/s） | 0.0376 | 0.0958 | +0.0581 | 0.0300 |
| 近静止用例右转 | 前向（m/s） | 0.0526 | 0.0863 | +0.0337 | 0.0300 |
| 行走初态直走 | 偏航（rad/s） | 0.1843 | 0.2492 | +0.0649 | 0.0500 |

## 结论与阶段判定

`0.25` 的姿态权重通过了两个停步姿态门槛，也保留了完整参考能力。松杆末半秒的两组残余速度都接近零，此时可以直接比较站姿关节。有指令的转向阶段，左转平均前向速度从 `0.334` 降至 `0.254 m/s`，因此转向姿态改善和减速同时发生。四项移动阶段误差超过预设容差，**F2 整体筛查未通过**。阶段留在 F2，后续在相同起点和预算下检查姿态权重与速度误差的权衡。

结论限于一个训练种子和一条留出片段，CPU 回放出现 29 条求解器迭代上限警告。逐步数据、汇总、评测日志及代码版本记录位于本机 `supermimic/outputs/f2-posture-s0-20261123/`，评测权重 SHA-256 为 `070756438dfd6d5b9c14051d546ca0160a356e9760ec1032721ad7a3e9d0eb71`。

## 未达标诊断

2,000 次外层迭代相当于 256,000 条真实仿真转移，按现有批量与回放调度约有 996 次策略更新。从本机离线日志提取的 F2 `Policy/reward` 在后半程趋平。该指标汇总 32 步想象轨迹上的速度、姿态回报和末端价值估计。

| 外层迭代区间 | F2 平均想象回报 | F2 平均世界模型 `Loss/state` |
| --- | ---: | ---: |
| 1,000–1,450 | 3.4114 | 0.003365 |
| 1,500–1,700 | 3.4228 | 0.002959 |
| 1,750–1,950 | 3.4168 | 0.003002 |

训练回报趋平，仍需按留出命令逐段判断。离线保存使用同名权重覆盖中间版本，因此目前只有 2,000 次迭代的闭环结果。单个最终点无法判定继续训练后，四个超差项会缩小还是保持。

把 F1 和 F2 固定摇杆轨迹的步末状态统一代入 F2 奖励，用实际速度和足部平移速率以及由记录姿态重建的足部高度重算。以下是阶段内的平均速度项、乘过 `0.25` 的姿态项和两者相加的即时分数。

| 阶段 | 速度项 F1 → F2 | 姿态项 F1 → F2 | 总分 F1 → F2 |
| --- | ---: | ---: | ---: |
| 近静止起步 | 0.905 → 0.863 | 0.115 → 0.168 | 1.020 → 1.031 |
| 近静止用例左转 | 0.924 → 0.850 | 0.100 → 0.146 | 1.024 → 0.996 |
| 近静止用例右转 | 0.940 → 0.879 | 0.114 → 0.138 | 1.054 → 1.017 |
| 行走初态直走 | 0.862 → 0.825 | 0.131 → 0.147 | 0.993 → 0.972 |

起步段的姿态收益足以补偿速度项损失，说明当前目标在该段确实允许用一些速度换姿态。左右转段则连即时总分也下降。左转最后半秒，F2 平均前向速度仍只有 `0.238 m/s`，F1 为 `0.329 m/s`。行走初态直走的偏航退化主要出现在前半秒，F2 偏航 MAE 为 `0.316 rad/s`，F1 为 `0.182 rad/s`。这几个现象给出了两类待区分因素，分别是训练预算对特定命令的影响，以及姿态、速度与转向过渡在当前目标中的权衡。统一重评分数是物理即时分数，和训练日志中的想象回报具有不同的时间尺度及采样分布。

### 固定权重的预算复验

先将外层迭代从 `2,000` 增加到 `6,000`，共 `768,000` 条物理转移。从同一预训练 checkpoint 重新开始，保持姿态权重 `0.25`、种子、数据、命令分布和其他参数相同。当前 checkpoint 文件保存模型权重，优化器与经验池没有保存，因此采用重新起跑的预算对照。

```bash
cd /home/hi/SuperMimic-Official/supermimic
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 \
  --motion-file '../data/lafan1_official/walk2_subject3.npz,../data/lafan1_official/walk3_subject3.npz' \
  --runner.independent-velocity True --runner.posture-weight 0.25 \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --runner.actor-obs rep_state --runner.history-length 1 \
  --runner.run-name f2-velocity-posture-6000-s0 \
  --runner.max-iterations 6000 --runner.log-interval 50 --runner.save-interval 500 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 \
  --runner.wm.horizon 8 --runner.policy.batch-size 128 --runner.policy.horizon 32
```

复用同一闭环评测和原门槛，重点比较四个超差项与两个停步姿态项。若加预算后仍有同类超差，再回到 2,000 次预算单独比较较低姿态权重。若更长预算的 F2 达标，正式比较再配对运行相同预算的 F1。阶段出口仍要求姿态和速度联合达标。

### 6,000 次复验结果

运行 `2026-11-23_16-49-21` 于 16:52:07 正常结束。从本地配置逐项核对，和上一轮相比只有迭代数及运行名不同。复用 `walk1_subject1.npz`、种子 0、两个初态和相同命令。六条回放均跑满，完整参考的两条轨迹与 F1 逐字节一致，绝对起步、转向及松杆速度门槛也通过。

| 行走中松杆后的指标 | F1 | F2 2,000 次 | F2 6,000 次 |
| --- | ---: | ---: | ---: |
| 末 0.5 s 名义站姿关节 RMS（rad） | 0.359 | 0.200 | 0.270 |
| 松杆段平均躯干倾斜（°） | 13.65 | 3.32 | 2.77 |
| 末 0.5 s 平均残余平移速率（m/s） | 0.0033 | 0.0037 | 0.0052 |

F2 6,000 次的站姿关节 RMS 和躯干倾斜分别比 F1 低 `24.8%` 和 `79.7%`，达到两项姿态门槛。原来的左转前向 MAE 从 `0.0958` 降至 `0.0398 m/s`，右转前向 MAE 从 `0.0863` 降至 `0.0667 m/s`，均回到容差内。近静止起步的前向误差和行走初态直走的偏航误差仍超差，另有四个新超差项。

| 6,000 次后超差的阶段 | 指标 | F1 MAE | F2 2,000 次 MAE | F2 6,000 次 MAE | 6,000 次相对 F1 增量 | 容许增量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 近静止用例起步 | 前向（m/s） | 0.0490 | 0.0824 | 0.0955 | +0.0465 | 0.0300 |
| 近静止用例右转 | 偏航（rad/s） | 0.0947 | 0.1173 | 0.1909 | +0.0962 | 0.0500 |
| 近静止用例左斜移 | 前向（m/s） | 0.0325 | 0.0468 | 0.0675 | +0.0351 | 0.0300 |
| 近静止用例左斜移 | 偏航（rad/s） | 0.1599 | 0.1525 | 0.2108 | +0.0509 | 0.0500 |
| 行走初态直走 | 侧向（m/s） | 0.0592 | 0.0685 | 0.0916 | +0.0325 | 0.0300 |
| 行走初态直走 | 偏航（rad/s） | 0.1843 | 0.2492 | 0.2650 | +0.0807 | 0.0500 |

在这次 6,000 次运行中，`Policy/reward` 在 1,750–1,950、3,750–3,950 和 5,750–5,950 次的平均值分别为 `3.4092`、`3.3900` 和 `3.3703`。加预算修复了部分命令，整体速度超差项却从四项增加到六项。**固定权重 `0.25` 的复验仍未通过，阶段留在 F2。** 这组结果表明把本次预算增至 6,000 次仍不足以跨过联合门槛。单个种子和片段也限制了对训练收敛和奖励权重的因果判断。

原始轨迹、评分、训练曲线和比较记录位于本机 `supermimic/outputs/f2-posture-6000-s0-20261123/`。CPU 回放记录了 39 条求解器迭代上限警告。评测权重 SHA-256 为 `e41a184db9ca3c10a6a657491778dd628a92ed21d60f85fbe3eb5a67211fdf05`。

### 固定预算的姿态权重复验

恢复到首轮的 2,000 次预算。从相同预训练起点重新训练，将姿态权重由 `0.25` 降至 `0.125`，以相同种子、命令分布和数据比较。继续按原门槛验收，检查速度是否恢复、两项停步姿态改善是否仍各达 20%。

```bash
cd /home/hi/SuperMimic-Official/supermimic
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 \
  --motion-file '../data/lafan1_official/walk2_subject3.npz,../data/lafan1_official/walk3_subject3.npz' \
  --runner.independent-velocity True --runner.posture-weight 0.125 \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --runner.actor-obs rep_state --runner.history-length 1 \
  --runner.run-name f2-velocity-posture-weight-0125-s0 \
  --runner.max-iterations 2000 --runner.log-interval 50 --runner.save-interval 500 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 \
  --runner.wm.horizon 8 --runner.policy.batch-size 128 --runner.policy.horizon 32
```
