# F2 平地速度与姿态试跑

2026-11-23。本文记录 F1 姿态诊断和 F2 开跑前验证。F2 闭环结果待手动训练后补录。

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

## 训练后验收

用终端 `Logging to` 的本地目录替换 `F2_CKPT`。量化回放输出为 JSONL 和 `summary.json`。浏览器回放打开终端打印的本地地址，点击 **Play**，用速度滑块检查前进、转向和松杆。

```bash
cd /home/hi/SuperMimic-Official/supermimic
F2_CKPT=logs/SuperMimic-GenCoord-G1/替换为本次训练时间戳
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
  --checkpoint-dir "$F2_CKPT" --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --record-posture True --output-dir outputs/f2-posture-s0

OMP_NUM_THREADS=2 .venv/bin/python -u scripts/play_speed_test.py \
  --checkpoint-dir "$F2_CKPT" --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --device cpu --start-paused True --num-steps-rollout 32
```

试跑优先检查两个指标。行走初态松杆的末 0.5 s 站姿关节 RMS 与松杆段的平均躯干倾角，分别比 F1 下降至少 20%。固定摇杆的移动阶段逐轴比较 MAE，容许前向和侧向各增加 `0.03 m/s`，偏航增加 `0.05 rad/s`。仍须完成起步、同向转向及松杆的速度门槛，整段无跌倒，完整参考两条轨迹与 F1 一致。走动时另外记录支撑脚足速和腰部角度，用配对本地回放检查动作是否更自然。达到这些条件才能把速度和姿态都纳入下一阶段的判断。
