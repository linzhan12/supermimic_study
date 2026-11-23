# F2 多速度自然步态与回归验收

2026-11-23。平地仿真 F2 通过。验收权重为运行 `2026-11-23_21-29-10` 在第 1,600 次迭代保存的中间 checkpoint，位于本机 `supermimic/outputs/f2-steering-i1600-snapshot/`。actor SHA-256 为 `fe5435854a8b8a5aeee49c63a1c74f085754effd12e7c8585c3ab1f9c1f56b51`。本轮从失败的速度与摆臂版本出发，依次验证了多动作的速度匹配、连续相位、贴地步态的修复、镜像与跟速、转向和斜移补训。 [训练过程和失败记录](2026-11-23-f2-multispeed-gait.md)

## 最后的转向与斜移补训

从已经通过三档直走步态筛查的镜像实验迭代 1,000 权重启动。训练新增的状态相关补偿，最多改动髋、膝、踝和腰部关节目标 `0.30 rad`，只在达到侧向或偏航指令门槛时介入。补训用真实的横移、转向和走转切换，增加较窄的三轴跟速奖励，权重 `0.90`。全部 94 个已有 actor 张量逐项冻结，新增分支有 10 个参数张量。参考分支、直走、精确零速继续使用已有策略，三档稳态与连续直走换速轨迹在补训前后逐字节一致。

| 保存点 | 主场景结果 | 留出场景结果 |
| --- | --- | --- |
| 800 | 右转侧向和偏航误差仍超限 | 未作为最终候选 |
| **1,600** | **通过** | **三条额外留出动作均通过** |
| 3,200 | 通过 | `walk2_subject4` 通过，`walk4_subject1` 左转偏航差 `+0.0600 rad/s` 高于 `+0.0500` 容差 |

## 三档步态与换速

固定指令持续 6 秒，表中用最后 4 秒。姿态量由真实 MuJoCo 轨迹测量，落脚间距与离地峰值是按左右完整摆动计算的中位数。

| 目标（m/s） | 实测前向（m/s） | 落脚前后间距 左 / 右（cm） | 脚底离地峰值 左 / 右（mm） | 膝关节摆幅 左 / 右（°） |
| ---: | ---: | ---: | ---: | ---: |
| 0.22 | 0.224 | 12.5 / 19.9 | 30 / 27 | 37.3 / 33.6 |
| 0.35 | 0.347 | 20.0 / 23.4 | 41 / 43 | 42.0 / 42.2 |
| 0.50 | 0.489 | 25.2 / 28.1 | 44 / 46 | 44.8 / 45.4 |

直接检查 0.35 m/s 的配对回放，能够看到交替屈膝、离地和明确的向前落脚，左右臂随相反侧腿摆动。完整参考的跨步更大，和它的实际速度及该片段的动作风格均有关。候选的左右肩俯仰摆幅为 `27.6° / 26.9°`，肩差与髋差的相关系数为 `−0.992`。沿 `0.22 → 0.50 → 0.35 → 0` 换速，各段末半秒平均前向速度为 `0.221 / 0.495 / 0.338 / −0.004 m/s`。另外两条留出回放的最后一段前向速度绝对值均不超过 `0.005 m/s`。本机可查看 `supermimic/outputs/f2-steering-i1600-s0/forward-gait-comparison.gif` 和 `acceleration-and-stop.gif`。

## 完整回归

四个回放用例均使用下游训练集之外的动作片段。每个用例的 F1 和候选使用同一命令序列及同一初态。共 40 条闭环回放完成，覆盖完整参考、参考速度、固定摇杆起停与转向斜移、三档持续前进、连续换速。原完整参考 8 条轨迹与对应 F1 逐字节一致，原跟速和停步门槛、三档步态筛查同时满足。

| 留出动作与初态帧 | 行走松杆末半秒关节 RMS F1 → F2（rad） | 松杆阶段躯干倾斜 F1 → F2（°） | 停稳平移速率（m/s） | 移动阶段最大 MAE 增量 前向 / 侧向 / 偏航 |
| --- | ---: | ---: | ---: | ---: |
| `walk1_subject1`，`0 / 205` | 0.359 → 0.205 | 13.65 → 2.77 | 0.013 | +0.01155 / +0.01936 / +0.04175 |
| `walk2_subject4`，`70 / 5599` | 0.357 → 0.206 | 13.35 → 2.05 | 0.014 | +0.01087 / +0.01576 / +0.02531 |
| `walk4_subject1`，`60 / 3559` | 0.353 → 0.207 | 12.95 → 3.41 | 0.007 | +0.00446 / +0.01513 / +0.02147 |
| `walk3_subject4`，`50 / 8045` | 0.352 → 0.207 | 9.76 → 2.52 | 0.015 | +0.00679 / +0.01805 / +0.01723 |

移动阶段容差为 `+0.03 m/s / +0.03 m/s / +0.05 rad/s`，停步关节 RMS 和躯干倾斜均需下降至少 `20%`。`walk2_subject4` 与 `walk4_subject1` 用于从通过主场景的权重中挑选保存点，`walk3_subject4` 是权重确定后的独立复验。回放使用 CPU，模型使用一个训练种子和仓库当前的 `rep_state`。通过结论适用于当前 50 Hz 平地仿真验收。仓库测试共 228 个通过。

## 复现命令

镜像步态实验从 `2026-11-23_21-13-22` 初始化，并保存 `outputs/f2-reflection-i1000-snapshot/`。转向补训从该中间权重启动。训练七条输入中含五条本地 LAFAN1 片段、ACCAD 的 `B1-standtowalk` 和 CMU 的 `16_34`。二者均用仓库转换脚本从已下载的 IsaacLab AMP G1 文件生成 50 Hz 格式。 [外部动作来源及筛选](2026-11-23-f2-multispeed-gait.md)

外部输入分别是 `g1/ACCAD/Female1Walking_c3d/B1-standtowalk_poses_120_jpos.npz`（原始归档 SHA-256 `a573d18c84acb7de845bdba5c367c0823df0d4b0d79e401d00331d310498bc66`）与 `g1/CMU/16/16_34_poses_120_jpos.npz`（`25aeb8439621ac4dfe8a3fd5f079dd8a38c943c094791f779daa72f9ff3045f0`）。分别以 `B1-standtowalk.npz` 与 `16_34.npz` 为本地文件名，用 `scripts/convert_amp_g1_motion.py` 转入下述输出目录。 [Berkeley G1 发布库](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1/tree/da464c1)

```bash
cd /home/hi/SuperMimic-Official/supermimic
LAFAN='../data/lafan1_official'
MOTIONS="$LAFAN/walk1_subject2.npz,$LAFAN/walk2_subject3.npz,$LAFAN/walk3_subject2.npz,$LAFAN/walk3_subject3.npz,$LAFAN/walk3_subject5.npz,outputs/f2-external-accad-50hz/B1-standtowalk.npz,outputs/f2-external-cmu-50hz/16_34.npz"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 --motion-file "$MOTIONS" \
  --runner.independent-velocity True --runner.posture-weight 0.25 --runner.moving-posture-scale 0 \
  --runner.arm-swing-control 0.45 --runner.arm-swing-roll 0.26 --runner.arm-swing-speed-gate 0.12 --runner.arm-swing-weight 0.20 \
  --runner.init-checkpoint-dir outputs/f2-reflection-i1000-snapshot \
  --runner.stop-encoder-init-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-35-09 \
  --runner.forward-gait-weight 0.80 --runner.forward-gait-action-limit 0.60 --runner.forward-gait-phase True \
  --runner.forward-foot-cycle-weight 1.45 --runner.forward-gait-reflection 1.0 --runner.forward-precision-weight 0.85 \
  --runner.forward-gait-command-fraction 0.25 --runner.forward-gait-hold-s '(2.0, 3.0)' \
  --runner.steering-recovery-limit 0.30 --runner.steering-recovery-weight 0.90 \
  --runner.actor-obs rep_state --runner.history-length 1 --runner.run-name f2-multispeed-steering-repro \
  --runner.max-iterations 1601 --runner.log-interval 100 --runner.save-interval 800 --runner.policy.lr 0.00014 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 --runner.wm.horizon 8 \
  --runner.policy.batch-size 128 --runner.policy.horizon 40
```

复现命令在保存迭代 1,600 后退出，原实验执行到 3,200 次。验收使用以下四组初态与已配对生成的 F1 路径。

```bash
cd /home/hi/SuperMimic-Official/supermimic
F2=outputs/f2-steering-i1600-snapshot
export OMP_NUM_THREADS=2
for test in 'walk1_subject1 0 205 f1-arm-check' \
            'walk2_subject4 70 5599 f2-holdout-subject4-f1' \
            'walk4_subject1 60 3559 f2-holdout-walk4-f1' \
            'walk3_subject4 50 8045 f2-holdout-walk3-f1'; do
  read -r motion start walk baseline <<< "$test"
  output="outputs/f2-acceptance-$motion"
  .venv/bin/python -u scripts/eval_p0.py --checkpoint-dir "$F2" \
    --motion-file "../data/lafan1_official/$motion.npz" --stand-frame "$start" --walk-frame "$walk" \
    --record-posture True --steady-forward True --extra-forward-speeds '(0.22, 0.50)' \
    --multispeed-profile True --output-dir "$output" && \
  .venv/bin/python scripts/check_f2_gait.py "outputs/$baseline" "$output" --output-json "$output/criteria.json" || break
done
```

交互回放使用本机 checkpoint。已核对本地服务启动，连续步进了 144 步摇杆换速、松杆和想象轨迹重置，随后切换完整参考步进 8 步。打开终端打印的网页，点击 **Play**，在 **Velocity Command** 中设置前向、侧向及偏航。

```bash
cd /home/hi/SuperMimic-Official/supermimic
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/play_speed_test.py \
  --checkpoint-dir outputs/f2-steering-i1600-snapshot --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --device cpu --start-paused True --num-steps-rollout 32
```
