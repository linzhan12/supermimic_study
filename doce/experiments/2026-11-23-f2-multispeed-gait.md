# F2 多速度动作拟合

本轮检验速度分支能否从多个完整动作中学习相近速度下的步态，并保留起停、转向、斜移和原完整参考接口。上轮失败现象见 [F2 步态复查](2026-11-23-f2-forward-gait-audit.md)。

## 多段动作的逐帧先验

训练使用 LAFAN1 的 `walk1_subject2`、`walk2_subject3`、`walk3_subject2`、`walk3_subject3`、`walk3_subject5`，下游按人划分，留出 subject1 和 subject4。连续前走筛查采用约 0.5 秒的按片段中心速度窗口及连续 0.30 秒航向约束，保留前向 `0.16–0.66 m/s`、侧向绝对值小于 `0.13 m/s`、偏航绝对值小于 `0.22 rad/s` 的样本。五档原始帧数依次为 `946 / 797 / 216 / 52 / 80`。每档最多取 80 帧，加入左右镜像，再按命令速度比较关节角、角速度和骨盆高度。在物理采集和世界模型想象中使用同一评分。

旧 F2 的共享 decoder、动作 encoder、速度 encoder 和停步 encoder 均冻结。直走任务由新增的状态相关动作补偿学习髋、膝、踝、肩和肘目标，初始补偿为零。斜移、转向和零速时补偿为零。训练分布中用 `45%` 概率额外抽取 `0.19–0.55 m/s` 纯前向指令。种子 0、128 个实例、2,400 次采集迭代，共 307,200 条物理转移。运行 `2026-11-23_20-55-14`。

```bash
cd /home/hi/SuperMimic-Official/supermimic
LAFAN='../data/lafan1_official'
MOTIONS="$LAFAN/walk1_subject2.npz,$LAFAN/walk2_subject3.npz,$LAFAN/walk3_subject2.npz,$LAFAN/walk3_subject3.npz,$LAFAN/walk3_subject5.npz"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 --motion-file "$MOTIONS" \
  --runner.independent-velocity True --runner.posture-weight 0.25 --runner.moving-posture-scale 0 \
  --runner.arm-swing-control 0.45 --runner.arm-swing-roll 0.26 --runner.arm-swing-speed-gate 0.12 \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-23_17-22-01 \
  --runner.stop-encoder-init-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-35-09 \
  --runner.forward-gait-weight 0.9 --runner.forward-gait-action-limit 0.42 --runner.forward-gait-command-fraction 0.45 \
  --runner.actor-obs rep_state --runner.history-length 1 --runner.run-name f2-multispeed-prior-s0 \
  --runner.max-iterations 2400 --runner.log-interval 100 --runner.save-interval 600 --runner.policy.lr 0.00015 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 --runner.wm.horizon 8 \
  --runner.policy.batch-size 128 --runner.policy.horizon 32
```

## 实测与结论

在 `walk1_subject1` 从同一个初始状态回放，以三个固定速度的最后 4 秒计算真实步态。落脚间距、脚底离地峰值均为完整摆动的中位数，列内按左 / 右排序。

| 目标（m/s） | 实测前向旧 → 新（m/s） | 新膝摆幅（°） | 新落脚间距（cm） | 脚底离地旧 → 新（mm） |
| ---: | ---: | ---: | ---: | ---: |
| 0.22 | 0.244 → 0.252 | 5.0 / 9.5 | 8.5 / 2.7 | 1.6 / 6.9 → 1.7 / 0.8 |
| 0.35 | 0.403 → 0.370 | 6.5 / 13.2 | 11.5 / 5.8 | 3.1 / 7.3 → 1.7 / 1.1 |
| 0.50 | 0.534 → 0.501 | 11.0 / 15.7 | 12.3 / 13.1 | 3.1 / 8.7 → 1.4 / 4.6 |

九条回放跑满，完整参考未变化。速度更接近目标，髋膝摆幅和脚底抬升却进一步缩小，原筛查中左转前向误差增加 `0.04384 m/s`、右斜移偏航误差增加 `0.07054 rad/s`，均超过容差。两项摆臂幅度筛查失败。**本轮 F2 未通过。** 可按当前状态独立挑选最相近参考帧的奖励允许策略长期利用动作幅度较小的片段。增加全身逐帧样本未使策略复现连续左右支撑交换。后续对照需要显式推进相位，约束完整周期的摆动。

```bash
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-23_20-55-14 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
  --steady-forward True --extra-forward-speeds '(0.22, 0.50)' --output-dir outputs/f2-multispeed-prior-s0-replay
.venv/bin/python scripts/check_f2.py outputs/f1-arm-check outputs/f2-multispeed-prior-s0-replay
.venv/bin/python scripts/audit_forward_gait.py \
  --trace v22 outputs/f2-multispeed-prior-s0-replay/steady_forward_0p220_scripted_velocity.jsonl 150 350 \
  --trace v35 outputs/f2-multispeed-prior-s0-replay/steady_forward_scripted_velocity.jsonl 150 350 \
  --trace v50 outputs/f2-multispeed-prior-s0-replay/steady_forward_0p500_scripted_velocity.jsonl 150 350
```

## 外部动作筛查

从 Berkeley 的 G1 重定向 AMASS 下载了 ACCAD 四条步行动作和 CMU 五条标为慢走的试次。源文件使用 IsaacLab AMP 的 29 个 G1 关节、30 Hz、`wxyz` 四元数。`scripts/convert_amp_g1_motion.py` 按关节名映射，平移和关节采用样条插值，旋转采用球面插值，按 50 Hz 重新计算速度。转换后用相同走速与连续性门槛检测。 [Berkeley 重定向库](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1) · [IsaacLab AMP 文件格式](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions) · [CMU 慢走试次索引](https://mocap.cs.cmu.edu/search.php?maincat=3&subcat=2)

ACCAD `B1-standtowalk` 保留 9 帧，`B2-walktostand`、`B3-walk1`、`B10-walkturnleft45` 均未保留。CMU `16_34` 保留 29 帧，`07_04`、`07_05`、`08_04`、`08_11` 均未保留。文件上的慢走标签无法替代机器人实际尺寸下的速度筛查。短片段先保留供起停或相位实验；本轮训练仅用了上述 LAFAN1 文件。外部文件保存在被 Git 忽略的 `supermimic/outputs/f2-external-{accad,cmu}-50hz/`，源自 AMASS 的动作适用其学术研究许可和子库条款。 [AMASS 许可](https://github.com/nghorbani/amass/blob/master/LICENSE.txt) · [上游重定向来源及许可](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics)

## 连续相位和完整参考策略蒸馏

对有明显左右腿运动的示范，用双髋俯仰差及其角速度估计相位，保留原片和镜像。增加速度相关的连续相位，供策略、价值网络和步态奖励使用。物理与想象从同一相位和同一步命令出发，连续直走换速时相位保持连续。原完整参考策略在同一参考状态输出教师动作，新增补偿头蒸馏其腿和手臂动作，再通过闭环训练调整速度。零速、明显横移或转向指令下关闭补偿。四环境 CPU 预检把同一物理步末状态送入想象评分，奖励差最大 `4.66×10⁻⁹`，相位对齐后刷新命令。227 个测试通过。

运行 `2026-11-23_21-05-11` 从旧 F2 `17-22-01` 启动。训练候选的原 84 个 actor 张量逐项与旧 F2 相同，补偿头 10 个张量相对蒸馏起点全部更新。训练使用上轮五条本地动作，再加 `B1-standtowalk` 与 `16_34`。相位过滤后的五档帧数为 `706 / 770 / 223 / 64 / 80`。蒸馏 450 步，归一化动作 MSE 从初始抽样批的 `0.11935` 降至末批的 `0.00277`。闭环执行 3,600 次采集迭代，共 460,800 条物理转移。

```bash
MOTIONS="$MOTIONS,outputs/f2-external-accad-50hz/B1-standtowalk.npz,outputs/f2-external-cmu-50hz/16_34.npz"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 --motion-file "$MOTIONS" \
  --runner.independent-velocity True --runner.posture-weight 0.25 --runner.moving-posture-scale 0 \
  --runner.arm-swing-control 0.45 --runner.arm-swing-roll 0.26 --runner.arm-swing-speed-gate 0.12 \
  --runner.init-checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-23_17-22-01 \
  --runner.stop-encoder-init-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-35-09 \
  --runner.forward-gait-weight 1.2 --runner.forward-gait-action-limit 0.60 --runner.forward-gait-command-fraction 0.55 \
  --runner.forward-gait-phase True --runner.forward-gait-teacher-steps 450 \
  --runner.actor-obs rep_state --runner.history-length 1 --runner.run-name f2-multispeed-phase-s0 \
  --runner.max-iterations 3600 --runner.log-interval 100 --runner.save-interval 800 --runner.policy.lr 0.00012 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 --runner.wm.horizon 8 \
  --runner.policy.batch-size 128 --runner.policy.horizon 40
```

蒸馏后的物理探针在 `0.35 m/s` 指令下做出了 `42.7 / 40.2 cm` 的落脚间距及 `50.0 / 62.8 mm` 的离地峰值，膝摆幅达 `52.4° / 43.9°`，但平均前向速度为 `0.713 m/s`。闭环后跟速通过三档稳态和连续换速，离地高度及双足交替再次退化。

| 目标（m/s） | 平均前向（m/s） | 膝摆幅（°） | 落脚间距（cm） | 左右离地峰值（mm） |
| ---: | ---: | ---: | ---: | ---: |
| 0.22 | 0.219 | 8.0 / 14.6 | 13.2 / 4.5 | 0.4 / 2.1 |
| 0.35 | 0.350 | 18.6 / 16.4 | 19.1 / −2.8 | 1.6 / 2.2 |
| 0.50 | 0.487 | 27.7 / 18.8 | 26.2 / 7.8 | 2.1 / 3.7 |

连续 `0.22 → 0.50 → 0.35 → 0` 的各段末半秒前向速度是 `0.218 / 0.478 / 0.348 / 0.005 m/s`。所有回放跑满。原筛查还有左转侧向误差超限 `0.04039 > 0.03 m/s` 及摆幅和协调不达标。**F2 仍未通过。** 结果表明速度接口的动作补偿有复现跨步的能力，闭环奖励允许低抬脚和不对称的替代做法。下一轮需直接约束足部浮升和交替支撑，验证能否同时保留蒸馏后的运动幅度和闭环跟速。

新增 `check_f2_gait.py` 在原速度、停步、摆臂筛查基础上，检查三档前进的误差、膝摆幅、有效摆动次数及其对称性、落脚前后间距、脚底离地、换速和停止。门槛用于剔除已经观察到的贴地、小步和左右失衡，实际自然度仍由同速度范围完整参考的配对视频复核。

```bash
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-23_21-05-11 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
  --steady-forward True --extra-forward-speeds '(0.22, 0.50)' --multispeed-profile True \
  --output-dir outputs/f2-multispeed-phase-s0-replay
.venv/bin/python scripts/check_f2_gait.py outputs/f1-arm-check outputs/f2-multispeed-phase-s0-replay
```

外部样本另以 `convert_amp_g1_motion.py --speed-scale .65` 做纯时间缩放筛查。保持空间姿态轨迹并降低速度后，四条原慢走样本仍仅有 `07_05` 的 6 帧通过持续前走筛查。时间缩放改善速度覆盖的潜力受侧向、转向及连续性限制，本轮没有使用这些重定时文件训练。

## 足部几何监督

用 G1 每只脚的碰撞胶囊最低点计算鞋底离地，以连续相位指定当前摆动腿，并按指令速度与步频确定左右落脚前后间距。奖励同时覆盖离地高度、双脚前后差与支撑脚平移。摆动中期的目标高度在测试速度范围内约 `39–50 mm`。两条训练动作的同相位原始脚底左右高度差中位数约为 `10–34 mm`，构成试跑量级的参照。主动参与的正向运动学与 MuJoCo 原生碰撞胶囊计算最大差 `1.96×10⁻⁸ m`。同一物理状态与想象评分最大差 `2.79×10⁻⁹`。

运行 `2026-11-23_21-13-22`，从 `21-05-11` 接续，使用相同七条本地文件，保持原 encoder 和 decoder 冻结。步态先验权重 `0.80`，足部周期权重 `1.45`，摆臂权重 `0.20`。`128 × 3,800` 次物理交互。蒸馏已有相位模型时完成，本轮未再次蒸馏。仍用相同三档与连续换速作筛查。

| 权重版本 | 慢走落脚间距和离地 左 / 右 | 中速落脚间距和离地 左 / 右 | 快走落脚间距和离地 左 / 右 |
| --- | --- | --- | --- |
| 迭代 800 | 4.6 / 14.1 cm，12.8 / 10.5 mm | 20.5 / 21.0 cm，38.9 / 46.6 mm | 27.4 / 15.1 cm，40.7 / 19.1 mm |
| 迭代 2,400 | 14.1 / 1.9 cm，28.5 / 22.2 mm | 22.9 / −0.1 cm，42.1 / 24.5 mm | 26.7 / 30.0 cm，45.5 / 48.8 mm |
| 最终 | 15.9 / 17.2 cm，32.2 / 41.9 mm | 1.3 / 18.7 cm，23.9 / 47.9 mm | 28.8 / 24.8 cm，48.2 / 48.7 mm |

最终三档实测前向速度为 `0.234 / 0.334 / 0.485 m/s`，均通过跟速，摆臂也通过原筛查。连续换速和停止用例完成。左转前向和偏航 MAE 仍超限，转向头 0.16 秒的平均偏航达到 `1.237 rad/s`。中速左脚有同侧重复落脚和较短的有效间距。**F2 仍未通过。** 直接足部监督显著改善了三档离地量，中速左右步幅及走转切换仍须修复。按速度逐档审查发现更久训练可以改善一个速度下的步态，并同时损害另一个速度下的左右协调。

复现时使用上节命令和同一 `MOTIONS`，起点换为 `2026-11-23_21-05-11`，去掉 `--runner.forward-gait-teacher-steps`，将 `--runner.forward-gait-weight` 设为 `0.80`，新增 `--runner.forward-foot-cycle-weight 1.45` 与 `--runner.arm-swing-weight 0.20`，将学习率设为 `0.00014`，迭代数设为 `3800`。回放目录为 `outputs/f2-multispeed-footcycle-s0/`。所有检查包括支撑脚 site P90 不超过 `0.25 m/s`、双脚同时离地采样不超过 `10%`、膝摆幅 `18–70°` 和摆动离地 `8–100 mm`，配对视频仍用于判断摆动与支撑是否自然。

### 镜像推理探针

在直走时把本体状态镜像为另一侧支撑问题，将相位移动半个周期，运行原策略并把输出反射回实际关节坐标，再与原输出混合。原完整参考、零速、原地转向和明显侧移走原路。采用 `21-13-22` 相同权重测两个混合系数，期间未训练。中速 `0.35 m/s` 工况下，`0.5` 混合将落脚间距从 `1.3 / 18.7 cm` 变为 `20.3 / 26.7 cm`，`1.0` 混合为 `25.3 / 21.1 cm`。对应实际前向分别升到 `0.485` 和 `0.537 m/s`，两组也均存在非交替落脚或原回归超限。镜像提供了对抗单侧短步的线索，速度和支撑连续性仍需要闭环优化。

## 镜像下重新跟速

运行 `2026-11-23_21-23-30`，从 `21-13-22` 接续。保持足部周期奖励和七条动作的设置，以 `--runner.forward-gait-reflection 1.0` 启用镜像，新增 `--runner.forward-precision-weight 0.85`，令额外采样的直走命令保持 `--runner.forward-gait-hold-s '(2.6, 4.0)'` 秒。其他运动指令仍为 1–2 秒。学习率 `0.00012`，4,800 次采集迭代，CPU 回放对齐上一轮。

第 1,000 次迭代已通过三档前进的全部定量步态筛查、连续换速和摆臂筛查，在 `0.35 m/s` 稳态中左右肩摆幅为 `27.6° / 26.9°`，肩差与髋差相关 `−0.992`。中速视频同时显示交替跨步和脚离地，完整参考有更大的前后间距。该 checkpoint 仍有左转前向、左转偏航、右斜移偏航三项超限。

| 目标（m/s） | 迭代 1,000 平均前向（m/s） | 落脚间距 左 / 右（cm） | 离地峰值 左 / 右（mm） |
| ---: | ---: | ---: | ---: |
| 0.22 | 0.224 | 12.5 / 19.9 | 30 / 27 |
| 0.35 | 0.347 | 20.0 / 23.4 | 41 / 43 |
| 0.50 | 0.489 | 25.2 / 28.1 | 44 / 46 |

第 2,000 次迭代也通过三档步态与连贯换速筛查，转向三项超限。`outputs/f2-reflection-i1000-snapshot/` 和 `outputs/f2-reflection-i2000-snapshot/` 保存了本地中间权重，包含 actor、世界模型和回放配置。`outputs/f2-reflection-i1000-s0/forward-gait-comparison.gif` 用同一视角比较完整参考、旧 F2 和本轮中间权重。

最终 `4,800` 次版本三档前向为 `0.225 / 0.365 / 0.512 m/s`，中速落脚间距约 `22.6 / 22.6 cm`，但左转前向、左转偏航和右斜移偏航的误差较 F1 分别上升 `0.1552 m/s`、`0.1975 rad/s` 和 `0.1028 rad/s`。第 1,000 次版本另在留出的 `walk2_subject4`（初态 `70 / 5599`）与 `walk4_subject1`（初态 `60 / 3559`）各通过了三档步态和连续换速筛查。两个额外用例均有走转误差超限。现象支持冻结直走并补训转向及斜移，完整 F2 尚未通过。

后续独立补训了转向和侧移，第 1,600 次保存点通过四组完整验收。 [最终验收和复现命令](2026-11-23-f2-multispeed-acceptance.md)
