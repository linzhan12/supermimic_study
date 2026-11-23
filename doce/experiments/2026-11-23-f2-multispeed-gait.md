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
