# F2 期望动作与全身步态复查

用户在对照完整参考后指出，上一版的步态仍明显失真。F2 保持未通过状态。此前闭环筛查保留，用于跟速、交替离地、连续换速、停步及完整参考回归。整套姿态和支撑切换以相近实测速度的完整参考为对照。 [上一版数值结果](2026-11-23-f2-multispeed-acceptance.md)

## 相近速度下的物理回放

从下游留出 `walk2_subject4` 的完整参考回放选择两个前进片段。中速是 `outputs/f2-holdout-subject4-f1/stand_start_motion.jsonl` 的行 `[355, 440)`，对应帧 `[425, 510)`。快走是 `outputs/f2-natural-dense-reference/stand_start_motion.jsonl` 的行 `[44, 129)`，对应帧 `[3763, 3848)`。两组旧 F2 取持续前进最后四秒。角度左右分别测量。

| 轨迹 | 实测前向（m/s） | 骨盆高度中位数（m） | 横向速度标准差（m/s） | 髋侧摆中位数 左 / 右（°） | 支撑期膝角中位数 左 / 右（°） | 肩俯仰摆幅 左 / 右（°） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整参考，中速 | 0.350 | 0.763 | 0.122 | +0.4 / −2.3 | 19.7 / 28.9 | 12.4 / 6.7 |
| 旧 F2，中速 | 0.347 | 0.787 | 0.022 | +11.4 / −10.2 | 16.1 / 16.4 | 27.6 / 26.9 |
| 完整参考，快走 | 0.490 | 0.759 | 0.081 | −2.4 / +6.1 | 18.5 / 31.3 | 13.3 / 9.9 |
| 旧 F2，快走 | 0.489 | 0.788 | 0.021 | +10.9 / −9.4 | 15.7 / 14.5 | 28.8 / 29.5 |

旧 F2 的骨盆偏高，横向重心转移弱，两侧髋外展偏置和肩摆过大。一次跨步的整体身体行为与完整参考有明显差异。之前只看幅度门槛、足部事件、跟速和停步，无法识别这种组合。参考片段短且来自同一录制者，数值用于定位问题和配对视图。慢走 `0.22 m/s` 暂缺相近实测速度并覆盖足够步态周期的完整参考窗口。

## 全身拟合改动

从上一版对照初始化，将前进补偿的输出扩至全部 G1 关节。原有 18 个输出保留，新增输出初始为零。完整参考接口、共享 decoder、停步和转向补偿权重冻结。用同一批五条 LAFAN1、两条外部步行动作，按指令速度和连续相位生成整套关节角、关节角速度、骨盆高度与倾角、横向及竖直速度目标。目标选择只依赖指令和相位，减少每步挑选容易模仿的参考帧所造成的塌缩。训练移除手设肩摆控制，降低足底正弦目标与额外瞬时跟速项的相对权重，为模仿姿态和重心转移留出空间。

四环境预检把同一批真实物理步末状态送入想象奖励，最大绝对差 `3.73×10⁻⁹`。相位与命令刷新时序一致。运行 `2026-11-24_12-49-20` 从 `outputs/f2-steering-i1600-snapshot/` 开始，使用 128 个实例，完成 4,300 次外层迭代和 550,400 条物理转移。

### 原始动捕监督的结果

迭代零探针只移除手设肩动作，扩充的新输出为零。中速肩摆从 `27.6° / 26.9°` 降至 `22.4° / 22.9°`，但髋外展仍约 `+12° / −11°`，骨盆仍高于完整参考。它通过旧数值筛查，说明该筛查无法识别这类姿态问题。

完成原始动捕拟合后，三档平均前向速度为 `0.220 / 0.353 / 0.500 m/s`。中速髋侧摆变为 `+5.6° / −5.2°`，肩摆为 `6.5° / 6.9°`，横向速度标准差为 `0.032 m/s`。同时骨盆降到 `0.746 m`，支撑膝长期维持 `45.6° / 46.0°`，形成新的屈膝偏差。两侧支撑膝比在相近实测速度的完整参考中的 `19.7° / 28.9°` 弯得更多。旧数值筛查报出肩摆小于原门槛。**整套 F2 步态仍未通过。**

检查训练目标发现，仅根据原始数据的速度范围和短窗筛选后，`0.35 m/s` 的合成周期膝角中位数为 `49.9°`，横向速度标准差为 `0.046 m/s`。所选原始样本的步态分布本身就与期望物理参考不同。学习走向屈膝与跟随该目标一致。

```bash
cd /home/hi/SuperMimic-Official/supermimic
LAFAN='../data/lafan1_official'
MOTIONS="$LAFAN/walk1_subject2.npz,$LAFAN/walk2_subject3.npz,$LAFAN/walk3_subject2.npz,$LAFAN/walk3_subject3.npz,$LAFAN/walk3_subject5.npz,outputs/f2-external-accad-50hz/B1-standtowalk.npz,outputs/f2-external-cmu-50hz/16_34.npz"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 --motion-file "$MOTIONS" \
  --runner.independent-velocity True --runner.posture-weight .25 --runner.moving-posture-scale 0 \
  --runner.init-checkpoint-dir outputs/f2-steering-i1600-snapshot \
  --runner.stop-encoder-init-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-35-09 \
  --runner.forward-gait-weight 1.75 --runner.forward-gait-action-limit .60 --runner.forward-gait-phase True \
  --runner.forward-gait-full-body True --runner.forward-foot-cycle-weight .45 --runner.forward-gait-reflection 1.0 \
  --runner.forward-precision-weight .28 --runner.forward-gait-command-fraction .55 --runner.forward-gait-hold-s '(2.6, 4.0)' \
  --runner.steering-recovery-limit .30 --runner.steering-recovery-weight .90 --runner.train-steering-recovery False \
  --runner.actor-obs rep_state --runner.history-length 1 --runner.run-name f2-full-body-prior-s0 \
  --runner.max-iterations 4300 --runner.log-interval 100 --runner.save-interval 1000 --runner.policy.lr .0001 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 --runner.wm.horizon 8 \
  --runner.policy.batch-size 128 --runner.policy.horizon 48
```

### 构建完整参考的物理示范库

改用原完整参考策略在训练录制者的三条片段中执行的八个四秒 MuJoCo 片段。只使用成功完整回放，保持为八个独立序列。骨盆和关节位置直接取自物理输出，关节角速度、骨盆竖直速度和横滚俯仰角速度按 50 Hz 序列差分。选样速度改为 1.48 秒滑动窗口，允许一次迈步中有横向重心转移。五个速度档保留 `41 / 126 / 105 / 147 / 151` 个候选帧。`0.35 m/s` 的合成目标膝角中位数变为 `32.7°`，横向速度标准差为 `0.138 m/s`。全身奖励读物理示范库，仿真复位仍走原始训练数据。没有使用表中的留出对照片段训练。

```bash
cd /home/hi/SuperMimic-Official/supermimic
export OMP_NUM_THREADS=2
for item in 'walk2_subject3 2028 432 d1' 'walk3_subject2 7722 3667 d2' \
            'walk3_subject3 1496 2222 d3' 'walk2_subject3 3651 627 d4'; do
  read -r clip start walk label <<< "$item"
  .venv/bin/python -u scripts/eval_p0.py \
    --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
    --motion-file "../data/lafan1_official/$clip.npz" --stand-frame "$start" --walk-frame "$walk" \
    --motion-only True --motion-window-s 4 --record-posture True --output-dir "outputs/f2-dense-train-$label"
done
.venv/bin/python scripts/pack_gait_teacher.py outputs/f2-dense-training-bank outputs/f2-dense-train-d{1,2,3,4}/*_motion.jsonl
DEMO_FILES="$(printf '%s\n' outputs/f2-dense-training-bank/*.npz | paste -sd,)"
```

物理示范实验沿用上述训练命令，将初始化设为迭代零的 `outputs/f2-full-body-i0-snapshot/`，追加 `--runner.forward-gait-demo-files "$DEMO_FILES"`，并将迭代数改为 `3900`。迭代零探针的世界模型来自上一版初始化，策略完成了全身动作输出扩展，尚未更新全身补偿参数。

运行 `2026-11-24_12-58-00` 完成 3,900 次迭代。中速实测 `0.353 m/s`，双侧支撑膝 `34.0° / 34.3°`，骨盆高度 `0.765 m`，相对原始动捕试验的 `45.6° / 46.0°`、`0.746 m` 接近配对参考。横向速度标准差仍只有 `0.049 m/s`，躯干倾斜中位数 `10.9°`，髋侧摆中位数 `+7.3° / −7.0°`。在两视角对照下，支撑期仍有外展和躯干前倾偏差。旧筛查另报摆臂协调及快走左右有效摆动计数 `4 / 2` 失败。**F2 未通过。**

### 分开约束直走跟速和横向重心转移

在连续纯前进时，用户的横向指令应约束持续漂移，逐帧把横向速度压到零会干扰每步的重量交换。本对照仍保留前向、偏航、斜移和转向原奖励。在纯前进有效区域，从跟速项淡出瞬时横向速度误差，把横向速度交给完整参考相位监督。增加实际躯干朝向的监督，保留旧筛查和两视角回放，以区分重心转移与漂移。四环境物理及想象奖励最大绝对差 `3.73×10⁻⁹`。230 个仓库测试通过。

本对照从物理示范迭代 1,000 `outputs/f2-full-body-physical-i1000/` 开始，使用同一物理示范库，启用 `--runner.forward-gait-weight-transfer True`，将额外瞬时精度权重设为 `0`，将 `--runner.policy.horizon` 改为 `64` 步（1.28 秒），迭代数改为 `3500`。共享 decoder 和转向补偿权重仍冻结。训练运行 `2026-11-24_13-04-15`。

第 2,000 次保存点的中速探针实测前向 `0.349 m/s`，平均横向 `−0.001 m/s`。扣除地面直线路径后，骨盆侧向摆幅（P95−P05）从旧版的 `1.2 cm` 增至 `8.5 cm`，与配对完整参考的 `8.5 cm` 接近。躯干倾斜中位数从物理示范版最终的 `10.9°` 降至 `9.3°`，仍高于完整参考中位数。训练目标加入躯干测量后，横向平衡动作有所恢复，完整姿态仍有差距。

最终 3,500 次权重在三档实测 `0.230 / 0.357 / 0.509 m/s`，稳态横向均值分别为 `−0.024 / 0.000 / −0.003 m/s`。相同三档的骨盆侧向摆幅为 `15.1 / 8.1 / 6.4 cm`，中速改善保留，低速摆动偏大。旧逐帧侧向 MAE 在两类前进场景超限，低速与回到中速的最后半秒仍有换速超限，快走有同侧重复落脚，**本组 F2 未通过**。仅恢复横向速度变化仍需要检查实际漂移、步态周期及换速。

另补充了慢速参考，以原完整参考策略回放留出 `walk3_subject1` 的帧 `461–545`。实测前向 `0.220 m/s`、平均横向 `0.006 m/s`，骨盆侧摆幅 `8.8 cm`，支撑膝中位数 `29.4° / 34.0°`，脚尖偏航 `+7.9° / −13.1°`。此前缺少低速配对，这段仅用于评测，不进入训练示范库。可使用 `--motion-only True --motion-window-s 3.2 --stand-frame 3622 --walk-frame 461` 复现，目标窗口为 `walk_start_motion.jsonl` 的行 `[0, 85)`。

### 支撑脚朝向

相近速度的正面画面显示，旧版支撑脚脚尖明显向两边外转。用脚部 site 的前轴投影到骨盆航向坐标测量，`0.35 m/s` 的完整参考左、右支撑脚脚尖偏航中位数为 `+2.5° / −6.3°`，上一版为 `+37.7° / −36.5°`。纯物理示范版改善为 `+18.2° / −19.4°`；重心转移版第 2,000 次保存点为 `+17.5° / −19.5°`。

新增 `--runner.forward-gait-foot-orientation True`。通过 G1 正向运动学，直接拟合在同一速度、相位下示范脚掌相对航向的前轴，包含脚尖偏航与脚掌俯仰。已有脚底高度奖励和原回归检查保留。此轮从 `outputs/f2-full-body-transfer-i2000/` 开始，沿重心转移配置继续训练 3,100 次，运行 `2026-11-24_13-09-28`；物理与想象的评分误差仍小于 `4×10⁻⁹`。
