# F2 同步参考相位与全身拟合

上一轮物理示范筛选改善了脚尖朝向和支撑膝弯曲，但同一库的肩摆目标在慢速和中速过大。`outputs/f2-full-body-curated-i2000/` 的中速探针实测前向 `0.365 m/s`、侧摆 `8.8 cm`、躯干倾斜 `6.4°`，支撑脚尖为 `+8.8° / −10.3°`，肩摆 `34.1° / 33.3°`。该保存点作为后续起点。 [此前的全身对照](2026-11-24-f2-full-body-mimic.md)

## 实验设置

原训练从任意原始动捕姿态复位，独立速度命令从固定步态相位起步。新条件下，`70%` 的纯前进训练复位抽取实测速度接近指令的物理姿态，同时同步该示范相位。其余训练初态沿用原采样，运行中的命令切换保留原逻辑。四环境检查确认，站立及含侧向的命令保留原复位，纯前进复位的命令、相位、骨盆高度、方向和速度对应所选示范。肩俯仰、肩外展、肘屈伸另加 `0.65` 权重，使用目标关节角度和角速度。

共用教师对照 `2026-11-24_13-36-34` 沿用 `outputs/f2-dense-curated-bank/`，依然包含原来的大肩摆目标。独立教师对照保持腿部库，肩和肘改由原八段物理库 `outputs/f2-dense-action-bank/` 提供。两组均按同一速度和相位选目标，左右镜像同时交换关节并反转肩外展方向。肩肘的两项损失均直接使用替换后的目标。这样可以测出同一组更好的脚部示范与另一组较小肩摆示范能否一起被闭环策略学到。

在之前使用三维落脚目标的命令上增加 `--runner.forward-gait-demo-reset-fraction .70 --runner.forward-gait-arm-weight .65`，前向精度权重仍为 `.65`。第一个对照用 `--runner.forward-gait-demo-files "$LEG_DEMOS"`。第二个加 `--runner.forward-gait-arm-demo-files "$ARM_DEMOS"`。

```bash
cd /home/hi/SuperMimic-Official/supermimic
LEG_DEMOS="$(printf '%s\n' outputs/f2-dense-curated-bank/*.npz | paste -sd,)"
ARM_DEMOS="$(printf '%s\n' outputs/f2-dense-action-bank/*.npz | paste -sd,)"
```

## 共用教师结果

`2026-11-24_13-36-34` 完成 3,800 次迭代。三档物理平均前向分别为 `0.236 / 0.362 / 0.520 m/s`，中速骨盆侧摆 `8.9 cm`，躯干倾斜 `6.0°`，支撑膝 `27.5° / 28.6°`，脚尖偏航 `+9.7° / −9.6°`。慢速、中速的肩摆仍为 `38.5° / 37.6°` 和 `28.3° / 28.2°`。快走肩摆为 `12.0° / 13.2°`。站立和步行初态下的前进侧向筛查、慢速右脚离地门槛和两档换速末半秒仍有超限。**F2 未通过。**

## 独立肩肘教师对照

训练运行 `2026-11-24_13-42-15`。两组从同一个 `i2000` 保存点起步，均训练 3,800 次，使用相同的速度、身体目标和复位参数。唯一差别是独立肩肘目标。以下命令复现这一对照，其中输入轨迹的采集和打包见上轮报告。

```bash
cd /home/hi/SuperMimic-Official/supermimic
LAFAN='../data/lafan1_official'
MOTIONS="$LAFAN/walk1_subject2.npz,$LAFAN/walk2_subject3.npz,$LAFAN/walk3_subject2.npz,$LAFAN/walk3_subject3.npz,$LAFAN/walk3_subject5.npz,outputs/f2-external-accad-50hz/B1-standtowalk.npz,outputs/f2-external-cmu-50hz/16_34.npz"
LEG_DEMOS="$(printf '%s\n' outputs/f2-dense-curated-bank/*.npz | paste -sd,)"
ARM_DEMOS="$(printf '%s\n' outputs/f2-dense-action-bank/*.npz | paste -sd,)"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python -u src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --device cuda:0 --num-envs 128 --env.seed 0 --motion-file "$MOTIONS" \
  --runner.independent-velocity True --runner.posture-weight .25 --runner.moving-posture-scale 0 \
  --runner.init-checkpoint-dir outputs/f2-full-body-curated-i2000 \
  --runner.stop-encoder-init-dir logs/SuperMimic-GenCoord-G1/2026-11-23_16-35-09 \
  --runner.forward-gait-weight 1.75 --runner.forward-gait-action-limit .60 --runner.forward-gait-phase True \
  --runner.forward-gait-full-body True --runner.forward-gait-reflection 1.0 \
  --runner.forward-gait-weight-transfer True --runner.forward-gait-foot-orientation True \
  --runner.forward-gait-foot-placement True --runner.forward-foot-cycle-weight 0 \
  --runner.forward-precision-weight .65 --runner.forward-gait-command-fraction .55 --runner.forward-gait-hold-s '(2.6, 4.0)' \
  --runner.forward-gait-demo-reset-fraction .70 --runner.forward-gait-arm-weight .65 \
  --runner.forward-gait-demo-files "$LEG_DEMOS" --runner.forward-gait-arm-demo-files "$ARM_DEMOS" \
  --runner.steering-recovery-limit .30 --runner.steering-recovery-weight .90 --runner.train-steering-recovery False \
  --runner.actor-obs rep_state --runner.history-length 1 --runner.run-name f2-full-body-arm-teacher-s0 \
  --runner.max-iterations 3800 --runner.log-interval 100 --runner.save-interval 1000 --runner.policy.lr .0001 \
  --runner.replay-ratio 4 --runner.wm.num-batches 1 --runner.wm.batch-size 128 --runner.wm.horizon 8 \
  --runner.policy.batch-size 128 --runner.policy.horizon 64
```

四环境物理及想象奖励最大绝对差 `1.86×10⁻⁹`。233 个仓库测试通过。完整结果以 50 Hz MuJoCo 闭环回放为准。

### 闭环结果与保存点选择

中间第 2,000 次权重的三档实测平均前向为 `0.248 / 0.368 / 0.519 m/s`，肩摆分别为 `12.2° / 10.0°`、`12.8° / 13.1°`、`14.5° / 15.5°`。保留中速重心转移 `8.3 cm`、躯干倾斜 `6.5°`、支撑脚尖 `+9.2° / −9.7°`。最终第 3,800 次权重的肩摆仍在该范围附近，主场景慢速右脚却连续多次重新落地，严格完整摆动计数为 `2 / 6`，右脚落脚前后间距中位数 `3.7 cm`、中位离地 `5 mm`。因此继续保留较早保存点作为复查起点。本机为 `outputs/f2-full-body-arm-teacher-i2000/`，actor SHA-256 `e81a6f2a3937eb9868acb44dece17b83f44b0dce46210485f13da0e364f6c135`。

将同一早期权重在三条额外留出动作上复验，共四组、四十条闭环路线完成。八条完整参考路线逐字节等同各自的 F1。所有组的直走逐帧侧向误差和慢速、中速换速末半秒仍超限。`walk4_subject1` 和 `walk3_subject4` 的快走完整摆动计数为 `4 / 2`，快走在不同初态下的支撑切换仍不稳定。主场景走后松杆末半秒关节 RMS 从 F1 的 `0.359` 降至 `0.194 rad`，平均躯干倾斜由 `13.65°` 降至 `4.38°`。

主场景慢速与中速换速最后半秒平均侧向 `−0.128 / −0.158 m/s`，按一个完整指令相位周期（74 步及 64 步）取平均为 `+0.008 / +0.002 m/s`。周期平均抵消了部分步态侧向变化，历史筛查结果原样保留。**四组 F2 仍未通过。** 快走支撑脚尖约 `+16° / −17°`，相近速度的完整参考约 `+4° / 0°`。下一组先检查真实快走的支撑与落脚时刻是否能用完整周期稳定监督。

### 同一实测速度视频对照

另用 `0.332 m/s` 指令生成中间权重的对照轨迹。选择各段平均实际前向均为 `0.350 m/s` 的 85 帧，旧 F2 和本轮权重各取稳态，完整参考取留出 `walk2_subject4`。正面和侧面同步显示，速度标签为该画面的瞬时值。文件在 `supermimic/outputs/f2-full-body-audit/f2-matched-full-body-035.gif`，同名 PNG 是一帧预览。

| 轨迹 | 支撑脚尖 左 / 右（°） | 支撑膝 左 / 右（°） | 骨盆侧摆（cm） | 肩摆 左 / 右（°） |
| --- | ---: | ---: | ---: | ---: |
| 完整参考 | +2.5 / −6.3 | 19.7 / 28.9 | 8.5 | 12.4 / 6.7 |
| 旧 F2 | +38.0 / −36.1 | 18.1 / 15.6 | 1.0 | 27.1 / 27.2 |
| 本轮中间保存点 | +8.0 / −8.6 | 27.7 / 26.6 | 8.7 | 12.2 / 12.3 |

相近速度下新权重的脚尖、骨盆转移和左右肩摆的量级更接近完整参考。正面和侧面仍能看到新权重左支撑膝偏屈、对照右臂摆幅偏大的差异，快走的落脚失衡也未解决。回放无需 W&B。

```bash
cd /home/hi/SuperMimic-Official/supermimic
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
  --checkpoint-dir outputs/f2-full-body-arm-teacher-i2000 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
  --extra-forward-speeds '(0.332,)' --output-dir outputs/f2-full-body-arm-teacher-035-repro
MUJOCO_GL=egl EGL_PLATFORM=surfaceless .venv/bin/python scripts/audit_forward_gait.py \
  --trace 'Complete reference' outputs/f2-holdout-subject4-f1/stand_start_motion.jsonl 355 440 \
  --trace 'Previous F2' outputs/f2-steering-i1600-s0/steady_forward_scripted_velocity.jsonl 180 265 \
  --trace 'Latest F2 probe' outputs/f2-full-body-arm-teacher-035-repro/steady_forward_0p332_scripted_velocity.jsonl 176 261 \
  --replay-gif outputs/f2-full-body-audit/f2-matched-full-body-035.gif
```

快走的下一轮物理教师筛选和训练记录见[快走参考校准](2026-11-24-f2-fast-reference-calibration.md)。
