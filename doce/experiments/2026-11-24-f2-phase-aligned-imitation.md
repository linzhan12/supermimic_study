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
