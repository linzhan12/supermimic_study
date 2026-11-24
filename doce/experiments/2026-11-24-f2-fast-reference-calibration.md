# F2 快走参考校准

上一轮保留的 `outputs/f2-full-body-arm-teacher-i2000/` 在中速的骨盆侧摆、脚尖和摆臂已接近同速完整参考。`0.50 m/s` 下两条留出动作的完整摆动计数为 `4 / 2`，主场景实测 `0.519 m/s` 的脚尖偏航约 `+16° / −17°`。这一轮筛选速度接近、左右都能支撑和落地的快走训练片段，保留原肩肘教师。 [上一轮完整结果](2026-11-24-f2-phase-aligned-imitation.md)

按原始动捕的前向均值接近 `0.50 m/s`、侧向与偏航接近直行、髋和膝角度挑选三个训练录制者片段，每条回放两个 115 帧窗口。完整参考策略的六个物理回放全部完成。下表列出其中两个用于新库的片段。后者补充区间上沿较伸展的支撑腿。

| 录像与起始帧 | 实测前向 / 侧向（m/s） | 左 / 右完整摆动 | 左 / 右脚尖（°） | 左 / 右支撑膝（°） |
| --- | ---: | ---: | ---: | ---: |
| `walk3_subject2`，2674 | 0.514 / −0.012 | 2 / 2 | +6.7 / −12.3 | 29.8 / 29.0 |
| `walk2_subject3`，465 | 0.566 / −0.015 | 2 / 1 | +6.0 / −13.6 | 17.7 / 22.2 |

```bash
cd /home/hi/SuperMimic-Official/supermimic
for item in 'walk1_subject2 3594 10494 w1s2' 'walk2_subject3 11472 465 w2s3' \
            'walk3_subject2 2674 3138 w3s2'; do
  read -r clip first second label <<< "$item"
  OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
    --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
    --motion-file "../data/lafan1_official/$clip.npz" --stand-frame "$first" --walk-frame "$second" \
    --motion-only True --motion-window-s 2.3 --record-posture True --output-dir "outputs/f2-fast-oracle-$label"
done
.venv/bin/python scripts/pack_gait_teacher.py outputs/f2-dense-curated-fast-bank \
  outputs/f2-dense-calibrated-w1s5/{stand_start,walk_start}_motion.jsonl \
  outputs/f2-dense-calibrated-w2s3/stand_start_motion.jsonl \
  outputs/f2-dense-train-d2/stand_start_motion.jsonl \
  outputs/f2-fast-oracle-w3s2/stand_start_motion.jsonl \
  outputs/f2-fast-oracle-w2s3/walk_start_motion.jsonl
```

训练沿用[独立肩肘教师命令](2026-11-24-f2-phase-aligned-imitation.md#独立肩肘教师对照)，只调整腿部教师目录至 `outputs/f2-dense-curated-fast-bank/`，起点换为 `outputs/f2-full-body-arm-teacher-i2000/`，使用 `--runner.run-name f2-full-body-fast-teacher-s0 --runner.max-iterations 3000`。这是一轮续训筛查，若要把效果归因给快走教师库，需从同一保存点用原库续训相同迭代数。

训练 `2026-11-24_13-54-34` 完成，主场景三档前向实测 `0.233 / 0.357 / 0.510 m/s`。快走左右脚尖从续训前的 `+16.0° / −16.7°` 收到 `+12.9° / −13.7°`，支撑膝从 `32.7° / 33.4°` 降至 `30.0° / 31.4°`，完整摆动为 `4 / 3`，肩摆 `12.4° / 13.5°`。快走躯干倾斜从约 `6.5°` 增到 `8.3°`。中速脚尖 `+8.3° / −8.8°`、肩摆 `12.7° / 14.5°`、骨盆侧摆 `8.7 cm`。三档稳态筛查在主场景均通过，站立和走动初态下前进的逐帧侧向 MAE 以及慢速、中速换速末半秒仍超限。**F2 未通过。**

额外复验 `walk2_subject4`、`walk3_subject4` 和 `walk4_subject1`，三组的三档稳态脚部筛查均通过，包括上轮出现过的快走 `4 / 2` 失衡用例。四组的前进逐帧侧向误差和慢速、中速换速判据仍未通过，完整参考回归逐字节等同 F1。新校准在留出初态下改善了落脚稳定性。持续换速和快走的脚尖与支撑腿姿态仍有差距。

本机同实测速度回放为 `supermimic/outputs/f2-full-body-audit/f2-latest-matched-035.gif`。候选使用 `0.344 m/s` 指令，85 帧平均实际前向为 `0.350 m/s`，对应轨迹 `outputs/f2-full-body-fast-teacher-035/steady_forward_scripted_velocity.jsonl` 的行 `[207, 292)`。这段的脚尖 `+8.1° / −8.5°`、骨盆侧摆 `8.7 cm`、肩摆 `12.5° / 14.0°`，完整参考分别为 `+2.5° / −6.3°`、`8.5 cm`、`12.4° / 6.7°`。下轮起点为 `supermimic/logs/SuperMimic-GenCoord-G1/2026-11-24_13-54-34/`。先结合完整步态周期平均及地面轨迹复查换速偏差，再针对真实侧向偏移与快速支撑腿做对照训练。
