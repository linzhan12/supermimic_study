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
