# F2 快走参考的速度重采样

同速正侧面对照中，常规快走的左支撑腿比完整参考多弯约 `10°`。训练库里另有较伸展的 `walk2_subject3` 快走片段，完整参考策略实测 `0.566 m/s`，超过当前 `0.50 m/s` 档。将同一训练片段作小幅时间重采样，再用完整参考策略执行，验证伸展姿态能否在更接近的实测速度下实现。 [同速支撑相对照](2026-11-24-f2-fast-stance-tracking.md)

截取原始轨迹 `[445,640)`，关节及根位置三次插值，四元数用球面插值，再计算 50 Hz 速度和角速度。对每个倍率做两个 2.4 秒的物理回放。此处选取起点较早的一组对照。

| 播放速度倍率 | 实测前向（m/s） | 有效摆动 左 / 右 | 支撑膝 左 / 右（°） | 脚尖 左 / 右（°） |
| --- | ---: | ---: | ---: | ---: |
| 0.86 | 0.476 | 1 / 1 | 17.0 / 25.8 | +3.9 / −12.5 |
| 0.90 | 0.501 | 1 / 1 | 17.5 / 24.7 | +5.5 / −12.9 |
| 0.94 | 0.531 | 0 / 1 | 17.1 / 23.3 | +6.4 / −13.3 |

`0.90` 的物理回放同时覆盖目标前向速度和双侧换步，保留较伸展的支撑姿态。其骨盆侧向范围 `10.9 cm` 和右脚外转仍偏大，作为多片段教师库的局部补充。训练时用该回放替换同一来源的原速度片段，避免重复计入。留出视频和其它原参考作为评测。

```bash
cd /home/hi/SuperMimic-Official/supermimic
.venv/bin/python scripts/retime_g1_motion.py ../data/lafan1_official/walk2_subject3.npz \
  outputs/f2-retimed-source/w2s3-090.npz --start-frame 445 --end-frame 640 --speed-scale .90
OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file outputs/f2-retimed-source/w2s3-090.npz --motion-only True --motion-window-s 2.4 \
  --stand-frame 22 --walk-frame 37 --record-posture True --output-dir outputs/f2-retimed-w2s3-090
.venv/bin/python scripts/pack_gait_teacher.py outputs/f2-dense-retimed-stance-bank \
  outputs/f2-dense-calibrated-w1s5/{stand_start,walk_start}_motion.jsonl \
  outputs/f2-dense-calibrated-w2s3/stand_start_motion.jsonl \
  outputs/f2-dense-train-d2/stand_start_motion.jsonl \
  outputs/f2-fast-oracle-w3s2/stand_start_motion.jsonl \
  outputs/f2-retimed-w2s3-090/stand_start_motion.jsonl
```

闭环对照采用同一快走支撑与侧向联合监督配置、起点、种子和 2,400 次迭代。只替换腿部教师中上述一个片段，目录改为 `outputs/f2-dense-retimed-stance-bank/`，运行名 `f2-fast-retimed-stance-s0`。
