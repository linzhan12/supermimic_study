# F2.3 短起步参考与上下身接缝

沿用冻结 actor、方向库和原始起步用例。将 `scan_f2_references.py` 搜索时长改为 1.2 和 1.5 s，挑选短参考，在 MuJoCo 以每段 75 帧实测。库中的五个起步窗口来自 `walk1_subject5 / 101`、`walk3_subject4 / 74`、`walk3_subject3 / 1071`、`walk1_subject2 / 1009`、`walk3_subject1 / 3265`。与原起步参考在站立接缝的差异主要在肩肘和手腕。例如 `walk3_subject4 / 74` 最小全身角差约 0.336 rad，下肢角差约 0.098 rad。

实验组在实测与参考均双脚支撑、且髋膝踝角 RMS 小于 0.18 rad 时，允许上身跨过原接缝的全身角差预算。保持足位判据。腿和根的五次接缝维持 0.16 s，其他关节经不同的插值时长渐入参考。对照组更换同一批短片段，保持原接缝。

| 组别 | 起步段实测前向均值（m/s） | 起步段前向 MAE（m/s） |
| --- | ---: | ---: |
| 旧参考库与原接缝 | 0.056 | 0.294 |
| 短起步库与原接缝 | 0.201 | 0.228 |
| 短库，上身 0.24 s | 0.221 | 0.212 |
| 短库，上身 0.32 s | 0.217 | 0.213 |
| 短库，上身 0.48 s | 0.246 | 0.193 |

五组均保存完整 525 步。新组均选择 `start-short-s5`。0.48 s 组超过起步绝对均速门槛 0.20 m/s，前向 MAE 仍比原 F1 高约 0.144 m/s，远超额外容差 0.03。连续转向的偏航误差仍在，且收步后给斜走指令时保持旧停止片段。F2 未通过。

## 复现

用原参考文件和 `eval_f2.py --motion-only True --motion-window-s 1.5 --record-posture True` 重放指定起点，以 `--motion-frames '(起点,)'` 单测或按源批量重放。保存到 `outputs/f2-sonic-contact-scan/<动作名>`。用 `pack_f2_bank.py` 以 `start` 类型和区间 `0 75` 打包五条。其余片段从 [方向库](2026-11-24-f2-sonic-steering.md) 复制，跳过两条旧 `start-*.npz`。下例从 `supermimic/` 运行，另将 0.48 改为 0.24 或 0.32 验证另外两组。去掉该参数为短库对照。

~~~bash
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python -u scripts/eval_f2.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
  --cases '("stand_start",)' --reference-bank outputs/f2-sonic-bank-short-start \
  --planned-only True --trackability-weight 2 --prediction-steps 20 --pace-margin 0.25 \
  --start-upper-blend 0.48 --output-dir outputs/f2-sonic-short-start-48
~~~
