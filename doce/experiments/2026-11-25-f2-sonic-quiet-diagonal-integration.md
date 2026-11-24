# F2 低摆臂三档与斜走周期联合回归

将[定足斜走周期](2026-11-25-f2-sonic-retargeted-diagonal.md)合入[低摆臂快档和快速起步参考库](2026-11-25-f2-sonic-quick-start-quiet-cycle.md)。保留独立三档在原隔离协议下的通过记录。联合评测使用同一原策略、0.06 rad 方向适配器、实测支撑接缝和世界模型排序。

| 组别 | 起步 1.5 s 平均前速 m/s | 左／右转平均偏航 rad/s | 左斜走 `vx, vy` m/s | 右斜走 `vx, vy` m/s | 末半秒平面速度 m/s |
| --- | --- | --- | --- | --- | --- |
| 原启动／主初态 | 0.055 | +0.063／+0.429 | 0.068, +0.053 | 0.201, +0.032 | 0.292 |
| 短启动／主初态 | 0.284 | +0.354／−0.369 | 0.047, −0.025 | 0.185, −0.063 | 0.003 |
| 短启动／留出 | 0.299 | +0.333／−0.373 | −0.009, +0.106 | 0.159, −0.156 | 0.057 |

主初态标准筛查分别有 17 和 19 项失败，短启动留出有 17 项失败。短启动使两条初态的左右转方向正确、末次松杆大幅减速；联合后左侧仍至少有一轴反号，右侧幅度下降。主初态短启动组还出现低速着地间距不足和中速双脚摆步失衡，连续换速的慢档也未稳定。留出短启动组低速完整摆步为 2/6，出现同侧重复着地，右脚离地中位数只有 0.003 m。三组原始站立与行走 F1 全参考轨迹分别与各自基线逐字节一致。F2 未通过。

## 复现

从 `supermimic` 运行。以 `outputs/f2-sonic-bank-{neutral,quick}-quiet-fast` 为底删除旧稳态 `diagonal-*.npz`，保留方向启动和转弯片段，加入 `outputs/f2-sonic-bank-strong-diagonal/cycle-*.npz`。将短启动合成库命名为 `outputs/f2-sonic-bank-quick-quiet-diagonal`。

```bash
export OMP_NUM_THREADS=2 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/supermimic-f2-mpl
.venv/bin/python scripts/eval_f2.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --reference-bank outputs/f2-sonic-bank-quick-quiet-diagonal \
  --record-posture True --trackability-weight 2 --prediction-steps 20 \
  --pace-margin 0.25 --command-cost-weight 3 --candidate-entries 1 \
  --restart-from-rest True --heading-strength 0.58 --directional-selection True \
  --contact-launch-duration 0.60 --rank-contact-launch True \
  --contact-launch-forward-lead 0.08 --contact-stop-duration 0.40 \
  --contact-stop-max-foot-speed 0.25 --axial-cycle-ranking True \
  --steering-measured-seam True --project-contact-seam True --rank-contact-seam True \
  --contact-adapter outputs/f2-sonic-adapter-06/contact_adapter.pt \
  --adapt-ranking True --adapter-steering-only True --diagonal-heading-lock True \
  --axial-heading-lock True --axial-heading-on-change True --contact-stop-capture True \
  --early-steering-handoff True --direction-cost-weight 4 \
  --start-upper-blend 0.48 --start-search-time 0.54 \
  --steady-forward True --extra-forward-speeds '(0.22,0.50)' --multispeed-profile True \
  --output-dir outputs/f2-sonic-diagonal-integrated-quick-main
.venv/bin/python scripts/check_f2_gait.py outputs/f2-sonic-current-f1 \
  outputs/f2-sonic-diagonal-integrated-quick-main \
  --output-json outputs/f2-sonic-diagonal-integrated-quick-main/criteria.json
```

原启动组改用 `f2-sonic-bank-neutral-quiet-diagonal` 并去掉短启动的上身混合及预搜索参数；留出组改用 `walk1_subject2.npz --stand-frame 50 --walk-frame 231`，用 `outputs/f2-sonic-f1-holdout` 作为 F1 数值基线。记录在 `outputs/f2-sonic-diagonal-integrated-{neutral-main,quick-main,quick-holdout}`。

0.10 rad 方向适配也补测了末次松杆。支撑捕获单独使主／留出的末半秒平面速度为 0.417／0.233 m/s，右斜走横速保持 −0.210／−0.190 m/s。再开启接触刷新后末半秒为 0.053／0.055 m/s，但第一段松杆改变了后续入口，右斜走横速为 −0.079／−0.102 m/s。轨迹位于 `outputs/f2-sonic-steer10-stop-{main,holdout}-{capture,refresh}`，联合组仍用 0.06 rad。
