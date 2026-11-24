# F2.3 转向类别过滤与站立直入斜走对照

保持短起步库和 0.48 s 上身接缝。候选的片段体轴速度经固定参考实测偏差校准，按横向和偏航的符号从同方向类别检索。第二组允许站立或收步中在实测双脚支撑时直接进匹配方向的走步，沿用腿部角差和足位兼容。第三组加双支撑收步回退。

| 组别 | 1.5 s 起步前向（m/s） | 左/右转实际偏航（rad/s） | 左/右斜走实际横向（m/s） | 最后松杆前向均值（m/s） |
| --- | ---: | --- | --- | ---: |
| 短起步 0.48 s | 0.246 | +0.049 / −0.101 | −0.001 / +0.002 | 0.001 |
| 方向过滤 | 0.246 | −0.055 / +0.719 | +0.065 / +0.133 | 0.216 |
| 加直入 | 0.246 | −0.055 / +0.719 | +0.065 / +0.133 | 0.216 |
| 加直入与收步回退 | 0.246 | −0.055 / +0.719 | +0.027 / +0.027 | 0.098 |

类别过滤改为只选左转参考后，实际左段却出现反向。右转无接缝兼容的同类片段，整个右转段继续旧左转。无回退时两组始终走出上一片段，直入未触发预期路线。回退后能于最后一次松杆接入方向库停步；斜移速度和换向仍差。**该过滤未改善 F2**。保留诊断开关，后续不作为默认做法。预演目前主要衡量能否跟踪所给参考，下一步应该直接比较生成状态中的实际平移和偏航指令误差，以及它对脚接触的影响。

## 复现

从 `supermimic/` 运行，使用 [短起步库](2026-11-24-f2-sonic-start-support.md)。

~~~bash
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
for name in intent direct direct-stop; do
  extra=()
  if [ "$name" != intent ]; then extra+=(--direct-steering-start True); fi
  if [ "$name" = direct-stop ]; then extra+=(--contact-stop-duration 0.40 --contact-stop-max-foot-speed 0.25); fi
  .venv/bin/python -u scripts/eval_f2.py \
    --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
    --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
    --cases '("stand_start",)' --reference-bank outputs/f2-sonic-bank-short-start \
    --planned-only True --trackability-weight 2 --prediction-steps 20 --pace-margin 0.25 \
    --start-upper-blend 0.48 --directional-selection True "${extra[@]}" \
    --output-dir "outputs/f2-sonic-$name"
done
~~~
