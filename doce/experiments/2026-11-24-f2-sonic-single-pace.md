# F2.3 单周期候选与双支撑停步对照

同一起点、同目标和同冻结跟踪器。只保留每档一条已校准的全身周期，沿用两条起步和三条停步。运动学与 20 步世界模型预演分别运行。前者中速筛查仍有左右摆动不平衡、重复同侧着地、落脚距离和离地高度四项失败；预演组在同初态的 `check_gait` 无失败。预演组慢、中、快实测前向 `0.242 / 0.358 / 0.475 m/s`，有效摆动 `2/3、2/2、2/2`。换速与收步各段末半秒前向为 `0.161 / 0.418 / 0.402 / −0.001 m/s`。旧三档、脚部和连续换速筛查这一子集已通过，转向、斜走和全身视觉验收仍待联合验证。

另用 [方向库](2026-11-24-f2-sonic-steering.md) 尝试将确有实际双支撑且前后计划共支撑的停步接缝全身 RMS 上限由 0.26 调到 0.37 rad。原失败片段和最近的双支撑停步参考姿态相差约 0.358 rad。放宽后两段松杆仍与基线结果相同，最后一次继续使用转向参考，整段平均前向 0.325 m/s。需要补匹配支撑脚位置及落点时序的过渡桥，不宜继续单独放宽全身角度门槛。

## 复现

以下从 `supermimic/` 目录运行。先建立 [周期库](2026-11-24-f2-sonic-cycle-plan.md)。

~~~bash
mkdir outputs/f2-sonic-bank-single-pace
for clip in slow-a medium-a fast-a start-a start-b stop-a stop-b stop-c; do
  cp -- "outputs/f2-sonic-bank-cycles/$clip.npz" outputs/f2-sonic-bank-single-pace/
done
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
CKPT=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
for item in 'kin 0 10' 'wm 2 20'; do
  read -r name weight horizon <<< "$item"
  .venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
    --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
    --steady-forward True --extra-forward-speeds '(0.22,0.50)' --multispeed-profile True \
    --cases '("steady_forward","steady_forward_0p220","steady_forward_0p500","multispeed_profile")' \
    --reference-bank outputs/f2-sonic-bank-single-pace --planned-only True \
    --trackability-weight "$weight" --prediction-steps "$horizon" --pace-margin 0.25 \
    --output-dir "outputs/f2-sonic-single-pace-$name"
done
.venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
  --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
  --cases '("stand_start",)' --reference-bank outputs/f2-sonic-bank-steering \
  --planned-only True --trackability-weight 2 --prediction-steps 20 --pace-margin 0.25 \
  --stop-double-support-pose 0.37 --output-dir outputs/f2-sonic-stop-double
~~~

保存点为 `outputs/f2-sonic-single-pace-{kin,wm}` 和 `outputs/f2-sonic-stop-double`。下一步优先保留这个通过直走子集的候选，独立解决站立到第一步和转向结束后到收步的接缝。完整 F2 通过前保留原验收门槛和未通过标记。
