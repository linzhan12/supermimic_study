# F2.3 以真实脚位锚定无候选停步

从方向库预演组的斜走结束后松杆失败切入。旧组接不到收步片段，最后两秒继续转向。新组仅在零指令、无相容停止片段、且实测双脚接触时，补一段参考。用 MuJoCo 的独立状态按双足位置和姿态投影过渡帧。上身随五次插值接近默认站姿，骨盆逐渐移到脚间。下半身用带阻尼的关节空间最小二乘求解 $ \Delta q=(J^\top J+\lambda^2I)^{-1}J^\top e$，每帧足位置残差须小于 4 mm。实测双脚速度作为可选进入判据。机器人继续由冻结的 motion actor 执行。

| 组别 | 松杆后开始收步 | 末半秒平面速度均值（m/s） | 两秒净位移（m） | 末半秒站姿关节 RMS（rad） |
| --- | ---: | ---: | ---: | ---: |
| 旧组 | 未开始 | 0.357 | 0.530 | 0.472 |
| 0.40 s，无脚速检查 | 0.30 s | 0.183 | 0.581 | 0.186 |
| 0.40 s，脚速上限 0.25 m/s | 0.70 s | 0.139 | 0.450 | 0.178 |
| 0.40 s，脚速上限 0.15 m/s | 0.70 s | 0.139 | 0.450 | 0.178 |
| 0.65 s，脚速上限 0.28 m/s | 0.70 s | 0.279 | 0.574 | 0.176 |
| 0.22 s，脚速上限 0.25 m/s | 0.70 s | 0.179 | 0.539 | 0.173 |
| 0.28 s，脚速上限 0.25 m/s | 0.70 s | 0.169 | 0.518 | 0.174 |

无脚速检查时在脚仍移动的落地过程中进入，净位移反而增大。有脚速检查的 0.40 s 组结束了旧转向参考，降低了尾速和关节误差，实际步态仍有滑移，未达到 0.10 m/s 门槛。已有录制停步能正常接入的行走起点用例保持原动作选择。后续应选取能被现有 actor 动力学稳定跟踪的末端姿态，以及更早可衔接的制动相位。

## 复现

从 `supermimic/` 目录用方向库运行。

~~~bash
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
for item in '040 0.40 none' '025 0.40 0.25' '015 0.40 0.15' '028-long 0.65 0.28' '22 0.22 0.25' '28 0.28 0.25'; do
  read -r name duration speed <<< "$item"
  limit=()
  if [ "$speed" != none ]; then limit=(--contact-stop-max-foot-speed "$speed"); fi
  .venv/bin/python -u scripts/eval_f2.py \
    --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
    --motion-file ../data/lafan1_official/walk1_subject1.npz \
    --record-posture True --cases '("stand_start",)' --planned-only True \
    --reference-bank outputs/f2-sonic-bank-steering \
    --trackability-weight 2 --prediction-steps 20 --pace-margin 0.25 \
    --contact-stop-duration "$duration" "${limit[@]}" \
    --output-dir "outputs/f2-sonic-contact-stop-$name"
done
~~~
