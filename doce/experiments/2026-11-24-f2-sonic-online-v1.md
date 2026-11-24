# F2.2 首次在线全身参考与世界模型预演

## 实现和假设

摇杆经限加速度滤波，以 10 Hz 检索全身动作片段。候选按目标速度、接缝处关节状态、共支撑脚相对位置和实测接触排序。参考接缝先沿支撑锚点做平面对齐，再用 160 ms 五次过渡保留接缝首帧位置与速度。关节速度含插值权重的导数项。规划结果以原有 640 维参考目标映射到冻结的 SuperMimic motion actor。规划操纵参考，执行动作仍由原网络生成。

结合点是从同一个 MuJoCo 实测状态出发，用 SuperMimic 的可微世界模型和实际执行的 motion actor 预演候选。每个候选给 10 步。记 $q_i^r,p_i^r$ 为候选，$\hat q_i,\hat p_i$ 为预演状态，$\Delta p$ 为从各自起点算出的位移。同时按下式检查关节误差、根位移/高度和预演重力倾斜。这个代价只参加候选排序。

$$
J_\mathrm{WM}
=\frac{1}{H}\sum_{i=1}^H
\left[
\frac{\|\hat q_i-q_i^r\|^2}{29(0.25\,\mathrm{rad})^2}
+\frac{0.5\|\Delta\hat p_{i,xy}-\Delta p^r_{i,xy}\|^2}{(0.10\,\mathrm m)^2}
+\frac{0.5(\hat p_{i,z}-p^r_{i,z})^2}{(0.08\,\mathrm m)^2}
+8\,e_{\mathrm{tilt},i}^2
\right].
$$

比较 `J_kin` 与 `J_kin + 0.8 J_WM`，并以严格/放宽支撑门槛隔离起步拒绝问题。设计参照 SONIC 的分层、原 SuperMimic 的闭环模型预演和已有的运动图检索。世界模型排序的实际控制收益由消融决定。 [SONIC](https://arxiv.org/html/2511.07820) · [SuperMimic](https://arxiv.org/pdf/2610.28480) · [Motion Graphs](https://graphics.cs.wisc.edu/Papers/2002/KGP02/)

## 现象与结论

严格版三档和连续换速均完成预定步数，但规划记录里的选中片段始终为空。起步时实际双脚接触，最相近起步片段的全关节角度 RMS 约 `0.284 rad`，超过当时 `0.26 rad` 的统一接缝门槛。该版一直给站姿参考，三个稳态档最后 4 s 的实测前向均为 `0.127 m/s`。这个爬行是未被控制的站姿漂移。

放宽接缝门槛后，各连续段末半秒结果如下。

| 相位 | 指令前向（m/s） | 规划前向（m/s） | 实测前向、侧向、偏航（m/s、m/s、rad/s） |
| --- | ---: | ---: | --- |
| 慢走 | 0.22 | 0.126 | 0.192 / +0.094 / −0.175 |
| 快走 | 0.50 | 0.098 | 0.592 / −0.058 / −0.073 |
| 中速 | 0.35 | 0.341 | 0.350 / −0.082 / +0.443 |
| 松杆 | 0 | 0 | 0.007 / −0.005 / +0.003 |

末半秒的中速偏航超限。快走末段 .098 m/s 的参考与 .592 m/s 实测也表明单看片段速度及末段实测无法证明稳定跟踪。松杆末半秒双脚持续着地。与纯运动学版配对的世界模型版实际做了 78 次多候选预演，选中片段、执行动作和实际速度逐步一致。此权重和本用例下尚未看到世界模型重排序收益。加入规划入口后的 `stand_start_motion.jsonl` 和原 F2.0 相同用例逐字节一致。目标布局、接缝边界及终端速度回归与既有测试共 23 项通过。

同实测速度正侧面对照保存在 `outputs/f2-sonic-plan-v1-relaxed/paired-slow.gif` 与 `paired-medium.gif`，慢档用摇杆 `[115, 200)` 对完整参考 `2425 / [25, 110)`，实测 `0.202 / 0.202 m/s`。中档用摇杆 `[384, 469)` 对完整参考 `5068 / [26, 111)`，实测 `0.365 / 0.366 m/s`。这些是局部诊断片段。**F2 未通过。** 下一轮先把起步接缝与走步接缝的约束分开验证，再检查周期连续性对快档和中速偏航的影响。

## 复现

先按 [F2.1](2026-11-24-f2-sonic-reference-screen.md) 生成参考回放。下面的运行目录均在 `supermimic/`。

~~~bash
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
C=outputs/f2-sonic-candidates
T=outputs/f2-sonic-transitions
.venv/bin/python scripts/pack_f2_bank.py --output outputs/f2-sonic-bank-v1 \
  --entry slow-a walk "$C/walk3_subject1/reference_243_motion.jsonl" 0 150 \
  --entry slow-b walk "$C/walk3_subject1/reference_397_motion.jsonl" 0 150 \
  --entry slow-c walk "$C/walk2_subject4/reference_2425_motion.jsonl" 0 150 \
  --entry medium-a walk "$C/walk2_subject4/reference_5068_motion.jsonl" 0 150 \
  --entry medium-b walk "$C/walk3_subject4/reference_8398_motion.jsonl" 0 150 \
  --entry fast-a walk "$C/walk2_subject4/reference_4179_motion.jsonl" 0 150 \
  --entry fast-b walk "$C/walk3_subject1/reference_1925_motion.jsonl" 0 150 \
  --entry start-a start "$T/walk1_subject5/reference_15_motion.jsonl" 80 150 \
  --entry start-b start "$T/walk1_subject1/reference_69_motion.jsonl" 80 150 \
  --entry stop-a stop "$T/walk4_subject1/reference_7118_motion.jsonl" 0 115 \
  --entry stop-b stop "$T/walk2_subject3/reference_1434_motion.jsonl" 0 135 \
  --entry stop-c stop "$T/walk3_subject4/reference_12169_motion.jsonl" 0 134

CKPT=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
for item in 'kin True 0.0' 'wm True 0.8' 'relaxed False 0.0' 'relaxed-wm False 0.8'; do
  read -r run strict weight <<< "$item"
  .venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
    --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
    --reference-bank outputs/f2-sonic-bank-v1 --planned-only True \
    --strict-contact "$strict" --trackability-weight "$weight" \
    --multispeed-profile True --cases '("multispeed_profile",)' \
    --output-dir "outputs/f2-sonic-plan-v1-$run" || break
done
# 严格版完整三档另加 --steady-forward True --extra-forward-speeds '(0.22, 0.50)'，
# 并去掉 --cases 或显式选择三档和 multispeed_profile。
~~~
