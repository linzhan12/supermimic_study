# F2.2 双支撑起步接缝对照

## 改动与对照

沿用首版 12 条参考、50 Hz 物理闭环和冻结权重。唯一变更是起步接缝在规划前后均有双脚支撑时，将全关节角度 RMS 上限从 `0.26` 调到 `0.32 rad`。此前最接近的起步约 `0.284 rad`。持续行走、走到停、支撑脚位移及接触条件沿用严格门槛。这个对照检验起步开放后，三档持续行走能否自己建立稳定周期。

## 现象与测量

| 指令前向（m/s） | 规划前向（m/s） | 实测前向 / 侧向 / 偏航 | 有效摆动左 / 右 | 肩俯仰幅度左 / 右（°） |
| ---: | ---: | --- | --- | --- |
| 0.22 | 0.104 | 0.179 / −0.017 / −0.182 | 2 / 2 | 9.9 / 7.1 |
| 0.35 | 0.289 | 0.369 / +0.021 / −0.011 | 2 / 2 | 17.9 / 9.3 |
| 0.50 | 0.143 | 0.525 / +0.040 / −0.147 | 10 / 7 | 15.3 / 6.6 |

前向、侧向单位 m/s，偏航单位 rad/s。三个固定命令各持续 6 s，取末 4 s，接触保持原始计数。严格版可以选择起步动作并进入三档，首版此前只保持零速参考。慢档和快档仍超出稳态偏航门槛。快档 4 s 内出现 10 / 7 个有效摆动事件，相比固定参考 3 s 内的 2 / 2 明显偏多，支撑膝角也增至 `31.7° / 41.0°`，且一侧摆臂不足。虽然快档实测均速接近指令，规划平均只给到 `0.143 m/s`，不满足自然的持续步态。连续换速末半秒和首版放宽对照相同，中速仍向左快速偏航。保持世界模型权重 `0.8` 后，换速统计也相同。

## 复现

先复用 [首次在线实验](2026-11-24-f2-sonic-online-v1.md) 的动作库。运行时选择严格支撑。

~~~bash
cd /home/hi/SuperMimic-Official/supermimic
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
CKPT=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
.venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
  --motion-file ../data/lafan1_official/walk1_subject1.npz --record-posture True \
  --reference-bank outputs/f2-sonic-bank-v1 --planned-only True \
  --steady-forward True --extra-forward-speeds '(0.22, 0.50)' --multispeed-profile True \
  --cases '("steady_forward", "steady_forward_0p220", "steady_forward_0p500", "multispeed_profile")' \
  --output-dir outputs/f2-sonic-plan-v2-start-kin
~~~

## 结论

双支撑起步限制解除后，速度响应恢复，**F2 仍未通过**。原始参考短片只有约 3 s，在线需要更长时间持续执行。下一次固定跟踪器，从有同侧落脚且接缝姿态相近的窗口建立多周期参考，先在无摇杆的完整跟踪中核对持续速度、朝向和摆动，再用同一个在线规划器重复筛查。
