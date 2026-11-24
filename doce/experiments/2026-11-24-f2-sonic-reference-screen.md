# F2.1 完整原参考筛查

## 对照设置

跟踪器、世界模型配置、仿真、种子和日志协议与 F2.0 一致。这里直接回放原始完整参考的连续截取，从该参考帧初始化机器人。先用运动学数据对 `data/lafan1_official/walk*.npz` 中 12 条录制的 3 秒窗口排序，再固定候选帧做 50 Hz 的闭环验证。排序使用固定初始航向下的净位移、根偏航变化、过低速度的比例和足底几何，起停另看首尾速度和双支撑比例。运动学标签仅供筛查，以下接触结果来自 MuJoCo。

## 现象及测量

18 条稳态及站立候选和 9 条起停候选各回放 150 步，均完成预定时长。以下列出速度覆盖较好的样本。表中摆动采用足底接近地面且前后均接触的单帧修补后重算，原字段仍在日志。短窗的摆动计数受起止相位影响。

| 用途 | 原始录制与起始帧 | 参考 / 实测前向（m/s） | 实测侧向（m/s） | 左 / 右有效摆动 | 左 / 右支撑期膝角（°） |
| --- | --- | ---: | ---: | --- | --- |
| 慢走 | `walk3_subject1 / 243` | 0.220 / 0.214 | +0.013 | 2 / 2 | 27.6 / 32.1 |
| 慢走 | `walk2_subject4 / 2425` | 0.213 / 0.206 | +0.005 | 1 / 1 | 32.3 / 36.2 |
| 中速 | `walk2_subject4 / 5068` | 0.335 / 0.365 | +0.005 | 1 / 3 | 21.9 / 32.3 |
| 中速 | `walk3_subject4 / 8398` | 0.364 / 0.381 | +0.004 | 2 / 3 | 9.7 / 23.9 |
| 快走 | `walk2_subject4 / 4179` | 0.502 / 0.530 | +0.014 | 2 / 2 | 20.1 / 29.2 |
| 快走 | `walk3_subject1 / 1925` | 0.493 / 0.549 | +0.006 | 2 / 2 | 28.9 / 29.3 |

参考前向 `0.341 m/s` 的 `walk3_subject1 / 4441` 实际仅 `0.279 m/s`，而 `walk1_subject1 / 3936` 实测 `0.579 m/s`，伴随 `−0.075 m/s` 侧向速度。两者测出的差异说明固定跟踪器会放大或缩小指定速度，同一名义速度不足以直接入库。两条接近 `0.21 m/s` 参考的离线正侧面对照为 `supermimic/outputs/f2-sonic-candidates/paired-slow-fixed.gif`。

`walk1_subject5 / 15`、`walk1_subject1 / 69` 的起步窗口前半段基本保持站立，最后半秒实测前向分别 `0.346`、`0.475 m/s`，后者早段曾倒移。首版入库截去前 80 帧，衔接后的响应要重新验证。`walk4_subject1 / 7118` 的前、后半秒实测前向 `0.347`、`0.008 m/s`，末半秒为全程双支撑。`walk2_subject3 / 1434` 与 `walk3_subject4 / 12169` 也完成收步，末半秒分别 `−0.025`、`−0.034 m/s`。另一个停步片段 `walk2_subject3 / 11556` 中间再次加速到 `0.364 m/s`，首版排除。

## 复现

~~~bash
cd /home/hi/SuperMimic-Official/supermimic
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
CKPT=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
C=outputs/f2-sonic-candidates
T=outputs/f2-sonic-transitions
mkdir -p "$C" "$T"
.venv/bin/python scripts/scan_f2_references.py \
  --motions ../data/lafan1_official/walk*.npz --seconds 3 --count 2 > "$C/windows-3s-transitions.json"

while IFS='|' read -r dst motion frames; do
  .venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
    --motion-file "../data/lafan1_official/$motion.npz" \
    --motion-only True --motion-frames "$frames" --motion-window-s 3 \
    --record-posture True --output-dir "outputs/$dst/$motion" || break
done <<'CASES'
f2-sonic-candidates|walk3_subject1|(243, 397, 4441, 1925)
f2-sonic-candidates|walk3_subject3|(596, 1215, 2146)
f2-sonic-candidates|walk3_subject4|(8398, 8210, 6230)
f2-sonic-candidates|walk1_subject1|(9374, 9955, 3936, 0)
f2-sonic-candidates|walk2_subject4|(2425, 2928, 5068, 4179)
f2-sonic-transitions|walk1_subject5|(15,)
f2-sonic-transitions|walk1_subject1|(69,)
f2-sonic-transitions|walk3_subject3|(56, 5157)
f2-sonic-transitions|walk2_subject3|(1434, 11556)
f2-sonic-transitions|walk4_subject1|(7118, 6673)
f2-sonic-transitions|walk3_subject4|(12169,)
CASES

MUJOCO_GL=egl .venv/bin/python scripts/audit_forward_gait.py \
  --trace 'Slow full ref A' "$C/walk3_subject1/reference_243_motion.jsonl" 0 150 \
  --trace 'Slow full ref B' "$C/walk2_subject4/reference_2425_motion.jsonl" 0 150 \
  --replay-gif "$C/paired-slow-fixed.gif"
~~~

## 结论

初始跟踪器能在完整参考下分别执行三档速度、站立起步和行走收步，可以开始测规划接口。F2.1 的结果是分离的短窗，上述候选只用于开发，无法据此判断连续切换、任意松杆和自然度达标。首版规划同时记录参考速度、固定回放估计的速度偏差和闭环实际速度。`subject2` 的原始录制虽在运动学候选列表中出现，未参与这轮物理筛查或加入动作库，留待独立闭环验证。
