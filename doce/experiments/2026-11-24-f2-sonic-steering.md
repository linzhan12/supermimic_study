# F2.3 速度候选过滤及转向、斜走参考接入

## 对照设计

周期参考使用完整周期的估计实测速度。根据前向、横向、偏航尺度 `0.15 m/s、0.16 m/s、0.60 rad/s` 计算无量纲误差。在符合接缝约束的片段中，保留距最佳速度误差不超过 `0.25` 的片段，再由运动学或世界模型排序。该规则控制候选的目标相关性。先仅用周期库比较，然后加入两侧转向和斜走参考。基线 checkpoint、MuJoCo、初态、指令和验收口径统一。

## 现象

仅用周期库时，纯运动学与预演组的三档实测前向分别为 `0.230 / 0.364 / 0.476 m/s` 和 `0.228 / 0.358 / 0.476 m/s`。纯运动学组的中速换速末半秒从上一轮 `0.321 / −0.105 / +0.196` 变为 `0.336 / −0.037 / +0.012`，但慢、快段末侧向仍分别 `+0.125 / +0.133 m/s`。预演组的该轮快档末半秒侧向 `+0.148 m/s`，中档偏航 `+0.264 rad/s`，各有不足。筛掉速度偏离的候选对一部分接缝有帮助。

固定参考对转向和斜移有正反方向能力。下表列出加入库的片段及 2 s 的实测平均速度。量纲为 m/s、m/s、rad/s。

| 片段与起始帧 | 实测前向 / 横向 / 偏航 |
| --- | --- |
| walk2_subject1 / 3934 | 0.391 / −0.070 / +0.383 |
| walk2_subject4 / 4810 | 0.361 / −0.046 / +0.367 |
| walk1_subject1 / 2326 | 0.342 / +0.018 / −0.426 |
| walk3_subject3 / 674 | 0.274 / −0.003 / −0.436 |
| walk2_subject4 / 8997 | 0.256 / +0.185 / +0.106 |
| walk1_subject2 / 3915 | 0.197 / −0.234 / +0.069 |
| walk3_subject2 / 9253 | 0.345 / −0.213 / +0.047 |

在线短时切换比固定片段难。加入完整方向库后，运动学组的左、右转阶段平均偏航 `+0.223 / −0.329 rad/s`，能区分两侧；从松杆切入斜走，两侧都没有成功走出侧移。预演组的两侧平均偏航 `+0.223 / −0.217 rad/s`，末次松杆段继续执行转向参考，整段平均前向 `0.325 m/s`，属于明确的停步失败。主序列最初 1.5 s 前进段均速仅 `0.056 m/s`。把起步片段从原回放第 103 行开始截取的对照，没有选到起步候选，继续漂移直到序列结束。截点从 80 改为 90 或 95 时，前进段均速分别变为 0.065 或 0.084 m/s，仍低于 0.20 m/s 的旧阈值。截点 100 和 103 都没有可用起步接缝。

**F2 未通过。** 保留方向库的可跟踪性证据。下一轮必须解决短时起步接缝以及停步无可行候选时继续行走的问题，然后重做换速、斜走和同速正侧面对照。按旧脚本完整评测，F1 两条完整参考回归无差异，方向库预演组仍有 20 项失败，包含起步、转向、斜走、稳态脚部和换速。本轮未改变验收阈值。

## 复现

周期库见 [周期库在线对照](2026-11-24-f2-sonic-cycle-plan.md)。固定参考用相同本地权重，文件为表中对应的 LAFAN1 G1 参考，命令使用 `--motion-only True --motion-window-s 2 --motion-frames '(起始帧,)' --record-posture True`，同源多个帧可列在同一个元组。物理重放留在 `outputs/f2-sonic-steering/<录制>`。生成方向库及主用例的命令如下，均从 `supermimic/` 运行。

~~~bash
S=outputs/f2-sonic-steering
.venv/bin/python scripts/pack_f2_bank.py --output outputs/f2-sonic-bank-steering \
  --entry turn-left-a walk "$S/walk2_subject1/reference_3934_motion.jsonl" 0 100 \
  --entry turn-left-b walk "$S/walk2_subject4/reference_4810_motion.jsonl" 0 100 \
  --entry turn-right-a walk "$S/walk1_subject1/reference_2326_motion.jsonl" 0 100 \
  --entry turn-right-b walk "$S/walk3_subject3/reference_674_motion.jsonl" 0 100 \
  --entry diagonal-left walk "$S/walk2_subject4/reference_8997_motion.jsonl" 0 100 \
  --entry diagonal-right-a walk "$S/walk1_subject2/reference_3915_motion.jsonl" 0 100 \
  --entry diagonal-right-b walk "$S/walk3_subject2/reference_9253_motion.jsonl" 0 100
cp -- outputs/f2-sonic-bank-cycles/*.npz outputs/f2-sonic-bank-steering/

export OMP_NUM_THREADS=2 WANDB_MODE=disabled
.venv/bin/python -u scripts/eval_f2.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --record-posture True --steady-forward True --extra-forward-speeds '(0.22, 0.50)' \
  --multispeed-profile True --reference-bank outputs/f2-sonic-bank-steering \
  --trackability-weight 2 --prediction-steps 20 --pace-margin 0.25 \
  --output-dir outputs/f2-sonic-v5-steering-wm
.venv/bin/python scripts/check_f2_gait.py \
  outputs/f2-sonic-current-f1 outputs/f2-sonic-v5-steering-wm \
  --output-json outputs/f2-sonic-v5-steering-wm/criteria.json
~~~

只测周期组时将动作库换为 `outputs/f2-sonic-bank-cycles`。运动学组设置权重 0，纯规划回放可加 `--planned-only True`。起步截短对照只改变方向库里的两条起步片段，分别使用 `[90, 150)`、`[95, 150)`、`[100, 150)`、`[103, 150)`。截短组只运行 `stand_start`。
