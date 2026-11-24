# F2.3 从实测静止重新检索方向起步

在 12 段原动作上检索 1.5 s 的方向过渡。以起步窗口前段的平面速度、双支撑和末段体轴速度排序，停步反向使用。共实测 23 个候选，按冻结跟踪器的末 0.4 s 平均速度给起步候选定方向。示例是左斜走 `walk3_subject3 / 5851` 的末段 `[0.208, +0.160, +0.021]`，右斜走 `walk2_subject3 / 2153` 为 `[0.183, −0.163, +0.002]`。前两项 m/s，末项 rad/s。

保持 `command-cost-weight=3`、三入点、较短起步库和 0.54 s 起步搜索。仅扩充参考库时，主场景轨迹不变，转向后停住再收到斜走时仍保持停止参考。允许按实测双脚接触且平面速度低于 0.08 m/s、偏航低于 0.15 rad/s 重新检索起步后，左斜走选到 `start-left-c`，原先没有候选，但该段实际横向均值仍是 `−0.049 m/s`。这段开环素材从其原始首姿态执行时末 0.4 s 可达到 `+0.152 m/s`，截进组合动作后出现反向，说明末段分类不足以保证接缝后的跟踪效果。

允许方向改变在接缝承诺结束后打断起步，右斜走不再继续上一条左斜走起步，实际横向均值仍 `+0.062 m/s`，且偏航 `+0.338 rad/s`。最后松杆末半秒可止住（前向 `−0.017 m/s`）。未通过 F2。下一步需要直接从当前支撑姿态校验转入后的实际速度，或改变参考形状让冻结策略可执行。完整测试套件在实测重启改动后 226 项通过，另有指令反向中断用例。

## 复现

从 `supermimic/` 运行，输入保持原 checkpoint。扫描命令如下。固定回放每个动作 1.5 s、75 帧，使用 `eval_f2.py --motion-only True --motion-window-s 1.5 --record-posture True --motion-frames '(起点,)'`。

~~~bash
.venv/bin/python scripts/scan_f2_references.py --motions ../data/lafan1_official/walk*.npz \
  --seconds 1.5 --count 3 --directional-transitions
~~~

扩充库使用 `pack_f2_bank.py` 并复制原短起步库全部片段。额外候选均取固定轨迹的 `[0,75)`。

| 名称 | 类型 | 动作及起点 |
| --- | --- | --- |
| start-left-a / b / c | start | walk3_subject3 / 5745、5851、132 |
| start-right-a / b / c / d | start | walk2_subject3 / 2153，walk3_subject3 / 8092，walk1_subject2 / 993，walk3_subject4 / 1142 |
| start-turn-left / right | start | walk3_subject4 / 64，walk2_subject3 / 342 |
| stop-diagonal-right / left | stop | walk3_subject5 / 11590，walk3_subject3 / 5795 |

沿用 [起步搜索](2026-11-24-f2-sonic-start-lookahead.md) 的 0.54 s 预演命令，替换 `--reference-bank outputs/f2-sonic-bank-directional-start`，加 `--restart-from-rest True`，输出 `outputs/f2-sonic-directional-interrupt`。关闭该开关的对照保存在 `outputs/f2-sonic-directional-all-ref`，同一开关但起步不随方向中断的诊断保存在 `outputs/f2-sonic-directional-restart`。
