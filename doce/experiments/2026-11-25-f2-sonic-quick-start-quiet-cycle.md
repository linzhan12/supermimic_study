# F2.4 低摆臂快步与快速起步的联合接续

将通过独立评测的 q5242-b 低摆臂快档与仅换速后的航向保持接入联合库。关闭直走的适配，转向、斜走及其后的松杆使用 0.06 rad 跟踪头。主初态保留原慢起步时，独立三档、两脚接触及换速条件仍全部满足，但首个 1.5 s 前向仅 0.055 m/s，转向方向也未过，完整 F2 有 16 项失败。相同配置留出初态为 21 项失败。两组均完成，F1 两条完整参考轨迹逐字节一致。

| 起步和入周期设置 | 主初态前向前 1.5 s（m/s） | 主初态完整筛查失败数 | 留出前向前 1.5 s（m/s） | 留出完整筛查失败数 |
| --- | ---: | ---: | ---: | ---: |
| 原低摆臂起步 | 0.055 | 16 | 0.113 | 21 |
| 0.48 s 分段的短起步 | 0.284 | 18 | — | — |
| 短起步，从实测支撑接进周期 | 0.287 | 19 | 0.302 | 18 |

实测起步接续把主初态中档完整摆步由 0 / 2 改到 1 / 2，离地净空仍不足，未达旧规则。前向发生较大节律起伏，F1 相对速度误差条件仍失败。短起步提高了响应速度，仍需训练或重选起步与周期的支撑衔接。

额外测试两侧双支撑直接合成前进一步，各取两种时长及骨盆倾斜供世界模型排序。主初态未接受任何合成窗，仍使用录制起步，前向均速约 0.059 m/s。留出初态合成候选被选中，实测前向均速仅 0.016 m/s。单次合成步不保留为默认发步方法。241 项单元测试通过。**F2 未通过。**

## 复现

从 `supermimic/`，在 [命令分流评测](2026-11-25-f2-sonic-command-scoped-tracking.md) 的原起步与 [短起步库](2026-11-25-f2-sonic-split-steering-seams.md) 的副本中，分别替换快档为 `outputs/f2-sonic-quiet-cycle-bank/q5242-b.npz`。启用 `--adapter-steering-only True --adapt-ranking True --axial-heading-lock True --axial-heading-on-change True --diagonal-heading-lock True --contact-stop-capture True`，运行完整评测，主、留出和短起步分别保存在 `outputs/f2-sonic-quiet-integrated-{main,holdout,quick}`。短起步分段仍用 0.48 s 和搜索 0.54 s。实测接续组另加 `--start-measured-handoff True`，保存在 `outputs/f2-sonic-quiet-measured-exit-{main,holdout}`。

合成步诊断使用原起步库，启用 `--contact-launch-axial True --start-measured-handoff True`，时长 0.44 或 0.60 s，骨盆提前量 0.00 或 0.08 m，仅测 `stand_start` 和 `walk_start` 规划路线，输出 `outputs/f2-sonic-axial-launch-*`。
