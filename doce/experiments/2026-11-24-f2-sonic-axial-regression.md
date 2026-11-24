# F2.3 独立直走候选与同速全身回放

在扩充方向库中，直走只保留此前筛出的 `slow-a`、`medium-a`、`fast-a` 三个周期。按方向筛选，斜走和转向仍可使用已录制的方向参考。两组统一冻结 checkpoint、世界模型权重和指令，分别关闭和打开实测接触起步与收步。

| 组别 | 稳态前向慢 / 中 / 快（m/s） | 快档换速末半秒横向（m/s） | 完整旧条件失败数 |
| --- | ---: | ---: | ---: |
| 扩充库，未隔离直走 | 0.121 / 0.347 / 0.494 | −0.103 | 29 |
| 隔离直走，方向与航向引导 | 0.227 / 0.325 / 0.466 | +0.128 | 20 |
| 隔离直走，加入实测接触起停 | 0.227 / 0.325 / 0.466 | +0.128 | 18 |
| 原单周期独立回归 | 0.242 / 0.358 / 0.475 | 见回归记录 | 直走及换速筛查无失败 |

隔离后去掉了向前指令过渡到转向动作的路径，但慢、中档仍出现足趾净空不足，快档换速仍侧漂。加接触桥后，首次松杆末半秒平均平面速度从 0.037 降至 0.003 m/s；第二次从 0.011 增至 0.054 m/s。左斜走的横向仍反号，前向约 −0.023 m/s。两组 F1 的站立起始和行走起始完整参考轨迹分别与基线逐字节相同。

## 同实际速度回放

三个比较都取物理闭环轨迹第 150 至 349 行。左右面板分别为原单周期库的规划跟踪和相近实测速率的完整参考跟踪。回放按时间起点对齐，左右的跨步相位可不同。

| 档位 | 实际前向规划 / 完整参考（m/s） | 规划左 / 右肘摆幅 | 完整参考左 / 右肘摆幅 |
| --- | ---: | ---: | ---: |
| 慢 | 0.242 / 0.221 | 13.7° / 11.3° | 11.9° / 10.6° |
| 中 | 0.358 / 0.347 | 12.8° / 7.5° | 10.6° / 6.7° |
| 快 | 0.475 / 0.477 | 22.8° / 35.0° | 21.2° / 34.9° |

摆幅取第 95 与第 5 百分位之差。在正侧面能看到慢档脚步和躯干有周期性侧摆、快档肘部活动更大而且左右不同。快档和参考的肘摆幅相近，说明现有完整参考本身也有较大摆臂。以它作为同速对照可以诊断跟踪偏差，验收还需要符合用户要求的手臂活动幅度。**F2 未通过。**

## 复现

从 `supermimic/` 运行。复制 `outputs/f2-sonic-bank-directional-start` 到 `outputs/f2-sonic-bank-isolated-axial` 时排除 `fast-b.npz`、`fast-native.npz` 和 `slow-native.npz`。沿用 [实测静止方向重启](2026-11-24-f2-sonic-directional-restart.md) 的命令，启用 `--directional-selection True --heading-strength 0.58`，去掉 `--planned-only` 与 `--cases`，加 `--steady-forward True --extra-forward-speeds '(0.22,0.50)' --multispeed-profile True`。输出 `outputs/f2-sonic-axial-guide`。接触对照加 `--contact-launch-duration 0.48 --contact-stop-duration 0.40 --contact-stop-max-foot-speed 0.25`，保存至 `outputs/f2-sonic-axial-launch`。两个目录都用 `check_f2_gait.py` 和 `outputs/f2-sonic-current-f1` 对照并保存 `criteria.json`。

离线回放保存在 `outputs/f2-sonic-regression-single-pace/paired-{slow,medium,fast}.gif`，同名 JSON 中有完整动作统计。中档示例，慢、快档对应替换为 `steady_forward_0p220`、`steady_forward_0p500` 和 `slow-a`、`fast-a`。

~~~bash
R=outputs/f2-sonic-regression-single-pace
MUJOCO_GL=egl .venv/bin/python scripts/audit_forward_gait.py \
  --trace 'F2 planner' "$R/steady_forward_planned.jsonl" 150 350 \
  --trace 'Full reference' outputs/f2-sonic-cycle-retimed-physical/medium-a/reference_0_motion.jsonl 150 350 \
  --replay-gif "$R/paired-medium.gif" > "$R/paired-medium.json"
~~~
