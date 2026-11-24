# F2.4 低摆臂快走参考的固定与在线闭环

在十二条本地 LAFAN1 G1 行走记录中，加权筛查均速、直线偏航、左右触地、支撑期膝角、躯干倾斜和肩肘。肩肘项同时考察参考的角度方差和下垂姿态，供候选排序。筛出的十六个 4 s 窗口逐个使用原 motion actor 闭环回放。快走同速样本来自 `walk2_subject4` 第 5173 和 5242 帧，原连续片段实测前向分别为 0.471 和 0.484 m/s，后者左右肘摆幅为 16.0° / 11.7°。

将低摆臂的同支撑窗口闭合成六条连续周期，按原周期闭合代码验证 7 s。三条候选各自替换原低摆臂库的快档，慢、中和起停仍用原数据。下表为末四秒实测。

| 参考 | 固定完整参考前向（m/s） | 规划快档实测前向（m/s） | 规划左右肘摆幅 | 规划左右完整摆步 | 三档与换速数值失败数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原低摆臂快档 | 0.477 | 0.465 | 16.9° / 26.9° | 2 / 2 | 0 |
| q5173 | 0.516 | 0.477 | 16.6° / 10.9° | 2 / 3 | 1 |
| q5242-b | 0.526 | 0.511 | 10.8° / 10.2° | 2 / 2 | 1 |
| q5439 | 0.515 | 0.503 | 9.3° / 16.2° | 2 / 3 | 2 |

q5242-b 快档的同速正侧面对照采用 0.511 与 0.526 m/s 的实测轨迹，参考的左右肘摆幅为 11.0° / 10.5°。快走时双手均在身体两侧，前后摆幅较旧快步缩小。在线快档保持两侧各两次完整跨步。连续快走切中速的段末仍有 +0.101 m/s 侧漂与 −0.324 rad/s 偏航，快档只完成局部验证。**F2 未通过。**

## 复现

从 `supermimic/` 运行。

~~~bash
.venv/bin/python scripts/scan_f2_references.py --motions ../data/lafan1_official/walk*.npz \
  --seconds 4 --count 3 --quiet-arms --forward-speeds 0.44 0.50
~~~

用 `eval_f2.py --motion-only True --motion-window-s 4 --record-posture True` 实测。四条用于闭合的原片段均出自 `walk2_subject4.npz`，帧为 4224、5173、5242、5439，物理轨迹在 `outputs/f2-sonic-quiet-physical-walk2_subject4`。以 `pack_f2_bank.py` 的 `0 200` 抽取原窗，以下索引交给 `repeat_f2_cycle.py`。

| 周期 | 起始 / 结束行 | 重采样倍率 |
| --- | ---: | ---: |
| q4224-a | 25 / 107 | 0.90 |
| q4224-b | 40 / 123 | 0.90 |
| q5173 | 26 / 98 | 1.10 |
| q5242-a | 107 / 183 | 1.00 |
| q5242-b | 76 / 145 | 1.02 |
| q5439 | 102 / 172 | 1.00 |

连续参考在 `outputs/f2-sonic-quiet-cycles`，各跑 7 秒，输出 `outputs/f2-sonic-quiet-cycle-<周期>`。再次打包时取 `0 350`，在 `outputs/f2-sonic-bank-low-arm-55` 副本中仅替换 `fast-a.npz`。沿用 [独立三档的四用例评测](2026-11-24-f2-sonic-reduced-arm-references.md)，结果为 `outputs/f2-sonic-quiet-planned-{q5173,q5242-b,q5439}`。双视角回放为 `outputs/f2-sonic-quiet-planned-q5242-b/paired-fast.gif`。
