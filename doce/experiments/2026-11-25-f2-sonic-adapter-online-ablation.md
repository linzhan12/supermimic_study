# F2.4 在线跟踪修正与世界模型排序消融

使用 [0.06 rad 接触相修正](2026-11-25-f2-sonic-contact-tracking-adapter.md) 在原联合库上做全流程回放。固定世界模型权重，分别比较冻结跟踪、修正仅用于执行、修正同时用于执行与预演、以及关闭世界模型排序。最后在新主体和初态上执行完整留出评测。绝对阈值、与 F1 的差值阈值、物理设置均沿用此前版本。

| 组别 | 稳态前向三档（m/s） | 左 / 右斜走实测横向（m/s） | 右斜走实测偏航（rad/s） | 两次松杆末半秒平面速度（m/s） | 旧数值失败数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 主初态，冻结跟踪 | 0.242 / 0.370 / 0.454 | −0.047 / −0.150 | +0.016 | 0.004 / 0.391 | 21 |
| 主初态，修正执行，按冻结跟踪排序 | 0.221 / 0.363 / 0.434 | +0.029 / +0.006 | −0.839 | 0.016 / 0.018 | 27 |
| 主初态，修正进入排序 | 0.221 / 0.363 / 0.434 | +0.059 / −0.078 | −0.380 | 0.016 / 0.013 | 26 |
| 主初态，修正执行，只做运动学排序 | 0.221 / 0.358 / 0.437 | +0.033 / −0.068 | −0.900 | 0.263 / 0.008 | 25 |
| 留出初态，修正进入排序 | 0.223 / 0.361 / 0.439 | −0.022 / −0.014 | −1.038 | 0.028 / 0.027 | 24 |

主初态适配执行的两组都解决末次松杆漂移。把修正纳入预演后，两侧斜走的横向符号均正确，右斜走误转减小。幅度仍明显不足，留出初态再次出现左斜走反号，故不能接受为转向及斜走通用方案。留出完整评测的两次转向偏航均值 +0.308 和 −0.470 rad/s，方向比冻结组正确，但 F1 相对误差仍超线。半增益对照仅跑两个主场景，左斜走侧向 +0.050 m/s 时前向 −0.026 m/s，右斜走前向 −0.067 m/s，观察未随增益单调变好。完整的五组都核对了 F1 站立起始、走中起始的完整参考轨迹逐字节一致。

同速正面和侧面比较使用四秒实际闭环，从第 150 至 349 行。适配预演组慢档为 0.221 对 0.221 m/s，左右肘分别 11.5° / 9.8° 对 11.9° / 10.6°；中档为 0.363 对 0.359 m/s；快档为 0.434 对 0.429 m/s，适配组右肘 27.5°、完整参考右肘 8.1°，快档完整摆步左右 1 / 4 对 3 / 2。中、快档同速完整参考的双手更靠近腰部，已额外提供手臂更下垂的中档对照。快档在同速对照可看到摆臂和跨步不对称。**F2 未通过，直走应优先保住独立低摆臂周期。**

## 复现

从 `supermimic/` 沿用 [实测方向接缝](2026-11-24-f2-sonic-measured-steering-seam.md) 开启组。去掉 `--cases`，添加 `--steady-forward True --extra-forward-speeds '(0.22,0.50)' --multispeed-profile True`。修正组增加 `--contact-adapter outputs/f2-sonic-adapter-06/contact_adapter.pt`，修正进入排序再加 `--adapt-ranking True`。纯运动学组使用 `--trackability-weight 0 --command-cost-weight 0 --rank-contact-launch False`，保留执行修正。半增益组加 `--adapter-gain 0.5`。留出组使用 `walk1_subject2.npz` 第 50/231 帧及对应独立 F1 目录。

输出为 `outputs/f2-sonic-wm-main-full-frozen` 与 `outputs/f2-sonic-adapter-live-{frozen-rank,adapt-rank,kinematic,holdout,half-main}`。用 `check_f2_gait.py` 生成各目录 `criteria.json`。留出 F1 同样由 `eval_f2.py` 使用原 `logs/SuperMimic-GenCoord-G1/2026-11-23_16-07-37`，记录第 50/231 帧的站立、走中起始，且不加载参考库或适配参数。正侧面 GIF 在 `outputs/f2-sonic-adapter-live-adapt-rank/paired-{slow,medium,medium-a,fast}.gif`，由 `audit_forward_gait.py` 生成。完整参考依次选 `outputs/f2-sonic-cycle-retimed-physical/{slow-a,medium-b,medium-a}/reference_0_motion.jsonl` 与 `outputs/f2-sonic-cycle-physical/medium-b/reference_0_motion.jsonl`，也取第 150 至 349 行。
