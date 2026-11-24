# F2.4 MOSAIC 公开 G1 参考筛查

选择八段公开的 MOSAIC GENMO 步行 G1 文件。数据卡记录了从 GMR 重定向到 BeyondMimic 导出的流程。原文件为 50 Hz，包含 29 列关节和 30 列刚体，未附关节名。按 Isaac Lab 的关节名称顺序重排到本地 MuJoCo，根取刚体 0 的位置、四元数和速度。每文件抽八帧，用正运动学和全局刚体位置匹配验证顺序。位置 RMS 为 0.229 至 0.294 微米。转换和回放均离线进行，原数据和转换产物保存在本机实验目录。[MOSAIC 数据卡及许可](https://huggingface.co/datasets/chuhonglin/MOSAIC_Dataset) · [BeyondMimic 导出器](https://github.com/HybridRobotics/whole_body_tracking/blob/main/scripts/csv_to_npz.py) · [SONIC G1 关节顺序](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/envs/env_utils/joint_utils.py)

从运动学筛出的 14 个短窗逐个用冻结 motion actor 闭环运行 1.5 s。起步及方向候选示例如下。前两轴单位 m/s，偏航 rad/s。

| 参考 | 起点 | 整段实测前向 / 横向 / 偏航 | 末 0.40 s 前向 / 横向 / 偏航 |
| --- | ---: | ---: | ---: |
| walk_0040 | 0 | +0.208 / −0.059 / +0.018 | +0.386 / −0.210 / −0.098 |
| walk_0001 | 232 | +0.142 / −0.022 / +0.294 | +0.431 / −0.175 / +0.195 |
| walk_0140 | 59 | +0.153 / +0.001 / +0.028 | +0.445 / −0.081 / +0.056 |
| walk_0020 | 176 | +0.337 / −0.050 / −0.094 | +0.162 / −0.111 / −0.047 |
| walk_0040 镜像 | 0 | +0.247 / −0.023 / −0.018 | +0.576 / +0.092 / +0.106 |

保留前四条到联合库，第二条截到第 68 行之前以保留转向而避免出口侧飘。两条初态的相同在线场景都未选用外部片段，实测动作与原方向候选组相同。确定素材存在可跟踪的右侧起步，但同初态下的足位和姿态接缝仍是瓶颈。镜像未提供足够的左向初段。**F2 未通过。**

## 复现

下载 MOSAIC 仓库 `G1/generated_genmo/g1_genmo_walk_*.npz` 中编号 `0001,0020,0040,0060,0080,0100,0120,0140` 八个文件到 `supermimic/outputs/f2-external-mosaic/raw`。从 `supermimic/` 运行下列转换与筛查。

~~~bash
.venv/bin/python scripts/import_f2_mosaic_motion.py --output outputs/f2-external-mosaic/converted \
  outputs/f2-external-mosaic/raw/*.npz
.venv/bin/python scripts/scan_f2_references.py --motions outputs/f2-external-mosaic/converted/*.npz \
  --seconds 1.5 --count 2 --directional-transitions --quiet-arms
~~~

固定回放用 `eval_f2.py --motion-only True --motion-window-s 1.5 --record-posture True`，帧分别用 `0040 (0,32,163,185,217,225,242)`、`0140 (50,59,84)`、`0001 (175,226,232)`、`0020 (176,)`，路径为 `outputs/f2-external-mosaic/physical-<编号>`。按 `pack_f2_bank.py` 前四行的回放分别打包 `0 75`、`0 68`、`0 75`、`0 75`，前三条为起步，第四条为走步。拷入原 `outputs/f2-sonic-bank-axial-quiet` 内容得到 `outputs/f2-sonic-bank-with-mosaic`，沿用 [航向锚点组](2026-11-25-f2-sonic-command-scoped-tracking.md) 的主、留出初态适配回放，输出 `outputs/f2-sonic-mosaic-online-{main,holdout}`。
