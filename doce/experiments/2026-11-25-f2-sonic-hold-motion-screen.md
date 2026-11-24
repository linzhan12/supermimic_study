# F2 筛选 HOLD 的 G1 全身慢走与侧步

公开数据来自 RAI Institute 与 Boston Dynamics 的 HOLD，使用 CC BY 4.0。取其发布的 G1 重定向文件，Git 版本为 f80b04d384d4ec8922c88deada878c88597a3cf9。仅取走路、侧步和起步样本。数据留在本机实验输出目录。[HOLD 数据和许可](https://github.com/rai-opensource/hold)

## 导入与物理筛查

保留原采样时间，按关节名映射到本工程的 29 轴顺序。四元数按元数据重排并以球面插值转为 50 Hz。在首批验证中，个别轨迹源角速度与世界系旋转差分的第 95 百分位差达 0.50 rad/s；转到世界系后抽查误差约 0.016 rad/s。最终导入从重采样四元数做世界坐标旋转差分，平移和关节速度由源线性插值。用两秒、四秒滚动窗口挑选直走、斜走和起停候选。物理验证使用原冻结 motion actor，四秒轨迹取第 25–199 行统计。以下是代表样本。

| HOLD 样本与帧 | 腰俯仰修正 rad | 实测前向／横向 m/s | 完整摆步 左／右 | 躯干倾角中位数 |
| --- | ---: | --- | --- | ---: |
| S4_slow_walk_1 1631 | 0 | 0.349／−0.033 | 2／2 | 15.8° |
| S4_slow_walk_1 1631 | −0.20 | 0.349／+0.005 | 4／2 | 10.5° |
| S4_slow_walk_1 1631 | −0.30 | 0.355／+0.027 | 3／1 | 8.4° |
| S4_slow_walk_1 2760 | 0 | 0.411／−0.068 | 3／3 | 20.6° |
| S0_slow_walk_0 5559 | 0 | 0.260／−0.099 | 1／1 | 18.9° |
| normal_sidestep_mirrored 229 | 0 | 0.276／−0.207 | 4／4 | 28.2° |
| normal_sidestep_mirrored 580 | 0 | 0.270／−0.180 | 4／4 | 29.1° |
| normal_sidestep_mirrored 580 | −0.30 | 0.408／−0.153 | 6／6 | 24.7° |
| normal_sidestep 580 | 0 | 0.128／+0.010 | 3／3 | 25.6° |
| normal_sidestep 580 | −0.30 | 0.140／+0.049 | 4／3 | 17.9° |

腰俯仰对照只在参考关节角减去固定量，使用 MuJoCo 关节范围裁剪，裁剪帧置该轴速度为零。其他通道一致。右侧参考新增了速度和摆步覆盖，但躯干倾斜大。左侧同轴镜像执行后横向仍很小，数值对称的参考在同一冻结策略下产生不同运动。S4 1631 的 −0.20 rad 版本是中档候选，左右肘摆幅约 16.6°／13.6°，进入周期闭合对照。当前未完成联合验收。

## 复现

从 supermimic 目录运行，需要 h5py。文件来自 data/g1/locomotion 下的 sidestep、walk、shuffle 目录。九个源文件是 slow_sidestep 与 normal_sidestep 及其镜像、S0_slow_walk_0、S1_slow_walk_0、S4_slow_walk_1、s11_slow_walk_000、shuffle_start，扩展名为 h5。

~~~bash
.venv/bin/python scripts/import_f2_hold_motion.py \
  --output outputs/f2-external-hold/motions-world outputs/f2-external-hold/*.h5
.venv/bin/python scripts/scan_f2_references.py --motions outputs/f2-external-hold/motions-world/*.npz \
  --seconds 4 --count 2 --quiet-arms --directional-transitions
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python scripts/eval_f2.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file outputs/f2-external-hold/motions-world/S4_slow_walk_1.npz \
  --motion-only True --motion-frames '(2760,1631,379,2654)' --motion-window-s 4 \
  --record-posture True --output-dir outputs/f2-external-hold/physical-world-S4_slow_walk_1
~~~

关节偏置组的转换文件与物理轨迹分别归档在 motions-waist-*、physical-waist-*，位于 outputs/f2-external-hold。原世界系物理组为 physical-world-*。扫描负责候选提名，是否纳入 F2 取决于实际动作和全量验收。

## 中档闭合对照

将 −0.20 rad 腰俯仰的 1631 固定回放用 pack_f2_bank.py 抽取 0–199 行，取第 49 与第 116 行的同支撑窗口闭合。原始接缝关节 RMS 差 0.076 rad。用 repeat_f2_cycle.py 分别按 1.00、0.93、0.85 倍重采样，生成 7 s 参考。物理验证使用 --motion-frames '(0,)' --motion-window-s 6.98，末四秒统计。

| 倍率 | 实测前向／横向 m/s | 完整摆步 左／右 | 躯干倾角中位数 |
| --- | ---: | ---: | ---: |
| 1.00 | 0.504／+0.091 | 5／2 | 10.0° |
| 0.93 | 0.526／+0.073 | 2／2 | 11.9° |
| 0.85 | 0.537／+0.067 | 4／3 | 13.3° |

闭合和拉长周期后仍比中档 0.35 m/s 快，侧漂也增大。因此固定片段的中档表现没有复现在周期闭环，保留原独立三档参考。周期和执行归档在 outputs/f2-external-hold 的 repeated-medium 与 physical-repeat-medium-*。
