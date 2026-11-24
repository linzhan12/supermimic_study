# F2 补充慢走动作的物理回放筛查

从 HOLD 筛选实际速度覆盖 `0.22 / 0.35 / 0.50 m/s` 的个体步态，补充原训练示范库。使用 RAI Institute 与 Boston Dynamics 发布的 29 关节 Unitree G1 数据，来源固定为 `f80b04d384d4ec8922c88deada878c88597a3cf9`，许可为 CC BY 4.0。 [HOLD 数据库](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9/data/g1/locomotion/walk)

下载六条非镜像轨迹。以文件时间戳对根位置与关节角重采样至 50 Hz，四元数用球面插值。按四秒窗口筛选前向均值、横向及偏航变化和髋关节交替。`S9_slow_walk_0` 和 `s11_slow_walk_002` 原始前向速度中位数分别为 `0.020` 和 `0.003 m/s`，符合直走限制的四秒窗口里没有达到最慢档 `0.22 m/s` 的持续前移。余下四条录制选八个片段，在冻结的完整参考策略中做闭环回放。侧向为 G1 航向坐标的四秒均值。

| 录制 | 起始帧 | 原始前向（m/s） | 物理前向（m/s） | 物理侧向（m/s） |
| --- | ---: | ---: | ---: | ---: |
| `S0_slow_walk_0` | 2535 | 0.222 | 0.349 | −0.166 |
| `S0_slow_walk_0` | 2646 | 0.483 | 0.613 | −0.253 |
| `S1_slow_walk_0` | 4591 | 0.218 | 0.319 | −0.134 |
| `S1_slow_walk_0` | 5021 | 0.351 | 0.461 | −0.168 |
| `S4_slow_walk_0` | 4176 | 0.219 | 0.335 | −0.128 |
| `S4_slow_walk_0` | 4291 | 0.495 | 0.701 | −0.284 |
| `u4_slow_walk` | 8 | 0.220 | 0.466 | −0.165 |
| `u4_slow_walk` | 5626 | 0.501 | 0.684 | −0.293 |

八个物理回放均完成。航向跟上时仍出现持续向右漂移，`S0`、`S4` 的快档样本在目标帧处的 29 关节 RMS 误差均为 `14.3°`，且实测前向超出计划用的速度档。当前完整参考策略在这批外部重定向数据上的闭环输出不适合作为直走教师，本轮未把它并入训练库。判断限于所查片段及当前参考策略。 [同期 F2 训练](2026-11-24-f2-full-body-mimic.md)

```bash
cd /home/hi/SuperMimic-Official/supermimic
mkdir -p outputs/f2-hold-source
REV=f80b04d384d4ec8922c88deada878c88597a3cf9
for name in S0_slow_walk_0 S1_slow_walk_0 S4_slow_walk_0 S9_slow_walk_0 s11_slow_walk_002 u4_slow_walk; do
  curl -fL "https://media.githubusercontent.com/media/rai-opensource/hold/$REV/data/g1/locomotion/walk/$name.h5" \
    -o "outputs/f2-hold-source/$name.h5" || break
done
# 转换脚本需要 Python 环境可导入 h5py。本次从临时目录加载 h5py 3.15.1。
.venv/bin/python scripts/convert_hold_g1_motion.py outputs/f2-hold-converted outputs/f2-hold-source/*.h5
for item in 'S0_slow_walk_0 2535 2646 s0' 'S1_slow_walk_0 4591 5021 s1' \
            'S4_slow_walk_0 4176 4291 s4' 'u4_slow_walk 8 5626 u4'; do
  read -r clip first second label <<< "$item"
  OMP_NUM_THREADS=2 .venv/bin/python -u scripts/eval_p0.py \
    --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
    --motion-file "outputs/f2-hold-converted/$clip.npz" --stand-frame "$first" --walk-frame "$second" \
    --motion-only True --motion-window-s 4 --record-posture True --output-dir "outputs/f2-hold-physical-$label"
done
```
