# F2.2 周期参考的固定跟踪验证

## 对照

从已检查的原始参考，按同侧落脚到下一次同侧落脚提取一个周期。入口和出口要求相同的支撑组合，检查全关节角度和根旋转误差，并保留动作原有的腿部、骨盆、肩肘协调。每周期末段对关节和根位置作 Hermite 修正，令下一周期首尾位姿及平移/关节速度连续。设 $u$ 为接缝窗口的归一化时间，$L$ 为窗口长度，位置差、速度差为 $\Delta q,\Delta\dot q$，末段补偿为

$$
\delta q(u)
=(3u^2-2u^3)\Delta q + L(u^3-u^2)\Delta\dot q.
$$

两端的值和导数分别对应原周期内部和下一周期入口。根朝向采用整周期的平滑闭合，保留周期内的方向变化。各周期根位置沿原净步长推进。再分别对五条参考做小幅重采样。重采样调整节奏，同一周期的动作选型和跨度保持一致。所有回放使用同一冻结的 motion actor。

## 现象

每条运行 7 s，取最后 4 s。全部十条闭环回放完成。

| 周期 | 原周期实测前向（m/s） | 重采样倍率 | 重采样实测前向 / 侧向 / 偏航 | 原周期 → 重采样有效摆动左 / 右 | 重采样支撑期膝角左 / 右（°） |
| --- | ---: | ---: | --- | --- | --- |
| slow-a | 0.182 | 1.08 | 0.221 / +0.023 / −0.001 | 4 / 3 → 2 / 3 | 27.7 / 33.6 |
| medium-a | 0.409 | 0.90 | 0.347 / +0.025 / −0.019 | 2 / 3 → 2 / 3 | 23.0 / 31.8 |
| medium-b | 0.429 | 0.90 | 0.359 / −0.079 / +0.016 | 3 / 2 → 3 / 2 | 9.8 / 22.2 |
| fast-a | 0.547 | 0.92 | 0.477 / +0.054 / −0.000 | 2 / 2 → 1 / 2 | 19.9 / 26.3 |
| fast-b | 0.543 | 0.92 | 0.509 / −0.028 / −0.000 | 1 / 1 → 3 / 2 | 28.3 / 31.7 |

前两轴单位 m/s，偏航为 rad/s。`medium-b` 的连续周期版本仍向侧向漂移，重采样后实际关节 RMS 跟踪误差 `0.327 rad`，明显高于其余周期 `0.139–0.155 rad`。首版在线周期库排除该片段。其余样本分别提供慢、中、快三档的闭环连续参考。快走 `fast-a` 的时序有一只脚在四秒窗口内只计入一次完整摆动，保留这一失败项并与另一个快走参考对照。固定周期下的速度和接触结果支持进入在线筛查。**该阶段只验证固定参考可跟踪性。**

## 复现

先按 [首次在线实验](2026-11-24-f2-sonic-online-v1.md) 生成 `outputs/f2-sonic-bank-v1`。各周期的索引为对应入库片段内的帧索引，终点含在原始接缝估计中。

~~~bash
cd /home/hi/SuperMimic-Official/supermimic
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
CKPT=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
for item in 'slow-a 48 138 1.08' 'medium-a 54 137 0.90' 'medium-b 8 93 0.90' 'fast-a 16 105 0.92' 'fast-b 61 133 0.92'; do
  read -r name first last scale <<< "$item"
  for mode in native retimed; do
    speed=1.0
    output=outputs/f2-sonic-cycle-physical
    if [ "$mode" = retimed ]; then
      speed="$scale"
      output=outputs/f2-sonic-cycle-retimed-physical
    fi
    cycle="outputs/f2-sonic-$mode-cycles/$name.npz"
    .venv/bin/python scripts/repeat_f2_cycle.py "outputs/f2-sonic-bank-v1/$name.npz" "$cycle" \
      --start "$first" --end "$last" --speed-scale "$speed"
    .venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" --motion-file "$cycle" \
      --motion-only True --motion-frames '(0,)' --motion-window-s 7 \
      --record-posture True --output-dir "$output/$name"
  done
done
~~~
