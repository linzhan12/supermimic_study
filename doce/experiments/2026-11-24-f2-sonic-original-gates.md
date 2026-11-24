# F2.3 恢复旧验收和真实步态周期侧移读数

## 方法

复用旧实验分支的数值验收策略，候选路线可选 `planned`，比较基线仍用 F1 的摇杆策略。完整参考轨迹逐字节核对。旧阈值和旧计分窗口沿用已保存的实现。另对实际接触事件找到两个同侧着地及其间的对侧着地，以段首航向坐标计算周期侧移。

$$
\bar v_{y,\mathrm{cycle}} =
 \frac{e_{y,\mathrm{segment}}^\top(p(t_2)-p(t_1))}{t_2-t_1}.
$$

确保原半秒机体系误差和完整周期固定航向漂移都可核查，原始接触与可选的几何修补也可对照。

## 现象与结论

对主场景从同一 `walk1_subject1 0 / 205` 初态分别重放 F1 和 [持续推进的纯运动学组](2026-11-24-f2-sonic-continuation.md)。两条完整参考轨迹逐字节相同。站立启动完整参考也与 F2.0 初始版本相同，SHA-256 为 `9c8f7bd577207b78b6f1cbf2d7a0bfedd256523316793a805e9670c322c318b9`。

数值筛查如期以状态码 1 返回，输出 `outputs/f2-sonic-gate-kin/criteria.json`。本组的参考库只包含直走和起停，明显失败项包括转向和斜走的速度、站立起步阶段的速度、慢档及快档的落脚距离/离地峰值，以及连续换速段末误差。固定中档实际速度接近目标且完整参考回归通过，仍不足以将本组判为 F2 通过。

## 复现

~~~bash
cd /home/hi/SuperMimic-Official/supermimic
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
for entry in 'current-f1 2026-11-23_16-07-37' 'gate-kin 2026-11-17_17-55-41'; do
  read -r name stamp <<< "$entry"
  args=()
  if [ "$name" = gate-kin ]; then
    args+=(--reference-bank outputs/f2-sonic-bank-cycles)
  fi
  .venv/bin/python -u scripts/eval_f2.py \
    --checkpoint-dir "logs/SuperMimic-GenCoord-G1/$stamp" \
    --motion-file ../data/lafan1_official/walk1_subject1.npz \
    --record-posture True --steady-forward True --extra-forward-speeds '(0.22, 0.50)' \
    --multispeed-profile True --output-dir "outputs/f2-sonic-$name" "${args[@]}"
done
.venv/bin/python scripts/check_f2_gait.py \
  outputs/f2-sonic-current-f1 outputs/f2-sonic-gate-kin \
  --output-json outputs/f2-sonic-gate-kin/criteria.json
~~~
