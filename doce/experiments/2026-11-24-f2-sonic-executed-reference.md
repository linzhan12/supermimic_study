# F2.3 将已有闭环动作作为过渡参考的筛查

从先前已验证的 F1 闭环轨迹提取关节和骨盆的实际执行状态，以相邻帧估计线速度、关节速度、四元数相对旋转速度。转换保留时间戳、关节顺序和来源哈希。用冻结的 motion actor 重放这些可直接跟踪的参考。原 F1 控制器的动作未输入新控制器。取两套与主场景不同的源动作，检验参考接口能否继承较快的闭环起步。

| 动作 | 源 | 记录素材前向 / 侧向 / 偏航 | motion actor 实测前向 / 侧向 / 偏航 |
| --- | --- | --- | --- |
| 起步 | walk2_subject1 | 0.287 / +0.027 / +0.015 | 0.023 / −0.018 / +0.031 |
| 起步 | walk3_subject4 | 0.302 / +0.001 / −0.052 | 0.048 / −0.016 / −0.034 |
| 左斜走 | walk2_subject1 | 0.224 / +0.145 / −0.005 | 0.042 / −0.016 / +0.013 |
| 左斜走 | walk3_subject4 | 0.213 / +0.157 / −0.040 | 0.044 / −0.028 / −0.001 |
| 右转 | walk3_subject4 | 0.318 / −0.018 / −0.280 | 0.119 / −0.042 / −0.268 |

线速度为 m/s，偏航为 rad/s。输出 `outputs/f2-sonic-executed-references` 中 14 段仿真均执行完。关节参考能弱化部分偏航误差，但可执行的实际速度没有随来源动作保留。说明 F1 的闭环启动不能直接作为冻结 motion actor 的有效起步参考。本轮未将这些录制动作加入在线库。

## 复现

从 `supermimic/` 运行。使用保存的 `outputs/f2-fresh-walk2-subject1-f1` 和 `outputs/f2-holdout-walk3-f1` 中的 `stand_start_scripted_velocity.jsonl`。提取脚本的截取范围以原轨迹行号计，左闭右开。以下每个源使用同一表。

~~~text
start           47  125  1.4
left-turn      120  175  1.0
right-turn     170  225  1.0
turn-stop      220  325  1.8
left-diagonal  322  375  1.0
right-diagonal 372  425  1.0
diag-stop      421  525  1.8
~~~

~~~bash
.venv/bin/python scripts/extract_f2_executed_reference.py "$TRACE" "$NPZ" "$FIRST" "$LAST"
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python -u scripts/eval_f2.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file "$NPZ" --motion-frames '(0,)' --motion-only True \
  --motion-window-s "$SECONDS_REPLAY" --record-posture True --output-dir "$OUT"
~~~
