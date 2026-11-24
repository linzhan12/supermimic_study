# F2.0 初始权重离线闭环基线

以工程初始提交 `3ad431c` 上的跟踪器和本机 2026-11-17 权重为固定起点。新增本地 MuJoCo 批量回放，复用既有 P0 的同初态、三路诊断与步态测量流程。速度路线直接读取摇杆，完整参考路线使用原有运动编码器。分析日志记录姿态、地面接触、动作与参考，不加载 W&B。

## 实验设置

- G1 GenCoord，CPU，单环境，种子 `0`，物理步长 `0.005 s`，策略步长 `0.02 s`。使用保存时的默认网络布局加载旧 checkpoint。只有该文件的 actor 参与物理执行。
- `actor.pt` SHA-256 为 `3d2cdf557efe308d49c247feb1ca12be1237988902caae3d055422c5cd15bb43`。
- `walk1_subject1.npz` SHA-256 为 `a518c4667f1b973b2e4f32b519342ac2b59fd56faa574e14ed8222005d42c957`。站立、行走初态帧 `0 / 205`。同一场景不同接口初态及本体输入逐元素相同。

## 现象与测量

各固定档命令持续 6 秒，统计最后 4 秒。单位为 m/s、m/s、rad/s。有效摆动采用接触与足底几何判据。

| 前向指令 | 实测前向、侧向、偏航 | 有效摆动左 / 右 |
| ---: | --- | --- |
| 0.22 | 0.000 / 0.000 / −0.003 | 0 / 0 |
| 0.35 | 0.001 / 0.001 / −0.006 | 0 / 0 |
| 0.50 | 0.459 / 0.032 / +0.119 | 2 / 2 |

连续 `0.22 → 0.50 → 0.35 → 0` 最后半秒的前向均值为 `0.001 / 0.623 / 0.013 / 0.021 m/s`。慢速和中速有命令却近乎原地，切到快走出现超调。闭环脚本的全部 10 条路线完成了预定时长，完整参考两路均无跌倒终止。仿真输出了接触求解迭代数告警，所有路线统一保留初始版本的求解器配置。

对 `walk2_subject3` 另做固定参考诊断，各回放 150 步，实际接触判定中另对照了满足足底几何条件的单帧漏检修补。该组用于定位参考选择问题。

| 参考起始帧 | 参考前向 / 实测前向（m/s） | 实测侧向（m/s） | 观测支撑期膝角左 / 右（°） |
| ---: | --- | ---: | --- |
| 319 | 0.223 / 0.267 | −0.020 | 16.5 / 30.0 |
| 5781 | 0.350 / 0.408 | +0.036 | 47.4 / 59.6 |
| 578 | 0.490 / 0.544 | −0.023 | 19.4 / 25.5 |

慢走和快走参考的正面与侧面均成功离线渲染。本地输出在 `supermimic/outputs/f2-sonic-ref-walk2-s3/paired-fixed-references.gif`。这条视频用于观察不同参考的跟踪结果，正式自然度验收另做同实测速度配对。

## 复现

~~~bash
cd /home/hi/SuperMimic-Official/supermimic
export OMP_NUM_THREADS=2 WANDB_MODE=disabled
CKPT=logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41
.venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --record-posture True --steady-forward True --extra-forward-speeds '(0.22, 0.50)' \
  --multispeed-profile True --output-dir outputs/f2-sonic-baseline-walk1-s1
.venv/bin/python -u scripts/eval_f2.py --checkpoint-dir "$CKPT" \
  --motion-file ../data/lafan1_official/walk2_subject3.npz \
  --motion-only True --motion-frames '(319, 578, 5781)' --motion-window-s 3 \
  --record-posture True --output-dir outputs/f2-sonic-ref-walk2-s3
MUJOCO_GL=egl .venv/bin/python scripts/audit_forward_gait.py \
  --trace 'Slow reference' outputs/f2-sonic-ref-walk2-s3/reference_319_motion.jsonl 0 150 \
  --trace 'Fast reference' outputs/f2-sonic-ref-walk2-s3/reference_578_motion.jsonl 0 150 \
  --replay-gif outputs/f2-sonic-ref-walk2-s3/paired-fixed-references.gif
~~~

## 结论

基线直接速度编码器未达到 F2，低、中档甚至没带出稳定跨步。冻结的完整参考跟踪器可以从已在运动的参考初态执行三条不同速度，但参考均速与执行速度相差约 `0.04–0.06 m/s`，其中一个中速参考的支撑腿屈曲明显。F2.1 需要从不同来源找更合适的三档完整步态及起停转换，分别用运动学筛查和闭环回放确认。F2.0 只建立对照与测量协议。
