# F2.4 转向与斜走的长时域适配

沿用 [有限接触适配](2026-11-25-f2-sonic-contact-tracking-adapter.md) 的八个通道和误差形式。只用横向、偏航指令与其后的松杆样本。时域增至 16 步、关节目标修正上限增至 0.10 rad，指令损失权重 2。采样时以实测状态启动当前参考的预测，即使线上后来重规划，也可用这个计划片段训练相同指令的响应。保留实测航向锚点。GPU 训练 500 次更新，用 423 个窗口，另一主体的 76 个窗口用于留出评估。留出预测指令损失从 1.071 降到 0.940。

| 同一线上路由 | 主初态左 / 右斜走实测侧向（m/s） | 留出左 / 右斜走实测侧向（m/s） | 两初态末次松杆速度（m/s） |
| --- | ---: | ---: | ---: |
| 0.06 rad 头，只用于方向与其后松杆 | +0.054 / −0.086 | +0.054 / −0.076 | 0.008 / 0.011 |
| 0.10 rad 长时域方向头 | −0.027 / −0.210 | −0.026 / −0.190 | 0.561 / 0.498 |

0.10 rad 组右斜走偏航分别 +0.017 / +0.030 rad/s，前向 +0.207 / +0.213 m/s。左斜走反号且最后停步恶化。右向表现与此前左向和转向造成的状态变化有关。固定参考仅在右斜走单段启用新头，主初态右向从 −0.150 到 −0.153 m/s，留出从 −0.016 到 −0.063 m/s，均未达到在线新头的右向水平。只在同物理初态的末次松杆启用新头，主初态末半秒 0.391 降至 0.060，留出 0.003 升至 0.013。两个单段试验在启用前的动作和姿态均完全一致。

保留较弱方向头作为联合对照。其两初态斜走符号一致，幅度仍小且右斜走偏航约 −0.28 rad/s。长时域头在线末次松杆时在找停步入点阶段继续方向起步，开始稳定前花费了过多时间。**F2 未通过。**

## 复现

沿用 [同初态短预测](2026-11-25-f2-sonic-world-model-audit.md) 的归档，再采集 `walk2_subject4.npz` 第 20/260 帧。开启原 0.06 rad 头、修正参与排序、斜走航向与分段实测接缝，诊断步距 3，输出 `outputs/f2-sonic-wm-audit-train-d`。从 `supermimic/` 运行。

~~~bash
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python -u scripts/train_f2_contact_adapter.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --train outputs/f2-sonic-wm-audit-{main,train-b,train-c,train-d}/stand_start_prediction.pt \
    outputs/f2-sonic-diag-lock-audit/stand_start_prediction.pt \
  --validation outputs/f2-sonic-wm-audit-holdout/stand_start_prediction.pt \
  --steering-only --include-replans --steps 500 --batch 32 --horizon 16 --limit-rad 0.10 \
  --command-weight 2 --output-dir outputs/f2-sonic-adapter-steering-10
~~~

线上对比沿用 [航向锚点组](2026-11-25-f2-sonic-command-scoped-tracking.md)，再加 `--adapter-steering-only True`，分别加载旧 0.06 和新 0.10 checkpoint。结果在 `outputs/f2-sonic-adapter-steering-{06,10}-{main,holdout}`。单段对照使用旧固定参考带并关闭航向锚点，按 [固定参考实验](2026-11-25-f2-sonic-contact-tracking-adapter.md) 加 `--adapter-probe-phases`，分别选择 `right_diagonal` 和 `release_after_diagonal`。
