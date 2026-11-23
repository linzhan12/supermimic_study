# F2 步态与训练流程复查

2026-11-23。用户在本地回放指出，F2 候选前进动作丢失了 SuperMimic 完整参考中的自然跨步。撤回此前对 F2 整体通过的判定。该候选仅通过速度、停步和摆臂筛查，整套步态未达标。

## 训练流程核对

| 环节 | 核对结果 |
| --- | --- |
| 数据和重定向 | 对两条下游训练片段 `walk2_subject3`、`walk3_subject3` 以及本轮的两条留出片段 `walk2_subject4`、`walk4_subject1`，逐数组核对了本机已有的 50 Hz G1 重定向文件与仓库转换后文件。29 个关节的顺序、角度和角速度、骨盆位姿及线角速度均完全一致。原始重定向本身采用的源数据和预训练覆盖仍需从当时的运行记录追溯。 |
| 预训练和完整参考 | 共同起点 `2026-11-17_17-55-41` 只有 actor、critic 和世界模型权重。当前复现用仓库默认配置重建能严格载入的 actor。完整参考接口输入跨帧关节目标和相对朝向，在留出片段能够做出明显跨步。相同两条完整参考轨迹在原始权重及 F1/F2 中逐字节一致。 |
| F1 命令和观察 | 独立命令只有前向、侧向、偏航三个目标量，保持 1–2 秒。运动片段在下游仅用于初态复位。在 F1 配置中，动作 encoder 与 decoder 冻结，latent 模仿损失置零，速度 encoder 只接收命令，decoder 另接收一帧 `rep_state`。这是本机试跑的真值运动状态观察，与实验方案的本体观测历史存在差异。 |
| F1 目标与更新 | 真实物理轨迹持续拟合世界模型，actor 通过 32 步想象轨迹优化跟速和生存。既有预检验证了步前命令、动作标签及同状态评分一致性。2,000 次外层迭代采集 256,000 条转移，只有速度 encoder 的 18 个 actor 张量变化。跟速目标不含参考动作的跨步时序、落脚位置或腿部关节轨迹。 |
| F2 姿态对照 | `0.25` 的早期对照用支撑与足部 site 平移约束走动，站立奖励促进停步。走动项没有按速度和支撑相位提供参考步态。延长到 6,000 次迭代仍未通过速度约束。 |
| F2 候选继承及更新 | `2026-11-23_17-22-01` 保留 F1 的移动 encoder 和共享 decoder，移动姿态权重为零。新增肩部控制从髋差驱动肩俯仰，并调整肩外展。128 个并行环境运行 200 次外层迭代，采集 25,600 条转移。actor 的训练更新只发生在独立停步 encoder，候选未训练新的移动步态。 |
| 回放及验收 | 原验收检查了速度、停步站姿、跌倒、完整参考一致性和摆臂。完整参考路径一致只说明原接口得到保留。本轮额外核对了速度分支自身的关节摆幅、腾空时长、脚底高度、触地时双足前后间距与支撑时足部 site 速度。 |

## 配对回放现象

第一组从 `walk2_subject4` 同一行走状态、同一初态帧 `3719` 启动。完整参考使用关节与朝向目标，三组速度接口都收到该参考片段对应的相同速度序列。分析前 119 个采样。另用原 `walk1_subject1` 固定 `0.35 m/s` 的最后 4 秒衡量持续直走；数值按左 / 右排序。

| 运行与输入 | 平均实测前向（m/s） | 落脚前后间距中位数（cm） | 髋俯仰 P95−P05（°） | 踝俯仰 P95−P05（°） |
| --- | ---: | ---: | ---: | ---: |
| 原权重，完整参考 | 0.439 | 46.1 / 47.8 | 51.2 / 48.6 | 29.3 / 29.3 |
| 原权重，参考速度 | 0.091 | — / 7.8 | 16.4 / 13.2 | 11.0 / 6.9 |
| F1，参考速度 | 0.337 | −0.1 / 10.4 | 18.7 / 10.5 | 10.3 / 9.6 |
| 旧 F2，参考速度 | 0.370 | 4.0 / 13.2 | 18.1 / 13.2 | 10.5 / 9.1 |
| F1，持续摇杆 | 0.321 | 1.5 / 9.5 | 9.4 / 7.3 | 4.4 / 5.1 |
| 旧 F2，持续摇杆 | 0.403 | 1.8 / 14.7 | 14.7 / 12.8 | 6.6 / 4.8 |

旧 F2 持续摇杆回放的左右摆动脚离地峰值中位数只有 `3.1 / 7.3 mm`，左脚发生 3 次完整摆动，右脚 7 次。膝关节摆幅 `12.8° / 17.2°`。在连续接触窗口内部，左足 site 平移 P90 从 F1 的 `0.047 m/s` 升到 `0.201 m/s`，仍包含滚足和足部转动。并排播放能看到完整参考明显交替送腿，F1/F2 则以小步快速挪动并保持屈膝。

其余两个物理参考窗口用于检验片段差异。`walk2_subject4` 帧 `5599–5702` 的平均前向速度 `0.499 m/s`，落脚间距 `48.3 / 34.3 cm`，脚底离地峰值 `16.7 / 14.7 mm`。`walk4_subject1` 帧 `3559–3624` 的平均前向速度 `0.394 m/s`，落脚间距 `23.0 / 22.6 cm`，离地峰值 `22.4 / 14.7 mm`，另伴有 `0.141 m/s` 左移。三个参考窗口各只有两个符合筛查条件的完整摆动，因此这些实测值适合作为诊断及配对回放证据，覆盖更多纯直走周期后才能确定验收范围。

落脚间距是触地帧两足 site 的前后投影差，落脚脚位于另一只脚后方时记为负。脚底高度取 G1 脚碰撞胶囊的最低点。完整摆动要求至少三个连续离地采样，触地后至少三个连续接触采样。支撑足速度排除了接触边界两侧各两步，用作滑移代理。统计以实际物理回放为准，参考窗口的起止帧可复现。

## 结论

原始 checkpoint 在同一初态切换到速度输入后已经失去正常跨步。F1 恢复了前进速度，小步、左右不对称和髋膝踝幅度不足依然存在。旧 F2 保持同一个移动 encoder，实际改善落在停步和摆臂。当前证据支持复查速度分支如何从本体状态恢复步态相位，并在移动目标中加入同实测速度、同支撑阶段的动作先验。现有结果尚不能判定冻结 decoder 是否构成瓶颈，也没有证据支持仅延长旧训练即可恢复模仿步态。

后续 F2 对照需要保留原始速度和停步门槛、完整参考一致性，并在训练片段以外检查完整周期、双侧落脚间距、腾空、支撑运动和肩髋协调。定量筛查应与本地配对视频同时通过。当前未推进至 F3 或地形实验。

## 本地复现

```bash
cd /home/hi/SuperMimic-Official/supermimic
export OMP_NUM_THREADS=2
for pair in '2026-11-17_17-55-41 f2-natural-dense-reference' \
            '2026-11-23_16-07-37 f2-gait-audit-f1' \
            '2026-11-23_17-22-01 f2-gait-audit-f2'; do
  read -r checkpoint output <<< "$pair"
  .venv/bin/python scripts/eval_p0.py \
    --checkpoint-dir "logs/SuperMimic-GenCoord-G1/$checkpoint" \
    --motion-file ../data/lafan1_official/walk2_subject4.npz \
    --stand-frame 3719 --walk-frame 5599 --record-posture True \
    --output-dir "outputs/$output"
done
.venv/bin/python scripts/eval_p0.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk4_subject1.npz \
  --stand-frame 3559 --walk-frame 3350 --record-posture True \
  --output-dir outputs/f2-gait-audit-reference-slower
.venv/bin/python scripts/audit_forward_gait.py \
  --trace dense outputs/f2-natural-dense-reference/stand_start_motion.jsonl 0 119 \
  --trace base-vel outputs/f2-natural-dense-reference/stand_start_reference_velocity.jsonl 0 119 \
  --trace f1-vel outputs/f2-gait-audit-f1/stand_start_reference_velocity.jsonl 0 119 \
  --trace f2-vel outputs/f2-gait-audit-f2/stand_start_reference_velocity.jsonl 0 119 \
  --trace f1-steady outputs/f1-arm-check/steady_forward_scripted_velocity.jsonl 150 350 \
  --trace f2-steady outputs/f2-phase-anchor-finetune-s0/steady_forward_scripted_velocity.jsonl 150 350 \
  --trace dense-2 outputs/f2-natural-dense-reference/walk_start_motion.jsonl 0 104 \
  --trace dense-3 outputs/f2-gait-audit-reference-slower/stand_start_motion.jsonl 0 66
```

本地回放数据与 `forward-gait-comparison.gif` 在 `supermimic/outputs/f2-natural-dense-reference/`。旧 F1/F2 稳态轨迹可按上轮记录中的 `--steady-forward True` 复现。所有回放使用本地权重和动捕文件。
