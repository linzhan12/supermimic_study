# F2.4 冻结原策略后的有限接触相跟踪修正

沿用 [同初态短预测](2026-11-25-f2-sonic-world-model-audit.md) 的 actor、世界模型和原全身目标。训练一个 48 隐单元的修正头，输入当前体轴速度与指令差、重力方向、参考的髋踝位置误差和角速度、实测与计划的左右脚接触。输出只加到左右髋俯仰、髋横滚、踝俯仰、踝横滚共八个位置目标。物理增量为 $\Delta q=0.06\tanh(f_\theta(x))$ rad，原始动作空间按各通道的尺度回换。执行前向和世界模型预演均从实际接触开始。预演后续接触取该全身参考的接触序列。

使用冻结动力学反传 8 步。损失含平均体轴指令误差、全身关节和骨盆高度误差、支撑足点速度超 0.20 m/s 后的惩罚、动作修正量与帧间变化。训练中不改原模型参数。三个初态来源共有 885 个计划连续的窗口，留出主体和初态有 120 个窗口，留出数据仅用于评估。RTX 5090 D v2 上跑 600 次 AdamW 更新，批量 32、学习率 0.001、种子 0。留出窗口的归一化指令损失从 0.938 降到 0.841，支撑足点惩罚从 1.207 降到 0.989。这两项是世界模型指标。

## 同参考物理对照

每步录下冻结组实际传给跟踪器的全身参考窗口。固定窗口回放时由当前实际根朝向重新构造原 motion goal。无修正回放与原在线对照的执行动作、足接触、关节姿态、速度逐帧完全一致。只松杆或只左斜走的实验，在启用修正前的执行动作、物理速度和姿态也完全一致。以下为 MuJoCo 实测。

| 同参考组 | 左斜走前向 / 横向（m/s） | 右斜走前向 / 横向（m/s） | 两次松杆末半秒平均平面速度（m/s） |
| --- | ---: | ---: | ---: |
| 主初态冻结 | −0.011 / −0.047 | +0.068 / −0.150 | 0.004 / 0.391 |
| 主初态全程修正 | 0.000 / −0.025 | +0.087 / −0.167 | 0.011 / 0.012 |
| 主初态只在左斜走修正 | +0.017 / −0.039 | +0.043 / −0.174 | 0.004 / 0.317 |
| 主初态只在末次松杆修正 | −0.011 / −0.047 | +0.068 / −0.150 | 0.004 / 0.005 |
| 留出初态冻结 | +0.222 / +0.075 | +0.299 / −0.016 | 0.331 / 0.003 |
| 留出初态全程修正 | +0.248 / +0.072 | +0.287 / −0.054 | 0.410 / 0.018 |

末次松杆的改善通过相同初态和全身参考隔离。左斜走只修正该段时前向转正，横向仍反号。留出首次松杆在参考仍执行旧走步的情况下速度增加，需修复松杆无可行入点时的计划。所有固定回放完成；原 F1 站立起始和行走起始轨迹逐字节一致，235 项单元测试通过。**F2 未通过。**

## 复现

从 `supermimic/` 运行。按 [短预测实验](2026-11-25-f2-sonic-world-model-audit.md) 收集主序列和留出初态。训练额外收集 `walk3_subject1.npz` 第 0/205 帧、以及 `walk2_subject3.npz` 第 64/245 帧，命令使用 `--prediction-audit-stride 3 --planned-only True`。前一组额外打开三档稳态与换速，分别存为 `outputs/f2-sonic-wm-audit-train-b` 和 `outputs/f2-sonic-wm-audit-train-c`。

~~~bash
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python -u scripts/train_f2_contact_adapter.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --train outputs/f2-sonic-wm-audit-{main,train-b,train-c}/*_prediction.pt \
  --validation outputs/f2-sonic-wm-audit-holdout/*_prediction.pt \
  --device cuda:0 --steps 600 --batch 32 --horizon 8 --limit-rad 0.06 \
  --output-dir outputs/f2-sonic-adapter-06
~~~

固定窗口从同样初态用 `--prediction-audit-stride 1 --planned-only True` 录到 `outputs/f2-sonic-fixed-tape-{main,holdout}`。物理回放改用 `--fixed-plan-dir` 指向相应目录，关闭诊断，按组选择 `--contact-adapter outputs/f2-sonic-adapter-06/contact_adapter.pt`。单段消融加 `--adapter-probe-phases "('left_diagonal',)"` 或 `"('release_after_diagonal',)"`，本机输出为 `outputs/f2-sonic-fixed-adapter-{main,holdout,left,release}`。
