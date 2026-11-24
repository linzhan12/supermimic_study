# F2 参考骨盆位移输入的跟踪适配

## 对照设计

原 SuperMimic motion actor 的 640 维目标包括十帧关节角、角速度和相对根朝向。根位移进入世界模型和规划代价，motion actor 的目标输入没有这个分量。这里保持 F1 的接口与权重，给 F2 有界适配头增加四个特征。前两个为实测骨盆到当前参考骨盆的平面偏移，后两个为参考骨盆从当前到未来 0.16 s 的平面位移。二者用实测航向变换到局部坐标，以 0.08 m 归一化。头部仍是单个 48 单元隐藏层，输出八个髋踝修正，最大 0.08 rad。原无位移检查点可直接加载。

两组使用相同种子、采样、冻结权重、损失和训练预算。在本机 GPU 做 450 次更新，每次 32 个窗，预测时域 16 步。三个初态共 228 个训练窗，未参与拟合的另一主体有 76 个评估窗。留出预测的指令损失从 0.4913 降为 0.3608（速度输入）和 0.3584（增加位移）。预测支撑足滑移项分别为 0.0505 和 0.0479。

## MuJoCo 闭环

| 适配头 | 主左斜走前向／横向 | 留出左斜走前向／横向 | 主右斜走前向／横向 | 留出右斜走前向／横向 | 两初态末次松杆 m/s |
| --- | --- | --- | --- | --- | --- |
| 原 0.06 rad | 0.134／+0.133 | 0.116／+0.161 | 0.246／−0.196 | 0.277／−0.174 | 0.009／0.007 |
| 同新数据，速度 | 0.147／+0.173 | 0.147／+0.173 | 0.227／−0.015 | 0.226／−0.016 | 0.012／0.012 |
| 同新数据，速度和位移 | 0.171／+0.155 | 0.169／+0.156 | 0.263／−0.192 | 0.262／−0.190 | 0.025／0.007 |

两组新头均参与候选预演和接触跟踪。右侧指令前的状态会受左侧选择影响，所以另用固定参考带，只在左斜走应用各个头。每个初态的三个实验前 325 步逐帧相同。主初态原／速度／位移的左斜走前向为 0.150／0.174／0.166 m/s，留出为 0.131／0.156／0.148 m/s。两组新头小幅改善固定动作下的前向，加入位移的优势出现在联合左右变向中。单凭留出世界模型损失无法推断这种差异。

## 复现

按[肩肘连续性联合库](2026-11-25-f2-sonic-quiet-release.md)运行，使用 --prediction-audit-stride 4、--planned-only True、--cases '("stand_start","walk_start")'。数据目录为 outputs/f2-sonic-quiet-diagonal-release-audit-{main,holdout,walk2-subject4,walk3-subject3}。额外主体初始帧各为 20／260、20／250。训练从 supermimic 目录执行。

~~~bash
OMP_NUM_THREADS=2 WANDB_MODE=disabled .venv/bin/python scripts/train_f2_contact_adapter.py \
  --checkpoint-dir logs/SuperMimic-GenCoord-G1/2026-11-17_17-55-41 \
  --motion-file ../data/lafan1_official/walk1_subject1.npz \
  --train outputs/f2-sonic-quiet-diagonal-release-audit-{main,walk2-subject4,walk3-subject3}/stand_start_prediction.pt \
  --validation outputs/f2-sonic-quiet-diagonal-release-audit-holdout/stand_start_prediction.pt \
  --steering-only --include-replans --steps 450 --batch 32 --horizon 16 --limit-rad 0.08 \
  --command-weight 2 --preview-displacement --output-dir outputs/f2-sonic-ingress-adapter-root
~~~

速度输入组省略 --preview-displacement。闭环遵循同一联合库方案并替换 --contact-adapter。固定带用原 0.06 rad 头按 --prediction-audit-stride 1 提取，保存在 outputs/f2-sonic-adapter-tape-{main,holdout}。回放加 --fixed-plan-dir 和 --adapter-probe-phases '("left_diagonal",)'，关闭 --adapt-ranking。输出前缀为 f2-sonic-adapter-fixed。245 项自动化测试通过。**F2 未通过。**
