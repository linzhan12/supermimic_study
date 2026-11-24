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

## 全量协议与起步接缝对照

位移头运行完整的三档、换速、起停、左右转、左右斜走协议，首个正常接缝选 0.48 s。第二组移除弱斜走周期，启用实测直走接续，延长上身混合至 0.55 s。四组 F1 站立和行走完整参考逐字节一致。

| 组别 | 起步前 1.5 s 前向 m/s | 左／右斜走横速 m/s | 末次松杆速度 m/s | 标准失败数 |
| --- | ---: | --- | ---: | ---: |
| 0.48 s，主 | 0.284 | +0.155／−0.192 | 0.025 | 15 |
| 0.48 s，留出 | 0.299 | +0.156／−0.190 | 0.007 | 12 |
| 0.55 s，主 | 0.305 | +0.164／−0.133 | 0.017 | 15 |
| 0.55 s，留出 | 0.300 | +0.160／−0.187 | 0.026 | 16 |

快速库中单独补回两个原慢启动片段，主／留出起步及直走数值与只保留短启动组相同。0.40 s 实测轴向接续主初态的完整中档摆步由 0／2 改为 1／2，0.55 s 为 2／2，但左脚离地中位数 0.007 m。留出用 0.55 s 仅 1／2，净空 0.001 m。两支启动对照及 0.48 s 组的连续换速慢档仍有失败。原始输出在 outputs/f2-sonic-quiet-{strong,mixed-start}-* 和 outputs/f2-sonic-root-full-{0.48,0.55}-{main,holdout}，完整校验用 check_f2_gait.py。

0.48 s 组各有离线 paired-left.gif 和 paired-right.gif。固定左斜走参考为 slow-left-20d 第 132–181 步，实测前向／横向 0.163／+0.152 m/s。固定右斜走参考为 slow-right-20 主／留出第 73–122 和 74–123 步，实测为 0.264／−0.194 和 0.268／−0.190 m/s。主初态左斜走肘摆幅 8.0°／13.9°，右斜走 6.6°／8.4°。四个闭环斜走窗口均缺少完整摆步。右斜走躯干倾角中位数约 14.2°，固定参考约 10.6°。轴向慢／中档和斜走实际迈步仍无法验收。

把左右斜走各延长到 3 s 的独立诊断显示，位移头第二秒实际前向／横向达到主 0.239／+0.187、留出 0.236／+0.193 m/s，第三秒横向回落到 +0.103／+0.072 m/s。三秒左侧完整摆步主为 1／1，留出 1／2。回落时两个初态都在弱、中、强参考间切换，留出在去掉弱候选之前的这一组中也存在回落。右斜走在主初态第一秒仍短暂反向，第三秒约 −0.195 m/s。延长诊断只把 eval_f2.STAND_START 的 left_diagonal 和 right_diagonal 持续时间改成 3.0，输出到 f2-sonic-diagonal-hold-{original,root}-{main,holdout}，原验收时程保持一秒。首秒不完整主要与启动和转向接入相关，持续步态也存在节律不稳。后续需隔离固定整周期承诺和当前高频重选，测量落脚和脚间交替。
