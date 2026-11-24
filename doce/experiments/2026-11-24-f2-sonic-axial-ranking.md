# F2.3 在联合库中分别排序直走周期与方向过渡

在隔离直走库中，只把三个直走周期替换成 [55% 肩肘参考](2026-11-24-f2-sonic-reduced-arm-references.md)。联合用例统一开启方向过滤、航向引导、斜步世界模型预演与实测收步。对照组保持每段三个入点并加指令误差。直行专门排序组在直走时只取每个周期一个候选入点，稳定直行时使用世界模型跟踪误差；两组的方向过渡保留多入点和指令误差。

| 组别 | 稳态慢 / 中 / 快实测前向（m/s） | 换速快段末半秒横向（m/s） | 旧条件失败数 |
| --- | ---: | ---: | ---: |
| 联合库共享排序 | 0.222 / 0.349 / 0.455 | +0.150 | 26 |
| 独立直走排序 | 0.242 / 0.370 / 0.454 | −0.006 | 23 |

独立排序降低快换速侧漂，但慢段实际前向 0.351 m/s、中段偏航 −0.297 rad/s，收步仍有 0.106 m/s 的前向均速。中速两脚完整摆步为 0 / 2。站立开始主序列首段前向均速 0.257 m/s；转向偏航均值 +0.411 / −0.477 rad/s，但与 F1 比较的速度误差仍超线。右斜走尽管选中了同向库片段，实际平均偏航约 −0.881 rad/s。两组站立起始、行走起始的 F1 完整参考轨迹分别与基线逐字节相同。当前联合组 **F2 未通过**。

从 `supermimic/` 运行。建立 `outputs/f2-sonic-bank-axial-quiet`，从 `outputs/f2-sonic-bank-isolated-axial` 复制所有文件，从 `outputs/f2-sonic-bank-low-arm-55` 覆盖三个 `{slow,medium,fast}-a.npz`。两组均沿用 [独立直走回归](2026-11-24-f2-sonic-axial-regression.md) 的接触组，其上将合成斜步时长改为 0.60 s，添加 `--rank-contact-launch True --contact-launch-forward-lead 0.08`。关闭 `--axial-cycle-ranking` 保存为 `outputs/f2-sonic-mode-ranking-broad`，开启保存为 `outputs/f2-sonic-mode-ranking-one`。完整筛查两组均使用 `check_f2_gait.py outputs/f2-sonic-current-f1 <回放目录> --output-json <回放目录>/criteria.json`。
