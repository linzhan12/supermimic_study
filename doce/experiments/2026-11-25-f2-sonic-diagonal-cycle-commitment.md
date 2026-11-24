# F2 同方向斜走持有单周期对照

从[骨盆位移 0.48 s 组](2026-11-25-f2-sonic-root-preview-adapter.md)原样恢复控制状态、参考库和两路初态。原一秒左右斜走与延长三秒各运行一次对照。收到相同侧向指令且实测仍有支撑时，当前周期在一整周期内继续按时间播放。方向或松杆改变后走原重规划路径。实验四组均完成，进入左斜走前的 325 步与各自原组逐帧一致。

| 时程和初态 | 左斜走平均前向／横向 m/s | 右斜走平均前向／横向 m/s | 完整摆步 左侧段／右侧段（每段按左／右脚计） | 松杆末半秒速度 m/s |
| --- | --- | --- | --- | ---: |
| 一秒，主 | 0.154／+0.160 | 0.248／−0.129 | 0／1，1／1 | 0.012 |
| 一秒，留出 | 0.154／+0.161 | 0.257／−0.142 | 0／2，0／1 | 0.013 |
| 三秒，主 | 0.172／+0.153 | 0.185／−0.101 | 1／4，1／2 | 0.019 |
| 三秒，留出 | 0.178／+0.157 | 0.183／−0.090 | 2／3，1／1 | 0.019 |

原一秒位移组两个方向窗口均无完整摆步，持有周期后能观察到落脚事件。但右斜走主／留出横速从 −0.192／−0.190 降到 −0.129／−0.142 m/s。三秒条件下左侧第三秒为 +0.108／+0.114 m/s，原路由 +0.103／+0.072，仍达不到 +0.20。后半段弱周期未再入选，速度依然衰减。周期持有影响接触节奏，对响应速度有代价。暂不纳入默认路由，下一组应在同物理初态上比较接缝后的可行落点和摆脚净空。F2 未通过。

## 复现

用骨盆位移 0.48 s 组参数，保留 stand_start 的 planned 路由。在包装入口中，用测试子类替换 eval_f2.LocomotionPlanner。长时程组只把 STAND_START 内两个斜走段各改为 3.0 s，其余命令与阈值一致。子类 update 如下。

~~~python
def update(self, command, observed_contact, score_prediction=None, **kwargs):
    direction = self._direction(command)
    hold = (
        self.clip is not None
        and self.clip.mode == "walk"
        and self.clip.period_steps
        and direction in ((-1, 0), (1, 0))
        and self.active_direction == direction
        and self.offset < self.clip.period_steps
        and bool(observed_contact.any())
    )
    if not hold:
        return super().update(command, observed_contact, score_prediction, **kwargs)
    original = self.committed_steps
    self.committed_steps = self.clip.period_steps
    try:
        return super().update(command, observed_contact, score_prediction, **kwargs)
    finally:
        self.committed_steps = original
~~~

实测目录为 outputs/f2-sonic-diagonal-cycle-lock-{0,1}-{main,holdout}。0 为一秒，1 为三秒。数据均从 stand_start_planned.jsonl 按命令阶段统计。
