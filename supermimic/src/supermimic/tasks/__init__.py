"""将 SuperMimic 两种状态表示的任务名称登记到 mjlab 注册表。

本文件没有训练循环；它作为包入口，在 import supermimic.tasks 时执行顶层注册。
注册表保存任务名到训练环境配置、播放环境配置及 RL 配置的映射。
训练脚本先导入 mjlab.tasks 注册基础 G1 任务，再导入本包注册两个扩展任务，
随后即可通过 list_tasks/load_env_cfg 按字符串选任务。

两个任务共享 G1 跟踪基础，但 full_state 布局分别为广义坐标和逐刚体最大坐标。
配置工厂定义于 supermimic/src/supermimic/tasks/env_cfgs.py；这里仅负责绑定名称。
"""

# 【文件作用】任务注册入口
# 将广义坐标和最大坐标两种 G1 跟踪任务注册到 mjlab，供训练和回放按名称查找。
# 【阅读顺序】导入环境配置工厂 → 创建训练/播放配置 → 注册任务；导入本模块即执行注册。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 来自 mjlab_research/src/mjlab/tasks/registry.py：load_rl_cfg 深拷贝基础任务的 RL 设置，
# register_mjlab_task 把训练/播放配置写入模块级注册表，并拒绝重复 task_id。
from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task

# 点号表示同一个包下的模块：supermimic/src/supermimic/tasks/env_cfgs.py。
# 两个工厂分别构造完整状态布局不同的环境，play=True 时使用播放设置。
from .env_cfgs import gencoord_env_cfg, maxcoord_env_cfg

# 基础任务必须已注册，下面 load_rl_cfg 才能查询其 RL 设置；环境观察仍由本项目工厂构建。
BASE_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"

# 导入此模块即向 mjlab 注册任务；训练和回放分别使用对应的环境配置。
# 训练配置与播放配置分别创建；play=True 让基础任务选择播放用设置。
# register_mjlab_task 把名称映射到配置，供入口通过 load_env_cfg 查询。
# 模块被重复加载时可能触发已注册错误，下面保留已有注册结果。
try:
  register_mjlab_task(
    task_id="SuperMimic-GenCoord-G1",
    env_cfg=gencoord_env_cfg(),
    play_env_cfg=gencoord_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # 当前调用链中重复 task_id 会抛出 ValueError，此处忽略该异常，保留已有注册。
  pass

# 最大坐标版本使用不同名称；环境的完整状态改为逐刚体布局，训练/播放须选匹配任务。
try:
  register_mjlab_task(
    task_id="SuperMimic-MaxCoord-G1",
    env_cfg=maxcoord_env_cfg(),
    play_env_cfg=maxcoord_env_cfg(play=True),
    rl_cfg=load_rl_cfg(BASE_TASK_ID),
  )
except ValueError:
  # 当前调用链中重复 task_id 会抛出 ValueError，此处忽略该异常，保留已有注册。
  pass
