"""解析训练配置、创建物理环境并启动 SuperMimic 世界模型与策略联合训练。"""

# 【文件作用】训练启动入口
# 解析任务和配置，确定设备与动作来源，创建物理仿真环境并启动训练组织器。
# 【阅读顺序】任务注册 → 参数解析 → 动作与采样配置 → 环境创建 → SuperMimicRunner.run。
# 下方先说明各函数用途，再在函数体内解释数据、步骤与边界条件。


# 推迟求值类型注解：下面的类型标注主要供阅读、检查和命令行解析使用。
from __future__ import annotations



# os 提供路径相关工具；dataclass 帮我们生成保存配置所需的初始化方法。
import math
import os
from dataclasses import dataclass

# 导入任务包时，其顶层注册代码会执行；因此这里即使不直接引用该名字也不能随意删除。
import mjlab.tasks  # noqa: F401
# tyro 根据 Python 类型和 dataclass 字段，构造并解析命令行参数。
import tyro
# mjlab 环境配置类型：场景、动作、观察、奖励、结束条件等都由它组织；不是环境实例。
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
# 来自 mjlab_research/src/mjlab/tasks/registry.py：list_tasks 返回已注册任务名；
# load_env_cfg 按任务名深拷贝训练或播放配置，避免本次覆盖字段污染注册表原配置。
from mjlab.tasks.registry import list_tasks, load_env_cfg

# 这里导入本工程的任务注册模块；注册后才能通过任务名查到对应配置。
# noqa: F401 告诉代码检查器：此处导入虽未直接引用，仍然有注册用途。
import supermimic.tasks  # noqa: F401
# 来自 supermimic/src/supermimic/runners/supermimic_runner.py：Cfg 保存算法设置，
# Runner 负责创建模型和经验池、采集数据、交替训练、记录及保存；这里只调用它的入口。
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
# 来自 supermimic/src/supermimic/scripts/train_utils.py：setup_device 选择设备；
# download_registry_motions 获取 registry 动作路径；create_env 设置并行数并包装物理环境。
# 导入的是辅助函数，实际网络和优化器仍由 runner 及 learner 构造。
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)
from supermimic.tasks.forward_gait import ForwardGaitPrior
from supermimic.tasks.velocity_posture import VelocityArmSwingReward, VelocityPostureReward
from supermimic.tasks.velocity_tracking import configure_velocity_env, physical_turn_tracking_reward


# @dataclass 为配置类自动生成初始化方法；类可以理解为配置对象的模板。
@dataclass
class TrainConfig:
  """训练入口所需的完整配置，作为 tyro 第二阶段参数解析的模板。

  env 描述物理世界和任务，runner 描述网络、回放和训练算法；num_envs、device
  及 motion_file/registry_name 决定本次执行资源与动作来源。
  from_task 先生成任务默认值，main 再用命令行覆盖，最终据此创建实际对象。
  本类只保存参数，不实例化仿真器，也不执行训练循环。
  """
  # env 描述仿真环境，runner 描述网络、采集、训练和保存等设置。
  # 冒号后的内容是类型注解；这里声明字段，具体值由 from_task 提供。
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  # 同时运行的仿真环境数量；可用 --num-envs 覆盖，并非训练轮数。
  num_envs: int = 4096
  # None 表示未指定设备：setup_device 稍后根据 CUDA 是否可用选择。
  # str | None 表示该字段预期是字符串或 None。
  device: str | None = None
  # 未传本地动作时，默认尝试使用这个 W&B registry 中的动作来源。
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  # 本地动作路径；None 表示没有指定。命令行中对应 --motion-file。
  motion_file: str | None = None

  # 静态方法不需要 self；调用形式是 TrainConfig.from_task(任务名)。
  # 函数作用：按任务名生成训练默认配置，并返回一个新 TrainConfig。
  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    """按任务名生成训练默认配置，并返回一个新 TrainConfig。

    task_id 必须已在导入阶段注册；load_env_cfg 返回训练配置的深拷贝，
    runner 使用当前默认 SuperMimicRunnerCfg。静态方法无需已有配置实例。
    此处不解析其余命令行参数，不创建环境，参数覆盖发生在 main 的第二次 tyro.cli。
    """
    # 按已注册的任务名读取环境配置；play=False 选择训练用配置。
    # task_id: str 和 -> "TrainConfig" 分别标明参数和返回值的预期类型。
    env_cfg = load_env_cfg(task_id, play=False)
    # 返回一个新配置对象；括号中的 name=value 是按参数名传值。
    return TrainConfig(
      env=env_cfg,  # 环境配置来自所选任务。
      runner=SuperMimicRunnerCfg(),  # 训练组织器使用自己的默认配置。
    )


# main 是本文件的主入口；-> None 表示没有供调用方使用的返回值。
# 函数作用：处理一次训练启动，最终进入 runner.run，正常完成时返回 None。
def main() -> None:
  """处理一次训练启动，最终进入 runner.run，正常完成时返回 None。

  从进程命令行先解析任务，再解析该任务对应的完整配置；设置设备和动作来源，
  确定坐标表示及重置模式后创建物理环境，最后构建训练组织器并执行其主循环。
  本地动作优先于 registry；motion 和采样模式必须在创建环境之前写入配置。
  网络参数更新发生在 runner 调用的 learner 内，入口只负责依赖和配置的装配。
  """
  # 第一次解析只确定任务名；其他命令行参数先保留，供下面第二次解析。
  # 查询注册表里当前可选的任务名。
  all_tasks = list_tasks()
  # 返回的两个值分别赋给 chosen_task 和 remaining_args，这叫序列解包。
  chosen_task, remaining_args = tyro.cli(
    # 任务名必须来自 all_tasks 中的候选项。
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,  # 第一阶段不单独添加帮助选项。
    return_unknown_args=True,  # 将尚未解析的选项交给第二阶段处理。
    config=mjlab.TYRO_FLAGS,  # 使用 mjlab 项目的 tyro 约定。
  )
  # 根据任务构造默认配置，再用剩余命令行参数覆盖相应字段。
  args = tyro.cli(
    TrainConfig,  # 根据配置类的字段和类型建立参数接口。
    args=remaining_args,  # 接着解析上一阶段留下的参数。
    default=TrainConfig.from_task(chosen_task),  # 先按任务生成默认值。
    config=mjlab.TYRO_FLAGS,
  )

  if args.runner.init_checkpoint_dir and args.runner.wandb_run_path:
    raise ValueError("Choose one checkpoint initialization source.")
  posture_weight = args.runner.posture_weight
  moving_posture_scale = args.runner.moving_posture_scale
  arm_swing_weight = args.runner.arm_swing_weight
  arm_swing_control = args.runner.arm_swing_control
  turn_tracking_weight = args.runner.turn_tracking_weight
  gait_weight = args.runner.forward_gait_weight
  gait_limit = args.runner.forward_gait_action_limit
  gait_fraction = args.runner.forward_gait_command_fraction
  if not math.isfinite(gait_weight) or gait_weight < 0:
    raise ValueError("Forward gait reward weight must be finite and nonnegative.")
  if not 0 <= gait_limit <= 0.7 or not 0 <= gait_fraction <= 1:
    raise ValueError("Forward gait action limit must be in [0, 0.7] and fraction in [0, 1].")
  if (gait_limit or gait_fraction or gait_weight) and not args.runner.independent_velocity:
    raise ValueError("Forward gait shaping requires the independent velocity task.")
  if not math.isfinite(posture_weight) or posture_weight < 0:
    raise ValueError("Posture reward weight must be finite and nonnegative.")
  if not 0.0 <= moving_posture_scale <= 1.0:
    raise ValueError("Moving posture scale must be between 0 and 1.")
  if not math.isfinite(arm_swing_weight) or arm_swing_weight < 0:
    raise ValueError("Arm swing reward weight must be finite and nonnegative.")
  if not 0 <= arm_swing_control <= 1:
    raise ValueError("Arm swing control must be between 0 and 1.")
  if not 0 <= args.runner.arm_swing_roll <= 0.45:
    raise ValueError("Arm swing shoulder roll must be between 0 and 0.45 radians.")
  if args.runner.arm_swing_roll and not arm_swing_control:
    raise ValueError("An outward shoulder target requires arm swing control.")
  if not 0 <= args.runner.arm_swing_speed_gate <= 0.35:
    raise ValueError("Arm swing speed gate must be between 0 and 0.35 m/s.")
  if args.runner.arm_swing_speed_gate and not arm_swing_control:
    raise ValueError("An arm swing speed gate requires arm swing control.")
  if args.runner.stop_encoder_init_dir and not args.runner.independent_velocity:
    raise ValueError("A separate stop encoder requires independent velocity training.")
  if args.runner.stop_encoder_init_dir and not args.runner.init_checkpoint_dir:
    raise ValueError("A separate stop encoder requires a local moving-policy checkpoint.")
  if not math.isfinite(turn_tracking_weight) or turn_tracking_weight < 0:
    raise ValueError("Turn tracking weight must be finite and nonnegative.")
  if (posture_weight or arm_swing_weight or arm_swing_control or turn_tracking_weight) and not args.runner.independent_velocity:
    raise ValueError("F2 posture, arm swing and turn settings require the independent velocity task.")
  if args.runner.independent_velocity:
    if chosen_task != "SuperMimic-GenCoord-G1" or args.runner.representation != "gencoord":
      raise ValueError("Independent velocity fine-tuning requires SuperMimic-GenCoord-G1.")
    if not args.runner.init_checkpoint_dir or not args.motion_file:
      raise ValueError("F1 requires a local checkpoint and local reference motion files.")

  # 若用户未指定设备，工具函数会在 CUDA 可用时选择 cuda:0，否则选择 cpu。
  device = setup_device(args.device)
  # f"...{变量}..." 是格式化字符串，花括号里的值会插入显示文本。
  print(f"[INFO] Loading environment: {chosen_task}")

  # 本地文件优先；路径可以是包含通配符的字符串，后续由动作加载器解析。
  if args.motion_file:
    # 字符串中含逗号时，将其视为多个动作路径；否则当作单个路径处理。
    if "," in args.motion_file:
      # 依次访问环境配置、commands 字典里的 motion 项和它的 motion_file 字段。
      args.env.commands["motion"].motion_file = [
        # split 按逗号拆分；strip 去两端空白；expanduser 展开路径开头的 ~。
        # 这是列表推导式：对拆出的每个 p 执行上述处理，结果组成列表。
        os.path.expanduser(p.strip()) for p in args.motion_file.split(",")
      ]
    else:
      # 单个字符串也可以包含 *.npz；这里不会展开 glob，加载器之后才处理。
      args.env.commands["motion"].motion_file = os.path.expanduser(args.motion_file)
    # 已使用本地文件，清除原 registry 名称，避免后续记录错误的来源。
    args.registry_name = None  # 避免关联与本次本地动作无关的 registry 来源
    print(f"[INFO] Using local motion file: {args.motion_file}")
  else:
    # 未指定本地路径时尝试从 W&B registry 获取；None 表示没有可用路径。
    motion_file = download_registry_motions(args.registry_name)
    if motion_file is not None:
      args.env.commands["motion"].motion_file = motion_file

  # 若任务名包含 MaxCoord 且当前配置仍为默认 gencoord，则切换表示。
  # 这是基于当前字段值的判断，不能区分默认 gencoord 与用户显式指定的 gencoord。
  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  # return 分布由 runner 计算；mjlab 接收外部分布的模式名是 external。
  # 必须在 create_env 之前写入配置，让环境构造时读到正确模式。
  # x if 条件 else y 是条件表达式：条件为真取 x，否则取 y。
  args.env.commands["motion"].sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )

  if args.runner.independent_velocity:
    args.runner.slot_mix = {"motion": 0.0, "velocity": 1.0}
    args.runner.latent_bc_weight = 0.0
    args.runner.sampling_mode = "uniform"
    configure_velocity_env(args.env, gait_fraction=gait_fraction)
    if gait_weight:
      args.env.rewards["forward_gait"] = RewardTermCfg(func=ForwardGaitPrior, weight=gait_weight)
    if posture_weight:
      args.env.rewards["posture"] = RewardTermCfg(
        func=VelocityPostureReward, weight=posture_weight,
        params={"moving_scale": moving_posture_scale},
      )
    if arm_swing_weight:
      args.env.rewards["arm_swing"] = RewardTermCfg(
        func=VelocityArmSwingReward, weight=arm_swing_weight,
      )
    if turn_tracking_weight:
      args.env.rewards["turn_tracking"] = RewardTermCfg(
        func=physical_turn_tracking_reward, weight=turn_tracking_weight,
      )

  # 按最终配置、并行环境数和设备创建并包装物理仿真环境。
  # 前面对动作文件、表示和采样方式的配置应在此之前完成。
  env = create_env(args.env, args.num_envs, device)

  # 构造训练组织器：初始化世界模型、策略、经验缓冲区等，具体见 runner 文件。
  runner = SuperMimicRunner(
    env=env,  # 上面创建的仿真环境。
    cfg=args.runner,  # 模型、采集、训练和保存配置。
    task_id=chosen_task,  # 本次选择的任务名称。
    registry_name=args.registry_name,  # 动作来源；使用本地文件时为 None。
    device=device,  # 张量和模型使用的设备。
  )

  # 真正开始采集、训练和保存；构造 runner 本身并不等于完成训练。
  runner.run()


# 直接运行本文件时 __name__ 是 "__main__"，于是调用 main()；
# 被其他模块导入时不会自动调用 main，但文件顶层的导入仍会执行。
if __name__ == "__main__":
  main()
