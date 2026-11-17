import torch, numpy as np
import mujoco_warp as mjw
import mjlab.tasks, supermimic.tasks
from mjlab.tasks.registry import load_env_cfg
from supermimic.scripts.train_utils import create_env

torch.manual_seed(1234)
cfg=load_env_cfg('SuperMimic-GenCoord-G1',play=False);cfg.seed=1234
cfg.commands['motion'].motion_file='/home/hi/SuperMimic-Official/data/lafan1_official/*.npz'
cfg.commands['motion'].sampling_mode='external'
wrapped=create_env(cfg,256,'cuda:0');env=wrapped.unwrapped;env.reset()
mask=int(env.sim.wp_model.opt.warn_overflow)
suppressed=int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
print('WARNING_MASK:',mask,'SUPPRESSED_BITS:',suppressed,flush=True)
assert (mask & suppressed)==0
assert (mask | suppressed)==int(mjw.OverflowType.ALL)
for i in range(8):
 env.step(torch.zeros((env.num_envs,env.action_manager.total_action_dim),device='cuda:0'))
torch.cuda.synchronize();flags=env.sim.wp_data.overflow.numpy()
print('SOLVER_FLAGGED:',int(np.count_nonzero(flags & int(mjw.OverflowType.ITERATIONS))))
print('LINESEARCH_FLAGGED:',int(np.count_nonzero(flags & int(mjw.OverflowType.LS_ITERATIONS))))
print('ALL_JOINTS_FINITE:',bool(torch.isfinite(env.scene['robot'].data.joint_pos).all().item()))
wrapped.close()
