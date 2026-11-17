from pathlib import Path
import numpy as np
import torch
import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.utils.forward_kinematics import ForwardKinematics

torch.set_num_threads(4)
m=mujoco.MjModel.from_xml_path(str(G1_XML))
ids=torch.arange(1,m.nbody,dtype=torch.long); rows=torch.arange(m.nbody-1,dtype=torch.long)
qadr=torch.tensor(m.jnt_qposadr[1:].copy(),dtype=torch.long)
fk=ForwardKinematics(m,ids,qadr,rows,device='cpu')
results=[]
for p in sorted(Path('/home/hi/SuperMimic-Reproduced/data/npz/lafan1').glob('*.npz')):
 with np.load(p) as a, np.load(Path('/home/hi/SuperMimic-Official/data/lafan1_official')/p.name) as b:
  idx=np.unique(np.linspace(0,len(a['joint_pos'])-1,200,dtype=int))
  vars=[torch.as_tensor(b[k][idx],dtype=torch.float32) for k in ('root_pos_w','root_quat_w','joint_pos','root_lin_vel_w','root_ang_vel_w','joint_vel')]
  with torch.no_grad(): recon=[x.numpy() for x in fk(*vars)]
  original=[a[k][idx] for k in ('body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')]
  err=[float(np.max(np.abs(x-y))) for x,y in zip(recon,original)]
  err[1]=float(np.max(np.minimum(np.max(np.abs(recon[1]-original[1]),axis=-1),np.max(np.abs(recon[1]+original[1]),axis=-1))))
  results.append((p.stem,*err))
print('clips:',len(results),'frames:',len(results)*200)
for col,key in enumerate(('position_m','quaternion_component_sign_invariant','linear_velocity_m_s','angular_velocity_rad_s'),start=1):
 name,*vals=max(results,key=lambda x:x[col]);print(key,'max:',vals[col-1],'clip:',name)
