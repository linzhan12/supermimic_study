from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv
import numpy as np
import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
SRC=Path('/home/hi/SuperMimic-Official/data/lafan1_official')
OUT=Path('/home/hi/Documents/Codex/2026-11-11/https-github-com-hybridrobotics-supermimic-git/outputs/40动作自碰撞抽查.csv')
_model=None

def audit(p):
 global _model
 if _model is None: _model=mujoco.MjModel.from_xml_path(str(G1_XML))
 m=_model;d=mujoco.MjData(m)
 with np.load(p,allow_pickle=False) as a:
  q=a['joint_pos'];qd=a['joint_vel'];n=len(q);root=a['root_pos_w'];rq=a['root_quat_w'];sample=np.unique(np.linspace(0,n-1,min(200,n),dtype=int))
  maxdepth=0;worstframe=-1;pair='';penetrated=0;over1=0
  for frame in sample:
   d.qpos[:3]=root[frame];d.qpos[3:7]=rq[frame];d.qpos[7:]=q[frame];d.qvel[:]=0;mujoco.mj_forward(m,d)
   contacts=[c for c in d.contact if c.dist<0]
   if contacts:
    penetrated+=1;c=min(contacts,key=lambda x:x.dist)
    if c.dist < -.01: over1+=1
    if -float(c.dist)>maxdepth: maxdepth=-float(c.dist);worstframe=int(frame);pair=f'{m.geom(c.geom1).name}|{m.geom(c.geom2).name}'
  delta=np.diff(q,axis=0)*50
  rms=float(np.sqrt(np.mean((delta-qd[:-1])**2))); jump=float(np.max(np.abs(np.diff(q,axis=0))))
  return dict(clip=p.stem,samples=len(sample),sample_frames_with_self_contact=penetrated,sample_frames_self_over_1cm=over1,max_self_penetration_m=maxdepth,worst_frame=worstframe,worst_pair=pair,joint_velocity_finite_diff_rms_rad_s=rms,max_joint_step_rad=jump)
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=6) as pool:r=list(pool.map(audit,sorted(SRC.glob('*.npz'))))
 with OUT.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0]);w.writeheader();w.writerows(r)
 print('clips',len(r),'self contact clips',sum(x['sample_frames_with_self_contact']>0 for x in r),'self over1cm clips',sum(x['sample_frames_self_over_1cm']>0 for x in r))
 print('sampled frames with self contact',sum(x['sample_frames_with_self_contact'] for x in r),'over1cm',sum(x['sample_frames_self_over_1cm'] for x in r))
 for x in sorted(r,key=lambda x:-x['max_self_penetration_m'])[:10]: print(x)
 print('highest vel finite diff rms:',sorted((x['joint_velocity_finite_diff_rms_rad_s'],x['clip']) for x in r)[-5:])
