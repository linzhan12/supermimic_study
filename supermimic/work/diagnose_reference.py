from pathlib import Path
import csv
import numpy as np
import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

src=Path('/home/hi/SuperMimic-Reproduced/data/npz/lafan1')
dst=Path('/home/hi/SuperMimic-Official/data/lafan1_official')
out=Path('/home/hi/Documents/Codex/2026-11-11/https-github-com-hybridrobotics-supermimic-git/outputs/reference_diagnostic.csv')
m=mujoco.MjModel.from_xml_path(str(G1_XML)); d=mujoco.MjData(m)
names=[m.body(i).name for i in range(1,m.nbody)]
gids=np.array([i for i in range(m.ngeom) if m.geom_contype[i] and m.geom_conaffinity[i] and int(m.geom_type[i]) in [int(mujoco.mjtGeom.mjGEOM_CAPSULE),int(mujoco.mjtGeom.mjGEOM_SPHERE)]])
feet=np.array([i for i in gids if 'foot' in (m.geom(i).name or '')])
qmin=m.jnt_range[1:,0]; qmax=m.jnt_range[1:,1]
rows=[]
for p in sorted(src.glob('*.npz')):
 with np.load(p) as a, np.load(dst/p.name) as b:
  assert names==a['body_names'].tolist()
  for k in ('joint_pos','joint_vel','joint_names'): assert np.array_equal(a[k],b[k]),(p,k)
  for k in ('pos','quat','lin_vel','ang_vel'): assert np.array_equal(a['body_'+k+'_w'][:,0,:], b['root_'+k+'_w']),(p,k)
  q=b['joint_pos']; n=len(q); idx=np.unique(np.linspace(0,n-1,min(n,200),dtype=int))
  low=1e9; flow=1e9; penetration=0; deep=0; err=0; worst_frame=0
  for i in idx:
   d.qpos[:3]=b['root_pos_w'][i];d.qpos[3:7]=b['root_quat_w'][i];d.qpos[7:]=q[i];mujoco.mj_kinematics(m,d)
   poserr=np.linalg.norm(d.xpos[1:]-a['body_pos_w'][i],axis=-1).max();err=max(err,float(poserr))
   mat=d.geom_xmat.reshape(-1,3,3); lows=d.geom_xpos[gids,2]-m.geom_size[gids,0]-np.where(m.geom_type[gids]==int(mujoco.mjtGeom.mjGEOM_CAPSULE),m.geom_size[gids,1]*np.abs(mat[gids,2,2]),0)
   mval=float(lows.min());
   if mval<low:low=mval;worst_frame=int(i)
   fl=d.geom_xpos[feet,2]-m.geom_size[feet,0]-m.geom_size[feet,1]*np.abs(mat[feet,2,2]);flow=min(flow,float(fl.min()))
   penetration+=mval < -0.01;deep+=mval < -0.05
  beyond=np.maximum(qmin-q,0)+np.maximum(q-qmax,0)
  rows.append(dict(clip=p.stem,samples=len(idx),sample_min_geom_z_m=low,sample_min_foot_z_m=flow,samples_below_minus_1cm=penetration,samples_below_minus_5cm=deep,worst_frame=worst_frame,body_position_max_diff_m=err,joint_max_limit_excess_rad=float(beyond.max()),frames_with_limit_excess=int((beyond>1e-3).any(axis=1).sum()),frames=n))
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('clips',len(rows),'sample_frames',sum(r['samples'] for r in rows),'all converted fields exact')
print('worst FK body diff',max(r['body_position_max_diff_m'] for r in rows))
print('clips with >1cm sampled penetration',sum(r['samples_below_minus_1cm']>0 for r in rows),'of 40; >5cm',sum(r['samples_below_minus_5cm']>0 for r in rows))
for r in sorted(rows,key=lambda x:x['sample_min_geom_z_m'])[:10]: print(r['clip'],'min_z',round(r['sample_min_geom_z_m'],4),'feet',round(r['sample_min_foot_z_m'],4),'count<-1cm',r['samples_below_minus_1cm'],'frame',r['worst_frame'],'joint_max',round(r['joint_max_limit_excess_rad'],4))
print('clips exceeding joint limits >0.001rad:',sum(r['frames_with_limit_excess']>0 for r in rows))
