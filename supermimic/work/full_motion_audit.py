from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv, sys, time
import numpy as np
import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
SRC=Path('/home/hi/SuperMimic-Reproduced/data/npz/lafan1')
DST=Path('/home/hi/SuperMimic-Official/data/lafan1_official')
_model=None

def audit(p):
 global _model
 if _model is None: _model=mujoco.MjModel.from_xml_path(str(G1_XML))
 m=_model; d=mujoco.MjData(m)
 gids=np.array([i for i in range(m.ngeom) if m.geom_contype[i] and m.geom_conaffinity[i] and int(m.geom_type[i]) in (int(mujoco.mjtGeom.mjGEOM_SPHERE),int(mujoco.mjtGeom.mjGEOM_CAPSULE))],dtype=int)
 feet=np.array([i for i in gids if 'foot' in (m.geom(i).name or '')],dtype=int)
 iscap=m.geom_type[gids]==int(mujoco.mjtGeom.mjGEOM_CAPSULE)
 with np.load(p,allow_pickle=False) as a,np.load(DST/p.name,allow_pickle=False) as b:
  for k in ('joint_pos','joint_vel','joint_names'): assert np.array_equal(a[k],b[k]),f'{p.name} {k}'
  for k in ('pos','quat','lin_vel','ang_vel'): assert np.array_equal(a['body_'+k+'_w'][:,0,:],b['root_'+k+'_w']),f'{p.name} root_{k}'
  n=len(b['joint_pos']); pos=b['joint_pos']; root=b['root_pos_w']; quat=b['root_quat_w']; nms=a['body_names'].tolist()
  assert nms[0]=='pelvis' and np.isclose(float(a['fps'][0]),50)
  lo=m.jnt_range[1:,0];hi=m.jnt_range[1:,1]; exc=np.maximum(lo-pos,0)+np.maximum(pos-hi,0)
  z=np.empty(n,dtype=np.float32); zfoot=np.empty(n,dtype=np.float32); worst_geo=np.empty(n,dtype=np.int16)
  vel=np.diff(root,axis=0)*50; rvel=b['root_lin_vel_w']; rms_vel=float(np.sqrt(np.mean((vel-rvel[:-1])**2)))
  for i in range(n):
   d.qpos[:3]=root[i];d.qpos[3:7]=quat[i];d.qpos[7:]=pos[i];mujoco.mj_kinematics(m,d)
   mat=d.geom_xmat.reshape(-1,3,3)
   lows=d.geom_xpos[gids,2]-m.geom_size[gids,0]-np.where(iscap,m.geom_size[gids,1]*np.abs(mat[gids,2,2]),0)
   j=int(lows.argmin());z[i]=lows[j];worst_geo[i]=gids[j]
   zfoot[i]=np.min(d.geom_xpos[feet,2]-m.geom_size[feet,0]-m.geom_size[feet,1]*np.abs(mat[feet,2,2]))
  i=int(z.argmin()); fi=int(zfoot.argmin())
  def longest(mask):
   arr=np.flatnonzero(np.diff(np.r_[False,mask,False]));return int((arr[1::2]-arr[::2]).max(initial=0))
  return dict(clip=p.stem,frames=n,min_z_m=float(z[i]),worst_frame=i,worst_geom=m.geom(int(worst_geo[i])).name,frames_under_1cm=int((z<-.01).sum()),frames_under_2cm=int((z<-.02).sum()),frames_under_5cm=int((z<-.05).sum()),longest_under_2cm_frames=longest(z<-.02),min_foot_z_m=float(zfoot[fi]),worst_foot_frame=fi,foot_frames_under_1cm=int((zfoot<-.01).sum()),foot_frames_under_2cm=int((zfoot<-.02).sum()),foot_frames_under_5cm=int((zfoot<-.05).sum()),max_joint_limit_excess_rad=float(exc.max()),root_fd_velocity_rms_m_s=rms_vel)

def main():
 files=sorted(SRC.glob('*.npz')); assert len(files)==40
 start=time.time(); rows=[]
 with ProcessPoolExecutor(max_workers=6) as pool:
  for row in pool.map(audit,files):
   rows.append(row);print('checked',len(rows),row['clip'],flush=True)
 out=Path('/home/hi/Documents/Codex/2026-11-11/https-github-com-hybridrobotics-supermimic-git/outputs/40动作逐帧检查.csv')
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print('SUMMARY clips',len(rows),'frames',sum(x['frames'] for x in rows),'seconds',round(time.time()-start,1),flush=True)
 for thresh in (1,2,5):
  key=f'frames_under_{thresh}cm';print(key,'clips',sum(x[key]>0 for x in rows),'frames',sum(x[key] for x in rows),flush=True)
 print('foot_under_2cm clips',sum(x['foot_frames_under_2cm']>0 for x in rows),'frames',sum(x['foot_frames_under_2cm'] for x in rows),flush=True)
 for x in sorted(rows,key=lambda x:x['min_z_m'])[:10]: print(x,flush=True)
if __name__=='__main__':main()
