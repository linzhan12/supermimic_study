from pathlib import Path
import numpy as np
import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

src=Path('/home/hi/SuperMimic-Reproduced/data/npz/lafan1')
dst=Path('../data/lafan1_official')
model=mujoco.MjModel.from_xml_path(str(G1_XML))
expected=[model.joint(i).name for i in range(1,model.njnt)]
files=sorted(src.glob('*.npz'))
assert len(files)==40, f'expected 40 clips, got {len(files)}'
total=0
for p in files:
    with np.load(p, allow_pickle=False) as d:
        names=d['joint_names'].tolist()
        assert names==expected, f'{p.name}: joint names/order mismatch'
        bodies=d['body_names'].tolist()
        assert bodies[0]=='pelvis', f'{p.name}: expected pelvis root'
        assert float(d['fps'].reshape(-1)[0])==50., f'{p.name}: fps differs from environment'
        n=len(d['joint_pos']); assert n>1
        fields={k:d[k] for k in ('joint_pos','joint_vel','joint_names')}
        assert fields['joint_pos'].shape==fields['joint_vel'].shape==(n,len(expected))
        for k in ('pos','quat','lin_vel','ang_vel'):
            a=d['body_'+k+'_w']; assert a.shape[:2]==(n,len(bodies))
            fields['root_'+k+'_w']=a[:,0,:]
        for k,a in fields.items():
            if k!='joint_names': assert np.isfinite(a).all(), f'{p.name}: nonfinite {k}'
        qnorm=np.linalg.norm(fields['root_quat_w'],axis=-1)
        assert np.allclose(qnorm,1,atol=1e-3), f'{p.name}: invalid quaternions'
        np.savez_compressed(dst/p.name,**fields)
        total+=n
print(f'converted {len(files)} clips, {total} frames, {total/50:.2f} seconds at 50 Hz')
