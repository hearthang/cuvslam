"""Compare a full-frame candidate using common-time and raw observation metrics."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import sys,json,hashlib
from pathlib import Path
sys.path.append(str(Path.cwd()))
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import backend as B
import evaluate_reference24 as E
from visual_inertial_ba24 import geometry

def residuals(poses,points,inputs,extr,kd):
    errors=[];behind=0;near=0
    for i in np.unique(inputs['pose_idx']):
        for c in (0,1):
            take=(inputs['pose_idx']==i)&(inputs['camera_idx']==c)
            if not take.any():continue
            T=np.linalg.inv(poses[i]@extr[c]);xyz=points[inputs['point_idx'][take]]
            Y=xyz@T[:3,:3].T+T[:3,3];good=np.isfinite(Y).all(1)&(Y[:,2]>0)
            behind+=int((~good).sum());near+=int((Y[:,2]<.3).sum())
            uv,_=cv2.fisheye.projectPoints(Y[good].reshape(-1,1,3),np.zeros(3),np.zeros(3),*kd[c])
            errors.extend(np.linalg.norm(uv.reshape(-1,2)-inputs['pixels'][take][good],axis=1))
    return dict(valid_observations=len(errors),behind_camera=behind,closer_than_30cm=near,
                median_px=float(np.median(errors)),p90_px=float(np.quantile(errors,.9)))

def main():
    tag=sys.argv[1];variant=sys.argv[2] if len(sys.argv)>2 else 'full_bundle_klt'
    suffix=sys.argv[3] if len(sys.argv)>3 else '_kltbundle24'
    parent=Path(sys.argv[sys.argv.index('--root')+1]) if '--root' in sys.argv else Path('results/reference_24h_20261010/colmap_corrected')
    root=parent/tag
    d=root/variant
    bundle_name='vi_bundle_klt.npz' if 'klt' in variant else 'vi_bundle_input.npz'
    with np.load(root/bundle_name) as archive:z={k:archive[k] for k in archive.files}
    with np.load(d/'optimized.npz') as archive:optimized={k:archive[k] for k in archive.files}
    meta=json.loads(str(z['metadata']));extr,_,kd=geometry(meta)
    before=E.load_trajectory(Path('results/reference_v1/trajectories')/f'{tag}.txt')
    after=E.load_trajectory(Path('results/fe5')/f'{tag}_ref{suffix}.txt')
    np.testing.assert_allclose(after[:,0],before[:,0],rtol=0,atol=1e-8)
    t=z['timestamps']
    if t[0]<before[0,0]-1e-6 or t[-1]>before[-1,0]+1e-6:raise ValueError('Observation times outside trajectory')
    p,r=E.interpolate(before,np.clip(t,before[0,0],before[-1,0]));P0=np.tile(np.eye(4),(len(p),1,1));P0[:,:3,:3]=r.as_matrix();P0[:,:3,3]=p
    bias=np.array(json.load(open('/mnt/f/data_set/slam/calibration_front_small.json'))['value0']['calib_gyro_bias'][:3])
    common=E.bag((tag,dict(v1='results/reference_v1/trajectories/{tag}.txt',candidate=suffix),Path('results/fe5'),Path('/mnt/f/data_set/slam'),bias))
    delta=np.linalg.norm(after[:,1:4]-before[:,1:4],axis=1)
    rotation_delta=np.degrees((Rotation.from_quat(before[:,4:8]).inv()*Rotation.from_quat(after[:,4:8])).magnitude())
    B.configure()
    fe=np.load(Path('results/fe5')/f'{tag}_fe.npz');kf=fe['kf_idx']
    base=np.load(Path('results/fe5')/f'{tag}_loops_v11.npz');extra=np.load(Path('results/fe5')/f'{tag}_extra_reference_a1ed4af676.npz')
    pairs=np.r_[base['ij'][base['active']],extra['ij']];loopT=np.r_[base['T'][base['active']],extra['T']]
    loop_metrics={}
    for name,rows in [('v1',before),('candidate',after)]:
        poses=np.array([B.mat(row[1:]) for row in rows]);translation=[];rotation=[]
        for (i,j),T in zip(pairs,loopT):
            err=np.linalg.inv(T)@np.linalg.inv(poses[kf[i]])@poses[kf[j]]
            translation.append(np.linalg.norm(err[:3,3]));rotation.append(np.degrees(Rotation.from_matrix(err[:3,:3]).magnitude()))
        loop_metrics[name]=dict(n=len(pairs),translation_median_m=float(np.median(translation)),translation_p90_m=float(np.quantile(translation,.9)),rotation_median_deg=float(np.median(rotation)),rotation_p90_deg=float(np.quantile(rotation,.9)))
    validation=json.load(open(d/'validation.json'));stats=json.load(open(Path('results/fe5')/f'{tag}_ref{suffix}.json'))
    active=optimized.get('active_points',np.ones(len(z['points']),bool))
    observations=active[z['point_idx']]
    filtered=dict(z)
    for key in ('pose_idx','camera_idx','point_idx','pixels'):filtered[key]=z[key][observations]
    summary=dict(tag=tag,variant=variant,common_metrics=common['metrics'],validation=validation,
        training_before=residuals(P0,z['points'],filtered,extr,kd),training_after=residuals(optimized['poses'],optimized['points'],filtered,extr,kd),
        rejected_training_tracks=int((~active).sum()),rejected_training_observations=int((~observations).sum()),
        displacement_from_v1_p95_m=float(np.quantile(delta,.95)),displacement_from_v1_max_m=float(delta.max()),
        rotation_from_v1_p95_deg=float(np.quantile(rotation_delta,.95)),rotation_from_v1_max_deg=float(rotation_delta.max()),
        original_loop_residuals=loop_metrics,solver=stats,
        note='Metrics characterize consistency and change relative to v1, not absolute accuracy. Loop measurements were used by both solvers; Basalt and v1 are not ground truth.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (d/'diagnostics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('validation','solver','common_metrics')},indent=2),flush=True)

if __name__=='__main__':main()
