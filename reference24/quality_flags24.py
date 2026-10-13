"""Sensor-support flags for offline references; not calibrated confidence scores."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,argparse,json,hashlib
from pathlib import Path
sys.path.append(str(Path.cwd()))
import numpy as np
from scipy.spatial.transform import Rotation
import backend as B
from imu_constraints24 import covered_gyro_intervals

BITS={1:'imu_interval_uncovered_or_gap_over_15ms',2:'handling_position_assumption',
      4:'both_visual_steps_missing_or_outside_existing_motion_gate',8:'image_interval_over_2x_median',
      16:'frontend_reinitialization_nearby'}

def intervals(mask,t):
    start=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))==1)
    end=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))==-1)-1
    return [[float(t[a]),float(t[b])] for a,b in zip(start,end)]

def generate(tag,fe,out):
    z=np.load(fe/f'{tag}_fe.npz');t=z['t'];ts=t*1e-9;dt=np.diff(t)*1e-9
    flags=np.zeros(len(t),np.uint16)
    _,covered,coverage=covered_gyro_intervals(z['imu'],t,B.calib['calib_gyro_bias'][:3])
    missing=~covered;flags[:-1][missing]|=1;flags[1:][missing]|=1
    held=B.handling_mask(z['imu'],t);flags[held]|=2
    plausible=[]
    for name in ['pose_in','pose_st']:
        a=z[name];valid=np.isfinite(a).all(1);pair=valid[:-1]&valid[1:]
        dp=np.linalg.norm(np.diff(np.nan_to_num(a[:,:3]),axis=0),axis=1)
        qq=a[:,3:7].copy();qq[~valid]=[0,0,0,1]
        rr=Rotation.from_quat(qq)
        dr=(rr[:-1].inv()*rr[1:]).magnitude()
        plausible.append(pair&(dp<1.5*dt)&(dr<np.radians(150)*dt))
    bad=~(plausible[0]|plausible[1]);flags[1:][bad]|=4
    gap=dt>2*np.median(dt);flags[:-1][gap]|=8;flags[1:][gap]|=8
    if 'reinit_t' in z:
        for rt in z['reinit_t']:flags[abs(t-rt)<1e9]|=16
    breakdown={label:dict(frames=int(((flags&bit)!=0).sum()),intervals_s=intervals((flags&bit)!=0,ts)) for bit,label in BITS.items()}
    summary=dict(tag=tag,frames=len(t),flagged_frames=int((flags!=0).sum()),bits=BITS,by_reason=breakdown,
        gyro_coverage=coverage,bag_limitations=['No independent ground truth; flags show sensor support, not absolute accuracy.',
        'Physical return is approximately within0.2m; endpoint distance alone is not an error measurement.',
        '15ms IMU gap and1.5m/s/150deg/s visual gates are explicit diagnostic thresholds, not estimated confidence intervals.'])
    if held.any():summary['bag_limitations'].append('Original estimator assumes return to pickup position while handled; unknown displacement can affect later positions too. Handling intervals must not be used as position truth.')
    np.savez(out/f'{tag}_flags.npz',t_ns=t,flags=flags)
    (out/f'{tag}_quality.json').write_text(json.dumps(summary,indent=2))
    return {k:v for k,v in summary.items() if k not in ('bits','by_reason')}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fe-dir',type=Path,default=Path('results/fe5'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    rows=[generate(f.name[:-7],a.fe_dir,a.output) for f in sorted(a.fe_dir.glob('*_fe.npz'))]
    (a.output/'summary.json').write_text(json.dumps(dict(bags=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('quality flags generated for',len(rows),'bags')

if __name__=='__main__':main()
