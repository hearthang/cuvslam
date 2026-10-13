"""Joint keyframe/landmark BA: stereo projections, IMU rotation, gravity, weak motion.

Whole validation tracks are never inserted into the graph. No endpoint constraint.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import sys,argparse,json,time,hashlib
from pathlib import Path
sys.path.append(str(Path.cwd()))
import numpy as np
import cv2
import gtsam
from gtsam.symbol_shorthand import X,L
from scipy.spatial.transform import Rotation
import backend as B
import reference as R
import evaluate_reference as E
from imu_constraints24 import covered_gyro_intervals

def geometry(meta):
    T0=np.array(meta['imu_from_left']);right_left=np.eye(4);right_left[:3]=meta['right_from_left']
    extr=[T0,T0@np.linalg.inv(right_left)];cal=[];kd=[]
    for name in ['left','right']:
        q=meta['camera_'+name]['params'];fx,fy,cx,cy,*d=q
        cal.append(gtsam.Cal3Fisheye(fx,fy,0,cx,cy,*d))
        kd.append((np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),np.array(d)))
    return extr,cal,kd

def validate(poses,tracks,extr,kd):
    cameras=np.array([[np.linalg.inv(T@e) for e in extr] for T in poses])
    errors=[];behind=0
    for track in tracks:
        A=[]
        for i,c,xy in track['train']:
            uv=cv2.fisheye.undistortPoints(np.array(xy,float).reshape(1,1,2),*kd[c]).ravel()
            P=cameras[i,c,:3];A.extend([uv[0]*P[2]-P[0],uv[1]*P[2]-P[1]])
        _,_,V=np.linalg.svd(A);point=V[-1,:3]/V[-1,3]
        for i,c,xy in track['test']:
            Y=cameras[i,c,:3,:3]@point+cameras[i,c,:3,3]
            if Y[2]<=.3:
                errors.append(1000.);behind+=1;continue
            uv,_=cv2.fisheye.projectPoints(Y.reshape(1,1,3),np.zeros(3),np.zeros(3),*kd[c])
            errors.append(float(np.linalg.norm(uv.ravel()-xy)))
    return dict(points=len(tracks),observations=len(errors),behind_camera=behind,
                median_px=float(np.median(errors)),p90_px=float(np.quantile(errors,.9)),errors_px=errors)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('input',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--iterations',type=int,default=80)
    ap.add_argument('--use-verified-loops',action='store_true',help='Keep all original keyframes and measured v1 loop constraints')
    ap.add_argument('--return-radius',type=float,default=None,help='Optional approximate physical return region, meters; never a zero-endpoint target')
    ap.add_argument('--return-sigma',type=float,default=.1)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    started=time.monotonic();z=np.load(args.input);meta=json.loads(str(z['metadata']));tag=meta['tag']
    extr,cal,kd=geometry(meta);tracks=json.loads(str(z['validation']))
    fe=np.load(Path('results/fe5')/f'{tag}_fe.npz')
    selected=np.array(meta['selected_kf'],int) if args.use_verified_loops else np.arange(len(z['timestamps']))
    observation_pose_indices=selected[z['pose_idx']]
    t=fe['t'][fe['kf_idx']]*1e-9 if args.use_verified_loops else z['timestamps']
    a=E.load_trajectory(Path('results/reference_v1/trajectories')/f'{tag}.txt')
    if t[0]<a[0,0]-1e-6 or t[-1]>a[-1,0]+1e-6:
        raise ValueError('Keyframe timestamps are outside the initial trajectory')
    t=np.clip(t,a[0,0],a[-1,0])  # serialization roundoff only, after explicit coverage check
    pp,rr=E.interpolate(a,t);poses=np.tile(np.eye(4),(len(t),1,1));poses[:,:3,:3]=rr.as_matrix();poses[:,:3,3]=pp
    before=validate(poses[selected],tracks,extr,kd)
    graph=gtsam.NonlinearFactorGraph();values=gtsam.Values();order=gtsam.Ordering()
    for i,point in enumerate(z['points']):values.insert(L(i),point);order.push_back(L(i))
    for i,T in enumerate(poses):values.insert(X(i),R.pose3(T));order.push_back(X(i))
    graph.add(gtsam.PriorFactorPose3(X(0),R.pose3(poses[0]),R.sig(1e-5,1e-5)))
    if args.return_radius is not None:
        from return_region24 import return_region_factor
        graph.add(return_region_factor(X(0),X(len(t)-1),args.return_radius,args.return_sigma))
    model=gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(2.),gtsam.noiseModel.Isotropic.Sigma(2,1.0))
    for i,c,j,xy in zip(observation_pose_indices,z['camera_idx'],z['point_idx'],z['pixels']):
        graph.add(gtsam.GenericProjectionFactorCal3Fisheye(xy,model,X(int(i)),L(int(j)),cal[c],False,False,R.pose3(extr[c])))
    # Known-calibration projection agreement test with OpenCV before optimizing.
    for n in np.linspace(0,len(z['pixels'])-1,50,dtype=int):
        i,c,j=observation_pose_indices[n],z['camera_idx'][n],z['point_idx'][n]
        pc=R.pose3(poses[i]@extr[c]);point=z['points'][j]
        got=gtsam.PinholeCameraCal3Fisheye(pc,cal[c]).project(point)
        Y=np.linalg.inv(poses[i]@extr[c])@np.r_[point,1.]
        expected,_=cv2.fisheye.projectPoints(Y[:3].reshape(1,1,3),np.zeros(3),np.zeros(3),*kd[c])
        np.testing.assert_allclose(got,expected.ravel(),atol=1e-6)
    imu=fe['imu'];ti=imu[:,0]*1e-9
    _,covered,cov=covered_gyro_intervals(imu,np.rint(t*1e9).astype(np.int64),B.calib['calib_gyro_bias'][:3])
    bias=np.array(cov['bias']);params=gtsam.PreintegratedRotationParams();params.setGyroscopeCovariance(np.eye(3)*.008**2)
    gyro_count=0
    for i in range(len(t)-1):
        rel=np.linalg.inv(poses[i])@poses[i+1]
        # This is an explicit weak v1 motion prior, not an independent observation.
        graph.add(gtsam.BetweenFactorPose3(X(i),X(i+1),R.pose3(rel),R.sig(.3,.05+.03*np.linalg.norm(rel[:3,3]))))
        if not covered[i]:continue
        mid=ti[(ti>t[i])&(ti<t[i+1])];edges=np.r_[t[i],mid,t[i+1]]
        w=np.stack([np.interp(edges,ti,imu[:,d]) for d in [4,5,6]],axis=1)
        pim=gtsam.PreintegratedAhrsMeasurements(params,bias)
        for rate,dt in zip(.5*(w[1:]+w[:-1]),np.diff(edges)):pim.integrateMeasurement(rate,float(dt))
        T=np.eye(4);T[:3,:3]=pim.deltaRij().matrix()
        graph.add(gtsam.BetweenFactorPose3(X(i),X(i+1),R.pose3(T),R.sig(.008*np.sqrt(t[i+1]-t[i]),1e3)))
        gyro_count+=1
    held=B.handling_mask(imu,np.rint(t*1e9).astype(np.int64));att=0
    for i,ts in enumerate(t):
        if held[i]:continue
        sel=(ti>ts-1)&(ti<ts+1)
        if sel.sum()<50:continue
        g=(imu[sel,1:4]-np.array(B.calib['calib_accel_bias'][:3])).mean(0)
        graph.add(R.attitude_factor(X(i),g,np.radians(1.5)));att+=1
    loop_count=0
    if args.use_verified_loops:
        q=json.load(open(Path('results/reference_v1/quality')/f'{tag}.json'))
        cfg=q['config'];base=np.load(Path('results/fe5')/f"{tag}_loops{cfg['be_suffix']}.npz")
        extra=np.load(Path('results/fe5')/f"{tag}_extra{cfg['extra_suffix']}.npz")
        noise=gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(2.),R.sig(np.radians(.5),.05))
        for pairs,TT in [(base['ij'][base['active']],base['T'][base['active']]),(extra['ij'],extra['T'])]:
            for (i,j),T in zip(pairs,TT):
                graph.add(gtsam.BetweenFactorPose3(X(int(i)),X(int(j)),R.pose3(T),noise));loop_count+=1
    optp=gtsam.LevenbergMarquardtParams();optp.setOrdering(order);optp.setMaxIterations(args.iterations)
    optp.setRelativeErrorTol(1e-6);optp.setVerbosityLM('SUMMARY')
    initial=graph.error(values);opt=gtsam.LevenbergMarquardtOptimizer(graph,values,optp)
    result=opt.optimize();afterposes=np.array([result.atPose3(X(i)).matrix() for i in range(len(t))])
    after=validate(afterposes[selected],tracks,extr,kd)
    rows=np.c_[t,afterposes[:,:3,3],Rotation.from_matrix(afterposes[:,:3,:3]).as_quat()]
    np.savetxt(out/'imu_tum.txt',rows,fmt='%.9f')
    np.savez(out/'optimized.npz',poses=afterposes,points=np.array([result.atPoint3(L(i)) for i in range(len(z['points']))]))
    errs=[];bad=0
    for i,c,j,xy in zip(observation_pose_indices,z['camera_idx'],z['point_idx'],z['pixels']):
        pc=R.pose3(afterposes[i]@extr[c]);pt=result.atPoint3(L(int(j)))
        if pc.transformTo(pt)[2]<=0:bad+=1;continue
        errs.append(float(np.linalg.norm(gtsam.PinholeCameraCal3Fisheye(pc,cal[c]).project(pt)-xy)))
    stats=dict(tag=tag,frames=len(t),points=len(z['points']),observations=len(errs),behind_camera=bad,
        train_median_px=float(np.median(errs)),train_p90_px=float(np.quantile(errs,.9)),
        holdout_before=before,holdout_after=after,gyro_factors=gyro_count,gravity_factors=att,gyro_coverage=cov,
        verified_loop_factors=loop_count,
        approximate_return_radius_m=args.return_radius,return_sigma_m=args.return_sigma if args.return_radius is not None else None,
        iterations=opt.iterations(),iteration_limit=args.iterations,error_initial=initial,error_final=graph.error(result),
        endpoint_before_m=float(np.linalg.norm(pp[-1]-pp[0])),endpoint_after_m=float(np.linalg.norm(rows[-1,1:4]-rows[0,1:4])),
        path_ratio=float(np.linalg.norm(np.diff(rows[:,1:4],axis=0),axis=1).sum()/np.linalg.norm(np.diff(pp,axis=0),axis=1).sum()),
        runtime_s=time.monotonic()-started,input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Whole-track holdout; v1 motion prior; fixed stereo calibration; no forced endpoint equality. Optional return region is an explicit user-provided physical prior, so endpoint improvement is not independent evidence. Not ground truth.')
    (out/'result.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps({k:v for k,v in stats.items() if k not in ('holdout_before','holdout_after')}),flush=True)

if __name__=='__main__':main()
