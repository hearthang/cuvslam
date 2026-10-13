"""Gyro interval coverage: missing samples must not become zero-rotation evidence."""
import numpy as np

def covered_gyro_intervals(imu, frame_ns, calibration_bias, max_gap_s=.015,gyro_scale_z=0.0):
    imu=np.asarray(imu,float);t=np.asarray(frame_ns,np.int64)
    if imu.ndim!=2 or imu.shape[1]!=7 or len(imu)<2:
        raise ValueError('Need at least two IMU measurements')
    ti=imu[:,0]
    if np.any(np.diff(ti)<=0) or np.any(np.diff(t)<=0):
        raise ValueError('Timestamps must increase')
    if not np.isfinite(gyro_scale_z) or gyro_scale_z<=-1:raise ValueError('Invalid gyro scale calibration')
    scale=np.array([1.,1.,1.+gyro_scale_z]);rates=imu[:,4:7]/scale
    # Default0 retains v1 exactly; scale calibration is a separate experiment.
    static=ti<ti[0]+1.5e9
    bias=rates[static].mean(0) if imu[static,1:4].std(0).max()<.15 else np.asarray(calibration_bias)/scale
    cum=np.vstack([np.zeros(3),np.cumsum((rates[1:]-bias)*(np.diff(ti)*1e-9)[:,None],axis=0)])
    c=np.stack([np.interp(t,ti,cum[:,d]) for d in range(3)],axis=1)
    covered=(t[:-1]>=ti[0]) & (t[1:]<=ti[-1])
    gap=np.diff(ti)*1e-9>max_gap_s
    for a,b in zip(ti[:-1][gap],ti[1:][gap]):
        # An interval touching the edge but not the interior is not affected.
        covered &= ~((t[:-1]<b)&(t[1:]>a))
    return np.diff(c,axis=0),covered,dict(bias=bias.tolist(),max_gap_s=max_gap_s,applied_gyro_scale_z=gyro_scale_z,
        uncovered_steps=int((~covered).sum()),long_sensor_gaps=int(gap.sum()),
        absent_start_s=float(max(0,ti[0]-t[0])*1e-9),absent_end_s=float(max(0,t[-1]-ti[-1])*1e-9))
