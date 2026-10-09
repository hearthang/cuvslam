"""GT-free check of the stereo front end's yaw drift against the gyro, for stereo self-calibration.

Visual yaw on this rig drifts while driving straight (a tiny left/right camera intrinsic mismatch biases
depth differently on the two image sides, so forward motion reads as rotation). Over 1 s windows the gyro is
a good local reference once its bias is known; the bias is taken from windows where the robot stands still.
Reports the visual-minus-gyro yaw rate on straight windows [deg/1000 s]; the calibration that zeroes it is
the self-calibrated one. Uses the stereo tracker poses (pose_st) of run_frontend.py outputs.

Usage: python selfcal_yaw.py DIR [DIR ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

calib = json.load(open("/mnt/f/data_set/slam/calibration_front_small.json"))["value0"]
GYRO_BIAS = np.array(calib["calib_gyro_bias"][:3])


def straight_drift(f, win_s=1.0):
    z = np.load(f)
    t = z["t"] * 1e-9
    ps = z["pose_st"]
    ok = ~np.isnan(ps[:, 0])
    t, ps = t[ok], ps[ok]
    yv = np.unwrap(Rotation.from_quat(ps[:, 3:7]).as_euler("zyx")[:, 0])
    imu = z["imu"]
    ti = imu[:, 0] * 1e-9
    # angular rate about gravity (accelerometer low-passed over +-2 s)
    acc = imu[:, 1:4]
    cs = np.vstack([np.zeros(3), np.cumsum(acc, 0)])
    n, k = len(acc), 800
    g = cs[np.clip(np.arange(n) + k, 0, n)] - cs[np.clip(np.arange(n) - k, 0, n)]
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    wz = np.sum((imu[:, 4:7] - GYRO_BIAS) * g, 1)
    cg = np.concatenate([[0], np.cumsum(wz[1:] * np.diff(ti))])
    edges = np.arange(t[0], t[-1], win_s)
    yv_e, cg_e = np.interp(edges, t, yv), np.interp(edges, ti, cg)
    pos = np.c_[[np.interp(edges, t, ps[:, c]) for c in range(3)]].T
    dv, dg = np.diff(yv_e), np.diff(cg_e)
    dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    static = (dist < 0.005) & (np.abs(dv) < np.radians(0.1))
    b = np.median(dg[static]) if static.sum() >= 5 else 0.0  # residual gyro bias [rad / window]
    straight = (dist > 0.1) & (np.abs(dg - b) < np.radians(1.0) * win_s)
    e = dv[straight] - (dg[straight] - b)
    return np.degrees(e.sum()) / (straight.sum() * win_s) * 1000, int(static.sum()), int(straight.sum())


if __name__ == "__main__":
    for d in sys.argv[1:]:
        r = [straight_drift(f) for f in sorted(Path(d).glob("*_fe.npz"))]
        rates = np.array([x[0] for x in r])
        print(f"{d:40s} straight-line visual yaw drift vs gyro: median {np.median(rates):+7.2f} deg/1000s "
              f"(per bag {np.round(rates, 1).tolist()}, static windows {[x[1] for x in r]})")
