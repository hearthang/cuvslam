"""Loosely coupled VIO on top of the cuVSLAM front end (run_frontend.py output), with GTSAM.

cuVSLAM's own IMU fusion only adds robustness, so its tilt and velocity are not held by gravity and the
accelerometer the way a real VIO (Basalt) does. Here, per main front-end frame (~5.3 Hz):
  states   pose X_k (imu_link in a z-up world), velocity V_k, IMU bias B_k
  factors  - CombinedImuFactor: preintegrated 400 Hz IMU between frames, bias random walk
           - visual BetweenFactorPose3 from the de-glitched front-end odometry (backend.vote_odometry), Huber
           - zero-velocity priors while the mower stands still (vision and gyro both quiet)
           - optional non-holonomic prior: body-frame lateral / vertical velocity ~ 0 (ground robot)
           - prior: gravity-levelled start (accelerometer while parked), V0 = 0, gyro bias from the static start
Solved incrementally with ISAM2; the *online* estimate of each frame (no future data, like Basalt's VIO output)
is written to {tag}_vio.txt (steps held to the physical limit causally, traj_limit.py) and the final smoothed
one to {tag}_vio_smooth.txt (TUM, imu_link).

Usage: python vio_fusion.py [FE_DIR=results/fe3] [--jobs 14] [--only TAG] [--suffix S] [param overrides]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("fe_dir", nargs="?", default="results/fe3")
ap.add_argument("--jobs", type=int, default=14)
ap.add_argument("--only", default="")
ap.add_argument("--suffix", default="")
ap.add_argument("--acc-noise", type=float, default=0.06, help="m/s^2/sqrt(Hz); mower vibration, not the datasheet")
ap.add_argument("--gyro-noise", type=float, default=0.008, help="rad/s/sqrt(Hz)")
ap.add_argument("--acc-bias-rw", type=float, default=1e-3)
ap.add_argument("--gyro-bias-rw", type=float, default=2e-5)
ap.add_argument("--gyro-scale-z", type=float, default=0.0085, help="gyro z scale-factor error (see run_frontend.py)")
ap.add_argument("--vis-rot-deg", type=float, default=0.15, help="visual relative rotation sigma per frame")
ap.add_argument("--vis-trans", type=float, default=0.01, help="visual relative translation sigma per frame [m]")
ap.add_argument("--vis-trans-frac", type=float, default=0.03, help="... plus this fraction of the step")
ap.add_argument("--zupt-sigma", type=float, default=0.01, help="m/s; 0 disables zero-velocity priors")
ap.add_argument("--nhc-sigma", type=float, default=0.0, help="m/s lateral/vertical body velocity; 0 disables")
ap.add_argument("--odom", default="vote", help="visual odometry from backend: vote | in | st | guard")
cli = ap.parse_args()

sys.argv = sys.argv[:1]
import backend  # noqa: E402  (backend parses its own argv at import)
from traj_limit import limit_causal  # noqa: E402

calib = json.load(open("/mnt/f/data_set/slam/calibration_front_small.json"))["value0"]
G = 9.81


def pose3(M):
    return gtsam.Pose3(gtsam.Rot3(M[:3, :3]), gtsam.Point3(*M[:3, 3]))


def nhc_factor(xk, vk, sigma):
    """Body-frame lateral (y) and vertical (z) velocity ~ 0."""
    model = gtsam.noiseModel.Isotropic.Sigma(2, sigma)

    def err(this, values, H):
        pose, v = values.atPose3(this.keys()[0]), values.atVector(this.keys()[1])
        R = pose.rotation().matrix()
        vb = R.T @ v
        if H is not None:
            # d(R^T v)/d(rotation perturbation) = skew(R^T v); d/d v = R^T
            sk = np.array([[0, -vb[2], vb[1]], [vb[2], 0, -vb[0]], [-vb[1], vb[0], 0]])
            H[0] = np.hstack([sk[1:3], np.zeros((2, 3))])
            H[1] = R.T[1:3]
        return vb[1:3]
    return gtsam.CustomFactor(model, [xk, vk], err)


def run(tag):
    fe = Path(cli.fe_dir)
    z = np.load(fe / f"{tag}_fe.npz")
    t = z["t"].astype(np.int64)
    if cli.odom == "vote":
        P = backend.vote_odometry(z)
    else:
        P = backend.build_odometry(z["pose_in"], z["pose_st"], cli.odom, t_ns=z["t"])[0]
    imu = z["imu"]
    ti = imu[:, 0].astype(np.int64)
    acc, gyr = imu[:, 1:4], imu[:, 4:7] / np.array([1, 1, 1 + cli.gyro_scale_z])

    # static start: gravity direction (levelled start), gyro bias
    st = ti < ti[0] + 1.5e9
    acc_bias0 = np.array(calib["calib_accel_bias"][:3])
    g_b = (acc[st] - acc_bias0).mean(0)
    R0, _ = Rotation.align_vectors([[0, 0, 1]], [g_b / np.linalg.norm(g_b)])  # body -> level world
    gyro_bias0 = gyr[st].mean(0) if acc[st].std(0).max() < 0.15 else np.array(calib["calib_gyro_bias"][:3])

    p = gtsam.PreintegrationCombinedParams.MakeSharedU(G)
    p.setAccelerometerCovariance(np.eye(3) * cli.acc_noise ** 2)
    p.setGyroscopeCovariance(np.eye(3) * cli.gyro_noise ** 2)
    p.setIntegrationCovariance(np.eye(3) * 1e-8)
    p.setBiasAccCovariance(np.eye(3) * cli.acc_bias_rw ** 2)
    p.setBiasOmegaCovariance(np.eye(3) * cli.gyro_bias_rw ** 2)
    bias0 = gtsam.imuBias.ConstantBias(acc_bias0, gyro_bias0)

    # world frame: levelled start (yaw 0); visual odometry is expressed relative to its own first pose
    T0 = np.eye(4)
    T0[:3, :3] = R0.as_matrix()
    Pw = np.array([T0 @ M for M in P])

    # zero-velocity frames: vision quiet over +-0.5 s and gyro quiet
    n = len(t)
    pos = P[:, :3, 3]
    zupt = np.zeros(n, bool)
    if cli.zupt_sigma > 0:
        half = 3
        gnorm = np.linalg.norm(gyr - gyro_bias0, axis=1)
        cg = np.concatenate([[0], np.cumsum(gnorm)])
        for k in range(half, n - half):
            if np.linalg.norm(pos[k + half] - pos[k - half]) < 0.005:
                a, b = np.searchsorted(ti, [t[k - half], t[k + half]])
                if b > a and (cg[b] - cg[a]) / (b - a) < np.radians(2):
                    zupt[k] = True

    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    graph, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    graph.add(gtsam.PriorFactorPose3(X(0), pose3(Pw[0]), gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.radians(0.5)] * 2 + [1e-4] + [1e-3] * 3))))
    graph.add(gtsam.PriorFactorVector(V(0), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, 0.01)))
    graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05] * 3 + [np.radians(0.02)] * 3))))
    vals.insert(X(0), pose3(Pw[0]))
    vals.insert(V(0), np.zeros(3))
    vals.insert(B(0), bias0)
    isam.update(graph, vals)
    online = [Pw[0]]
    nav = gtsam.NavState(pose3(Pw[0]), np.zeros(3))
    bias = bias0
    huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    j = np.searchsorted(ti, t[0], side="right")
    for k in range(n - 1):
        pim = gtsam.PreintegratedCombinedMeasurements(p, bias)
        tprev = t[k]
        while j < len(ti) and ti[j] <= t[k + 1]:
            dt = (ti[j] - tprev) * 1e-9
            if dt > 0:
                pim.integrateMeasurement(acc[j], gyr[j], dt)
            tprev = ti[j]
            j += 1
        if t[k + 1] > tprev:
            pim.integrateMeasurement(acc[j - 1], gyr[j - 1], (t[k + 1] - tprev) * 1e-9)
        graph, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        graph.add(gtsam.CombinedImuFactor(X(k), V(k), X(k + 1), V(k + 1), B(k), B(k + 1), pim))
        rel = np.linalg.inv(P[k]) @ P[k + 1]
        st_t = cli.vis_trans + cli.vis_trans_frac * np.linalg.norm(rel[:3, 3])
        vis_noise = gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Diagonal.Sigmas(
            np.array([np.radians(cli.vis_rot_deg)] * 3 + [st_t] * 3)))
        graph.add(gtsam.BetweenFactorPose3(X(k), X(k + 1), pose3(rel), vis_noise))
        if zupt[k + 1]:
            graph.add(gtsam.PriorFactorVector(V(k + 1), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, cli.zupt_sigma)))
        if cli.nhc_sigma > 0:
            graph.add(nhc_factor(X(k + 1), V(k + 1), cli.nhc_sigma))
        pred = pim.predict(nav, bias)
        x_init = nav.pose().compose(pose3(rel))
        vals.insert(X(k + 1), x_init)
        vals.insert(V(k + 1), pred.velocity())
        vals.insert(B(k + 1), bias)
        isam.update(graph, vals)
        xk = isam.calculateEstimatePose3(X(k + 1))
        vk = isam.calculateEstimateVector(V(k + 1))
        bias = isam.calculateEstimateConstantBias(B(k + 1))
        nav = gtsam.NavState(xk, vk)
        online.append(xk.matrix())
    est = isam.calculateEstimate()
    smooth = [est.atPose3(X(k)).matrix() for k in range(n)]

    # causal jump suppression of the online output (traj_limit.py); the smoothed output is left as solved
    online_pos, n_clamped = limit_causal(t * 1e-9, np.array([M[:3, 3] for M in online]))
    for M, pp in zip(online, online_pos):
        M[:3, 3] = pp

    def tum(Ms):
        return np.array([[ts * 1e-9, *M[:3, 3], *Rotation.from_matrix(M[:3, :3]).as_quat()] for ts, M in zip(t, Ms)])
    suf = cli.suffix
    np.savetxt(fe / f"{tag}_vio{suf}.txt", tum(online), fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
    np.savetxt(fe / f"{tag}_vio_smooth{suf}.txt", tum(smooth), fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
    b_end = est.atConstantBias(B(n - 1))
    return {"tag": tag, "closure_online": float(np.linalg.norm(online[-1][:3, 3] - online[0][:3, 3])),
            "closure_smooth": float(np.linalg.norm(smooth[-1][:3, 3] - smooth[0][:3, 3])),
            "closure_vis": float(np.linalg.norm(P[-1][:3, 3] - P[0][:3, 3])), "zupt_frames": int(zupt.sum()),
            "clamped_steps": int(n_clamped),
            "acc_bias_end": b_end.accelerometer().tolist(), "gyro_bias_end": b_end.gyroscope().tolist()}


if __name__ == "__main__":
    tags = sorted(p.name[:-7] for p in Path(cli.fe_dir).glob("*_fe.npz"))
    if cli.only:
        tags = [t for t in tags if cli.only in t]
    with Pool(cli.jobs) as pool:
        for r in pool.imap_unordered(run, tags):
            json.dump(r, open(Path(cli.fe_dir) / f"{r['tag']}_vio{cli.suffix}.json", "w"), indent=1)
            print(f"{r['tag']}: vis {r['closure_vis']:.3f}  vio online {r['closure_online']:.3f}  "
                  f"smooth {r['closure_smooth']:.3f}  zupt {r['zupt_frames']}", flush=True)
