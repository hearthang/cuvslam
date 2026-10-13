"""Offline global optimisation of the cuSLAM trajectory at the full front-end rate (reference trajectory).

backend.py solves a 4-DoF pose graph on keyframes and export_slam_traj.py spreads the keyframe corrections over
the frames in between. Here every front-end frame (16 Hz) is a 6-DoF pose in a global
problem; a sparse conditional position solve then regularizes acceleration while
retaining measured odometry and loop constraints. Later data can correct earlier poses:
  - visual odometry: relative pose of consecutive frames (cuVSLAM stereo+IMU tracker, backend fe_in chain
    without the hand-handling bridge), sigma growing with the step
  - gyro: relative rotation of consecutive frames (bias from the parked start)
  - while the mower is off the ground (backend.handling_mask): gyro rotation, no visual factor, position held
    softly per step and lift -> put-down position ~equal (it is put back where it was picked up)
  - gravity: roll/pitch from the low-passed accelerometer every 0.5 s while on the ground
  - loops: 6-DoF relative poses of keyframe pairs verified by two-way PnP (backend.py, its GNC inlier set),
    Cauchy kernel
Initial values: backend keyframe solution with interpolated corrections. Solved with Levenberg-Marquardt.

Writes FE_DIR/<tag>_ref<suffix>.txt (timestamp tx ty tz qx qy qz qw, imu_link in a gravity-aligned world) and
<tag>_ref<suffix>.json.
Usage: python refine.py [FE_DIR=results/fe5] [--be-suffix _v11] [--suffix _v11] [--jobs 7] [--only TAG]
"""
import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import gtsam
import numpy as np
from gtsam.symbol_shorthand import B as BK, V, X
from scipy.spatial.transform import Rotation, Slerp

ap = argparse.ArgumentParser()
ap.add_argument("fe_dir", nargs="?", default="results/fe5")
ap.add_argument("--be-suffix", default="_v11", help="backend.py output used for loops and initial values")
ap.add_argument("--suffix", default="_v11")
ap.add_argument("--jobs", type=int, default=7)
ap.add_argument("--calib", default="/mnt/f/data_set/slam/calibration_front_small.json")
ap.add_argument("--cam1-fscale", type=float, default=-0.00038)
ap.add_argument("--only", default="")
ap.add_argument("--vo-t", type=float, default=0.01, help="visual step translation sigma [m] ...")
ap.add_argument("--vo-t-frac", type=float, default=0.02, help="... plus this fraction of the step")
ap.add_argument("--vo-r-deg", type=float, default=0.15, help="visual step rotation sigma [deg] ...")
ap.add_argument("--vo-r-per-m", type=float, default=0.3, help="... plus this many deg per m of step")
ap.add_argument("--gyro-noise", type=float, default=0.008, help="rad/s/sqrt(Hz) (mower vibration)")
ap.add_argument("--hold-t", type=float, default=0.01, help="per-step position sigma while handled [m]")
ap.add_argument("--hold-end-t", type=float, default=0.05, help="lift -> put-down position sigma [m]")
ap.add_argument("--att-deg", type=float, default=1.0, help="gravity (roll/pitch) sigma [deg]")
ap.add_argument("--att-every-s", type=float, default=0.5)
ap.add_argument("--att-win-s", type=float, default=1.0, help="accelerometer averaging half-window [s]")
ap.add_argument("--loop-t", type=float, default=0.05)
ap.add_argument("--loop-r-deg", type=float, default=0.5)
ap.add_argument("--all-loops", action="store_true", help="also use loops the backend's GNC rejected")
ap.add_argument("--vo-huber", type=float, default=0.0, help="Huber scale on visual steps in sigmas; 0 = none")
ap.add_argument("--imu", action="store_true", help="IMU preintegration factors (velocity states, piecewise bias)")
ap.add_argument("--acc-noise", type=float, default=0.06, help="m/s^2/sqrt(Hz) (mower vibration, not the datasheet)")
ap.add_argument("--acc-bias-rw", type=float, default=1e-3, help="m/s^2 per sqrt(s)")
ap.add_argument("--gyro-bias-rw", type=float, default=2e-5, help="rad/s per sqrt(s)")
ap.add_argument("--bias-every-s", type=float, default=10.0)
ap.add_argument("--gyro-scale-z", type=float, default=0.0085, help="gyro z scale-factor error (see run_frontend.py)")
ap.add_argument("--gnc", type=float, nargs="+", default=[10, 2],
                help="graduated Cauchy scales: retain verified loops while bringing the full graph into agreement")
ap.add_argument("--smooth-acc", type=float, default=0.0,
                help="robust acceleration prior sigma in m/s^2; zero disables it")
ap.add_argument("--extra-suffix", default="", help="optional verified extra loop file suffix")
ap.add_argument("--position-acc", type=float, default=1.0,
                help="sparse conditional position refinement acceleration sigma, m/s^2; zero disables")
cli = None
import backend as B  # noqa: E402


def configure(config):
    global cli
    cli = argparse.Namespace(**config)
    B.configure(cli.calib, cli.cam1_fscale)


def pose3(M):
    return gtsam.Pose3(gtsam.Rot3(M[:3, :3]), gtsam.Point3(*M[:3, 3]))


def sig(rot_rad, trans_m):
    return gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.broadcast_to(rot_rad, 3), np.broadcast_to(trans_m, 3)])


def attitude_factor(key, up_body, sigma):
    """Gravity: world up (0, 0, 1) seen in the body frame should be up_body (unit vector, from the accelerometer).
    2-D residual in the tangent plane of up_body."""
    up_body = up_body / np.linalg.norm(up_body)
    a = np.cross(up_body, [1.0, 0, 0] if abs(up_body[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    Bt = np.stack([a, np.cross(up_body, a)])  # 2x3
    model = gtsam.noiseModel.Isotropic.Sigma(2, sigma)

    def err(this, values, H):
        R = values.atPose3(this.keys()[0]).rotation().matrix()
        v = R.T @ np.array([0.0, 0.0, 1.0])
        if H is not None:  # R <- R Exp(d): v <- v + skew(v) d
            sk = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            H[0] = np.hstack([Bt @ sk, np.zeros((2, 3))])
        return Bt @ v
    return gtsam.CustomFactor(model, [key], err)


def initial_values(z, t, kf_idx, opt_rows):
    """Front-end chain (with the handling bridge) corrected by the backend's keyframe solution, the corrections
    interpolated between keyframes (as export_slam_traj.py, without jump limits)."""
    P = B.frontend_odometry(z, "in")
    C = [B.mat(opt_rows[k, 1:8]) @ np.linalg.inv(P[kf_idx[k]]) for k in range(len(kf_idx))]
    tk = t[kf_idx].astype(float)
    tq = np.clip(t.astype(float), tk[0], tk[-1])
    Cp = np.stack([np.interp(tq, tk, [c[d, 3] for c in C]) for d in range(3)], 1)
    Cr = Slerp(tk, Rotation.from_matrix([c[:3, :3] for c in C]))(tq).as_matrix()
    out = np.empty_like(P)
    for i in range(len(t)):
        Ci = np.eye(4)
        Ci[:3, :3], Ci[:3, 3] = Cr[i], Cp[i]
        out[i] = Ci @ P[i]
    return out


def acceleration_factor(keys, dt0, dt1, sigma):
    """Weak world acceleration prior, with exact right-local Pose3 Jacobians.

    Huber tails allow genuine abrupt motion; this regularizes translation inside
    the graph instead of clamping or replacing the exported trajectory.
    """
    weights = np.array([1 / dt0, -1 / dt0 - 1 / dt1, 1 / dt1]) * (2 / (dt0 + dt1))
    model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(2),
                                         gtsam.noiseModel.Isotropic.Sigma(3, sigma))

    def error(this, values, H):
        poses = [values.atPose3(k) for k in keys]
        if H is not None:
            for j, (w, p) in enumerate(zip(weights, poses)):
                H[j] = np.hstack([np.zeros((3, 3)), w * p.rotation().matrix()])
        return sum(w * p.translation() for w, p in zip(weights, poses))

    return gtsam.CustomFactor(model, keys, error)


def add_imu(graph, vals, t, imu, P0):
    """ImuFactor per frame step (400 Hz preintegrated), velocity states, bias constant over bias_every_s
    with a random walk between segments; bias prior from calibration (acc) and the parked start (gyro)."""
    ti = imu[:, 0].astype(np.int64)
    acc, gyr = imu[:, 1:4], imu[:, 4:7] / np.array([1, 1, 1 + cli.gyro_scale_z])
    st = ti < ti[0] + 1.5e9
    acc_bias0 = np.array(B.calib["calib_accel_bias"][:3])
    gyro_bias0 = gyr[st].mean(0) if acc[st].std(0).max() < 0.15 else np.array(B.calib["calib_gyro_bias"][:3])
    p = gtsam.PreintegrationParams.MakeSharedU(9.81)
    p.setAccelerometerCovariance(np.eye(3) * cli.acc_noise ** 2)
    p.setGyroscopeCovariance(np.eye(3) * cli.gyro_noise ** 2)
    p.setIntegrationCovariance(np.eye(3) * 1e-8)
    bias0 = gtsam.imuBias.ConstantBias(acc_bias0, gyro_bias0)
    n = len(t)
    seg = ((t - t[0]) * 1e-9 // cli.bias_every_s).astype(int)
    for b in range(seg[-1] + 1):
        vals.insert(BK(b), bias0)
        if b == 0:
            graph.add(gtsam.PriorFactorConstantBias(BK(0), bias0, gtsam.noiseModel.Diagonal.Sigmas(
                np.r_[[0.05] * 3, [np.radians(0.02)] * 3])))
        else:
            graph.add(gtsam.BetweenFactorConstantBias(BK(b - 1), BK(b), gtsam.imuBias.ConstantBias(),
                                                      gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(cli.bias_every_s) * np.r_[
                                                          [cli.acc_bias_rw] * 3, [cli.gyro_bias_rw] * 3])))
    tf = t.astype(float) * 1e-9
    vel = np.gradient(P0[:, :3, 3], tf, axis=0)
    for i in range(n):
        vals.insert(V(i), vel[i])
    graph.add(gtsam.PriorFactorVector(V(0), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, 0.02)))
    j = np.searchsorted(ti, t[0], side="right")
    for k in range(n - 1):
        pim = gtsam.PreintegratedImuMeasurements(p, bias0)
        tprev = t[k]
        while j < len(ti) and ti[j] <= t[k + 1]:
            if ti[j] > tprev:
                pim.integrateMeasurement(acc[j], gyr[j], (ti[j] - tprev) * 1e-9)
            tprev = ti[j]
            j += 1
        if t[k + 1] > tprev and j > 0:
            pim.integrateMeasurement(acc[j - 1], gyr[j - 1], (t[k + 1] - tprev) * 1e-9)
        if pim.deltaTij() > 0:
            graph.add(gtsam.ImuFactor(X(k), V(k), X(k + 1), V(k + 1), BK(seg[k]), pim))


def run(tag):
    fe = Path(cli.fe_dir)
    z = np.load(fe / f"{tag}_fe.npz")
    t = z["t"].astype(np.int64)
    imu = z["imu"]
    kf_idx = z["kf_idx"]
    n = len(t)
    if n < 3 or np.any(np.diff(t) <= 0):
        raise ValueError(f'{tag}: need at least three frames with strictly increasing timestamps')
    if len(kf_idx) < 2 or kf_idx[0] != 0 or kf_idx[-1] != n-1 or np.any(np.diff(kf_idx) <= 0):
        raise ValueError(f'{tag}: keyframes must be ordered and cover the first and last frame')
    if imu.ndim != 2 or imu.shape[1] != 7 or not np.isfinite(imu).all() or np.any(np.diff(imu[:, 0]) <= 0):
        raise ValueError(f'{tag}: invalid IMU stream')
    opt_rows = np.loadtxt(fe / f"{tag}_be{cli.be_suffix}.txt")
    if len(opt_rows) != len(kf_idx) or not np.allclose(opt_rows[:, 0], t[kf_idx]*1e-9, atol=2e-6, rtol=0):
        raise ValueError(f'{tag}: backend timestamps do not correspond to frontend keyframes')
    L = np.load(fe / f"{tag}_loops{cli.be_suffix}.npz")
    extra_path = fe / f"{tag}_extra{cli.extra_suffix}.npz"
    extra = np.load(extra_path) if cli.extra_suffix else None
    P_vo = B.build_odometry(z["pose_in"], z["pose_st"], "in", t_ns=t)[0]
    held = B.handling_mask(imu, t)
    grv = B.gyro_rotvecs(imu, t)
    P0 = initial_values(z, t, kf_idx, opt_rows)

    graph, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for i in range(n):
        vals.insert(X(i), pose3(P0[i]))
    graph.add(gtsam.PriorFactorPose3(X(0), pose3(P0[0]), sig([np.radians(1.0), np.radians(1.0), 1e-6], 1e-6)))

    huber = gtsam.noiseModel.mEstimator.Huber.Create(cli.vo_huber) if cli.vo_huber > 0 else None
    free_t = 1e3
    n_vo = n_hold = 0
    for k in range(n - 1):
        dt = (t[k + 1] - t[k]) * 1e-9
        g_sig = cli.gyro_noise * np.sqrt(max(dt, 1e-3))
        Rg = Rotation.from_rotvec(grv[k]).as_matrix()
        if held[k] or held[k + 1]:
            if cli.imu:  # the IMU carries the motion while handled (the hand lifts it, so no per-step hold)
                n_hold += 1
                continue
            M = np.eye(4)
            M[:3, :3] = Rg
            graph.add(gtsam.BetweenFactorPose3(X(k), X(k + 1), pose3(M), sig(g_sig, cli.hold_t)))
            n_hold += 1
            continue
        rel = np.linalg.inv(P_vo[k]) @ P_vo[k + 1]
        d = np.linalg.norm(rel[:3, 3])
        noise = sig(np.radians(cli.vo_r_deg + cli.vo_r_per_m * d), cli.vo_t + cli.vo_t_frac * d)
        if huber is not None:
            noise = gtsam.noiseModel.Robust.Create(huber, noise)
        graph.add(gtsam.BetweenFactorPose3(X(k), X(k + 1), pose3(rel), noise))
        if not cli.imu:
            M = np.eye(4)
            M[:3, :3] = Rg
            graph.add(gtsam.BetweenFactorPose3(X(k), X(k + 1), pose3(M), sig(g_sig, free_t)))
        n_vo += 1

    if cli.imu:
        add_imu(graph, vals, t, imu, P0)

    if cli.smooth_acc > 0:
        dt = np.diff(t) * 1e-9
        for k in range(1, n - 1):
            if not held[k - 1:k + 2].any() and max(dt[k - 1:k + 1]) < 0.25:
                graph.add(acceleration_factor([X(k - 1), X(k), X(k + 1)],
                                              dt[k - 1], dt[k], cli.smooth_acc))

    # handling events: put back (almost) where it was picked up
    edges = np.flatnonzero(np.diff(held.astype(int)))
    starts = [e for e in edges if held[e + 1]]
    for a in starts:
        b = next((e + 1 for e in edges if e > a and not held[e + 1]), None)
        if b is not None:
            graph.add(gtsam.BetweenFactorPose3(X(a), X(b), gtsam.Pose3(),
                                               sig(10.0, cli.hold_end_t)))  # rotation free

    # gravity (roll/pitch), on the ground only, >= 1 s away from handling
    acc_bias = np.array(B.calib["calib_accel_bias"][:3])
    ti = imu[:, 0]
    cs = np.vstack([np.zeros(3), np.cumsum(imu[:, 1:4] - acc_bias, axis=0)])
    near_held = np.convolve(held.astype(float), np.ones(33), "same") > 0
    every = max(1, int(round(cli.att_every_s / (np.median(np.diff(t)) * 1e-9))))
    n_att = 0
    for i in range(0, n, every):
        if near_held[i]:
            continue
        a, b = np.searchsorted(ti, [t[i] - cli.att_win_s * 1e9, t[i] + cli.att_win_s * 1e9])
        if b - a < 10:
            continue
        graph.add(attitude_factor(X(i), (cs[b] - cs[a]) / (b - a), np.radians(cli.att_deg)))
        n_att += 1

    # A tight robust kernel from the first iteration can discard genuine loops:
    # their initial residual includes drift and inconsistent gravity alignment.
    # Start with a broad kernel and tighten only after global corrections settle.
    base_graph = graph
    use = np.ones(len(L["ij"]), bool) if cli.all_loops else L["active"].astype(bool)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setRelativeErrorTol(1e-6)
    res = vals
    stages = []
    for scale in cli.gnc:
        graph = gtsam.NonlinearFactorGraph(base_graph)
        loop_noise = sig(np.radians(cli.loop_r_deg), cli.loop_t)
        if scale > 0:
            loop_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(scale), loop_noise)
        for (i, j), T in zip(L["ij"][use], L["T"][use]):
            graph.add(gtsam.BetweenFactorPose3(X(int(kf_idx[i])), X(int(kf_idx[j])), pose3(T), loop_noise))
        if extra is not None:
            for (i, j), T in zip(extra["ij"], extra["T"]):
                graph.add(gtsam.BetweenFactorPose3(X(int(kf_idx[i])), X(int(kf_idx[j])), pose3(T), loop_noise))
        opt = gtsam.LevenbergMarquardtOptimizer(graph, res, params)
        res = opt.optimize()
        stages.append({"scale": scale, "iterations": int(opt.iterations()), "error": float(graph.error(res)),
                       "closure": float(np.linalg.norm(res.atPose3(X(n-1)).translation() - res.atPose3(X(0)).translation()))})
        print(f"{tag}: scale {scale:g}, closure {stages[-1]['closure']:.3f}, it {opt.iterations()}", flush=True)
    Pout = np.array([res.atPose3(X(i)).matrix() for i in range(n)])
    if not np.isfinite(Pout).all():
        raise ValueError(f'{tag}: nonfinite optimized poses')

    position_history = []
    if cli.position_acc > 0:
        from translation_refine import solve_positions
        increments = np.array([(np.linalg.inv(P_vo[k]) @ P_vo[k+1])[:3, 3] for k in range(n-1)])
        step_sig = cli.vo_t + cli.vo_t_frac * np.linalg.norm(increments, axis=1)
        mask = held[:-1] | held[1:]
        increments[mask] = 0
        step_sig[mask] = cli.hold_t
        pairs, transforms = L['ij'][use], L['T'][use]
        if extra is not None:
            pairs = np.concatenate([pairs, extra['ij']])
            transforms = np.concatenate([transforms, extra['T']])
        Pout[:, :3, 3], position_history = solve_positions(
            (t-t[0])*1e-9, Pout[:, :3, 3], Pout[:, :3, :3], increments, step_sig,
            kf_idx[pairs], transforms[:, :3, 3], loop_sigma=cli.loop_t,
            acceleration_sigma=cli.position_acc, loop_scale=cli.gnc[-1])

    rows = np.array([[ts * 1e-9, *M[:3, 3], *Rotation.from_matrix(M[:3, :3]).as_quat()] for ts, M in zip(t, Pout)])
    destination = fe / f"{tag}_ref{cli.suffix}.txt"
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    with open(temporary, "w") as fh:
        for r in rows:
            fh.write(f"{r[0]:.9f} " + " ".join(f"{v:.6f}" for v in r[1:]) + "\n")
    temporary.replace(destination)
    starts = np.flatnonzero(np.diff(np.r_[False, held, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, held, False].astype(int)) == -1) - 1
    stats = {"tag": tag, "frames": int(n), "vo_steps": n_vo, "held_steps": n_hold, "attitude": n_att,
             "loops": int(use.sum()), "iterations": int(opt.iterations()),
             "error_init": float(graph.error(vals)), "error_final": float(graph.error(res)),
             "closure_init_m": float(np.linalg.norm(P0[-1][:3, 3] - P0[0][:3, 3])),
             "closure_ref_m": float(np.linalg.norm(Pout[-1][:3, 3] - Pout[0][:3, 3])),
             "config": vars(cli), "stages": stages,
             "extra_loops": 0 if extra is None else len(extra['ij']),
             "position_iterations": position_history,
             "handling_intervals_s": [[float(t[a]*1e-9), float(t[b]*1e-9)] for a,b in zip(starts, ends)],
             "source_sha256": {p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                                for p in ['reference.py', 'translation_refine.py', 'backend.py']}}
    destination = fe / f"{tag}_ref{cli.suffix}.json"
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(stats, indent=2))
    temporary.replace(destination)
    return stats


if __name__ == "__main__":
    configure(vars(ap.parse_args()))
    fe = Path(cli.fe_dir)
    tags = sorted(p.name[:-7] for p in fe.glob("*_fe.npz") if (fe / f"{p.name[:-7]}_loops{cli.be_suffix}.npz").exists())
    if cli.only:
        tags = [t for t in tags if cli.only in t]
    with Pool(cli.jobs, initializer=configure, initargs=(vars(cli),)) as pool:
        for s in pool.imap_unordered(run, tags):
            print(f"{s['tag']}: closure init {s['closure_init_m']:.3f} -> ref {s['closure_ref_m']:.3f} m, "
                  f"loops {s['loops']}, it {s['iterations']}, err {s['error_init']:.0f} -> {s['error_final']:.0f}", flush=True)
