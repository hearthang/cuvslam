"""Loop-closure + 4-DoF pose-graph back end on top of the cuVSLAM front end (run_frontend.py output).

Per bag:
 1. odometry: cuVSLAM stereo+IMU tracker, with a per-frame divergence guard that substitutes the
    stereo-only tracker's increment wherever the inertial one disagrees (or is lost)
 2. keyframes: ORB features, stereo-triangulated 3D points (fisheye kb4 calib)
 3. loop candidates: earlier keyframes within an odometry-drift radius and similar heading
 4. geometric verification: PnP both ways (i->j and j->i) with RANSAC; both must agree
 5. 4-DoF pose graph (x, y, z, yaw; roll/pitch kept from the gravity-aware odometry, as in VINS-Mono)
    with a Cauchy kernel on loop edges and iterative rejection of inconsistent loops
Writes {tag}_be.txt (TUM, keyframe rate) and {tag}_be.json (stats).

Usage: python backend.py [FE_DIR=results/fe] [--jobs 12] [--odom guard|in|st] [--only TAG]
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("fe_dir", nargs="?", default="results/fe")
ap.add_argument("--calib", default="/mnt/f/data_set/slam/calibration_front_small.json")
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--odom", choices=["guard", "in", "st"], default="guard")
ap.add_argument("--only", default="")
ap.add_argument("--guard-t", type=float, default=0.05, help="inertial/stereo disagreement over 1 s [m] (+10%% of motion)")
ap.add_argument("--guard-r", type=float, default=2.0, help="... [deg]")
ap.add_argument("--rank", choices=["tiny", "dist"], default="tiny", help="loop candidate ranking")
ap.add_argument("--loops-per-query", type=int, default=2)
ap.add_argument("--suffix", default="", help="output name suffix, for experiments")
ap.add_argument("--min-inliers", type=int, default=30)
ap.add_argument("--min-gap-s", type=float, default=30.0, help="loop candidates must be this far apart in time")
ap.add_argument("--radius", type=float, default=1.0, help="candidate radius = radius + drift * path, capped")
ap.add_argument("--drift", type=float, default=0.05)
ap.add_argument("--radius-max", type=float, default=15.0)
ap.add_argument("--max-yaw-deg", type=float, default=60.0)
ap.add_argument("--cands", type=int, default=5, help="verified candidates tried per query keyframe")
ap.add_argument("--points", choices=["temporal", "stereo"], default="stereo",
                help="stereo: KLT stereo depth (default); temporal: triangulate against odometry neighbours")
ap.add_argument("--gravity", type=int, default=1, help="roll/pitch from IMU gravity (1) or odometry (0)")
ap.add_argument("--gnc", type=float, nargs="+", default=[40, 20, 10, 5, 2], help="Cauchy scales, in sigmas")
ap.add_argument("--reject-t", type=float, default=0.3)
ap.add_argument("--reject-yaw", type=float, default=3.0)
ap.add_argument("--no-loops", action="store_true", help="odometry only (for ablation)")
args = ap.parse_args()

MASK_BOTTOM = 90
calib = json.load(open(args.calib))["value0"]


def _T(d):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([d["qx"], d["qy"], d["qz"], d["qw"]]).as_matrix()
    M[:3, 3] = [d["px"], d["py"], d["pz"]]
    return M


def _kd(i):
    k = calib["intrinsics"][i]["intrinsics"]
    return (np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]]),
            np.array([k["k1"], k["k2"], k["k3"], k["k4"]]))


T_IC0, T_IC1 = _T(calib["T_imu_cam"][0]), _T(calib["T_imu_cam"][1])
T_C1C0 = np.linalg.inv(T_IC1) @ T_IC0
(K0, D0), (K1, D1) = _kd(0), _kd(1)
F = K0[0, 0]


def mat(row):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
    M[:3, 3] = row[:3]
    return M


def yaw_of(R):
    return np.arctan2(R[1, 0], R[0, 0])


def rz(y):
    c, s = np.cos(y), np.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------- odometry with divergence guard ----------------
MAX_SPEED, MAX_RATE_DEG = 1.5, 150.0  # real motion: <= 1.1 m/s, <= 86 deg/s


def build_odometry(pose_in, pose_st, mode, win=16, t_ns=None):
    """Compose per-frame increments. 'guard': inertial increments, except where the inertial and stereo
    trackers disagree over the surrounding `win`-frame (1 s) window by more than guard_t / guard_r
    (catches both sudden jumps and slow divergence); there the stereo increment is used."""
    n = len(pose_in)
    M_in = [mat(r) if not np.isnan(r[0]) else None for r in pose_in]
    M_st = [mat(r) if not np.isnan(r[0]) else None for r in pose_st]
    bad = np.zeros(n, bool)
    if mode == "guard":
        for k in range(win, n, 4):
            a0, a1, b0, b1 = M_in[k - win], M_in[k], M_st[k - win], M_st[k]
            if a0 is None or a1 is None:
                bad[k - win:k + 1] = True
                continue
            if b0 is None or b1 is None:
                continue
            A, Bm = np.linalg.inv(a0) @ a1, np.linalg.inv(b0) @ b1
            dt = np.linalg.norm(A[:3, 3] - Bm[:3, 3])
            dr = Rotation.from_matrix(A[:3, :3].T @ Bm[:3, :3]).magnitude()
            if dt > args.guard_t + 0.1 * np.linalg.norm(Bm[:3, 3]) or dr > np.radians(args.guard_r):
                bad[k - win:k + 1] = True
    def plausible(inc, dt):
        # the mower drives < 0.8 m/s; both trackers occasionally jump by metres in one frame
        return inc is not None and np.linalg.norm(inc[:3, 3]) < MAX_SPEED * dt and \
            Rotation.from_matrix(inc[:3, :3]).magnitude() < np.radians(MAX_RATE_DEG) * dt
    P = [np.eye(4)]
    prev, n_implausible = np.eye(4), 0
    for k in range(1, n):
        dt = max((t_ns[k] - t_ns[k - 1]) * 1e-9, 1e-3) if t_ns is not None else 1 / 16
        inc_in = np.linalg.inv(M_in[k - 1]) @ M_in[k] if M_in[k] is not None and M_in[k - 1] is not None else None
        inc_st = np.linalg.inv(M_st[k - 1]) @ M_st[k] if M_st[k] is not None and M_st[k - 1] is not None else None
        if mode == "st":
            order = [inc_st, inc_in]
        elif mode == "in":
            order = [inc_in, inc_st]
        else:
            order = [inc_st, inc_in] if bad[k] else [inc_in, inc_st]
        inc = next((c for c in order if plausible(c, dt)), None)
        if inc is None:  # both trackers glitched: constant-velocity bridge
            inc = prev
            n_implausible += 1
        prev = inc
        P.append(P[-1] @ inc)
    build_odometry.n_implausible = n_implausible
    return np.array(P), int(bad.sum())


# ---------------- keyframe features ----------------
def _norm(pts, K, D):
    return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)


def _match(d1, d2, ratio=0.8):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    return [(m.queryIdx, m.trainIdx) for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]


def kf_features(jl, jr, orb):
    """ORB on the left image; 3D points (cam0 frame) by sub-pixel KLT stereo matching left->right with a
    right->left consistency check. (ORB descriptor matching left/right was biased: integer keypoints and
    row-aligned mismatches on a scene that is mostly 10-40 m away, i.e. 1-2 px of disparity.)"""
    empty = {"un": np.zeros((0, 2), np.float32), "desc": None, "X": np.zeros((0, 3), np.float32)}
    L = cv2.imdecode(np.frombuffer(jl, np.uint8), cv2.IMREAD_GRAYSCALE)
    R = cv2.imdecode(np.frombuffer(jr, np.uint8), cv2.IMREAD_GRAYSCALE)
    if L is None or R is None:  # a few bag frames are corrupt JPEGs
        return empty
    mask = np.full(L.shape, 255, np.uint8)
    mask[L.shape[0] - MASK_BOTTOM:] = 0
    kl, dl = orb.detectAndCompute(L, mask)
    if dl is None:
        return empty
    pl = np.float32([k.pt for k in kl])
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p0 = pl.reshape(-1, 1, 2)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(L, R, p0, None, **lk)
    p2, st2, _ = cv2.calcOpticalFlowPyrLK(R, L, p1, None, **lk)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((p2 - p0).reshape(-1, 2), axis=1) < 0.3)
    a, b = _norm(pl, K0, D0), _norm(p1.reshape(-1, 2), K1, D1)
    Xh = cv2.triangulatePoints(np.eye(3, 4), T_C1C0[:3], a.T, b.T)
    Xc = (Xh[:3] / Xh[3]).T
    X1 = Xc @ T_C1C0[:3, :3].T + T_C1C0[:3, 3]
    ok &= (Xc[:, 2] > 0.3) & (Xc[:, 2] < 60) & (X1[:, 2] > 0.3)
    ok &= np.linalg.norm(Xc[:, :2] / Xc[:, 2:] - a, axis=1) < 0.5 / F
    ok &= np.linalg.norm(X1[:, :2] / X1[:, 2:] - b, axis=1) < 0.5 / F
    X = np.full((len(pl), 3), np.nan, np.float32)
    X[ok] = Xc[ok]
    return {"un": a.astype(np.float32), "desc": dl, "X": X}


def tiny_descriptor(jl):
    """SeqSLAM-style global descriptor: small, patch-normalised thumbnail of the left image."""
    L = cv2.imdecode(np.frombuffer(jl, np.uint8), cv2.IMREAD_GRAYSCALE)
    if L is None:
        return np.zeros(48 * 36, np.float32)
    L = L[: L.shape[0] - MASK_BOTTOM]
    t = cv2.resize(L, (48, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    # patch normalisation (8x8 blocks of the 48x36 thumbnail, i.e. 6x... cells)
    for y in range(0, 36, 6):
        for x in range(0, 48, 6):
            p = t[y:y + 6, x:x + 6]
            t[y:y + 6, x:x + 6] = (p - p.mean()) / (p.std() + 1e-3)
    v = t.ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def temporal_points(feats, Pk, base=0.5, max_look=400):
    """3D points (cam0 frame) for every keyframe, triangulated against a neighbouring keyframe >= base m away
    using the odometry relative pose. The 7.5 cm stereo baseline is too short for this mostly-far scene
    (and its row-aligned mismatches bias far depths low), the temporal baseline is ~7x longer and shares
    the odometry scale. The partner search is by distance, not keyframe count: the mower sits still for a
    while at the start and end of a run (time-triggered keyframes), exactly where the closing loop is."""
    n = len(feats)
    T_ci = np.linalg.inv(T_IC0)
    pos = Pk[:, :3, 3]
    for k in range(n):
        fk = feats[k]
        fk["X"] = np.full((len(fk["un"]), 3), np.nan, np.float32)
        if fk["desc"] is None:
            continue
        partners = []
        for rng in (range(k + 1, min(n, k + max_look)), range(k - 1, max(-1, k - max_look), -1)):
            near = far = None
            for m in rng:
                if feats[m]["desc"] is None:
                    continue
                d = np.linalg.norm(pos[m] - pos[k])
                rot = Rotation.from_matrix(Pk[k][:3, :3].T @ Pk[m][:3, :3]).magnitude()
                if rot > np.radians(20) or d > 4.0:
                    break  # view no longer shared
                if d >= base and near is None:
                    near = m
                if d >= 1.5:
                    far = m
                    break
            if far is not None or near is not None:
                partners.append(far if far is not None else near)
        for partner in partners:
            _triangulate_pair(fk, feats[partner], Pk[k], Pk[partner], T_ci)


def _triangulate_pair(fk, fm, Pkk, Pm, T_ci):
    """Fill fk['X'] (where still empty) from matches with keyframe fm, using odometry poses Pkk, Pm."""
    mt = _match(fk["desc"], fm["desc"])
    if len(mt) < 20:
        return
    ia, ib = np.array([i for i, _ in mt]), np.array([j for _, j in mt])
    a, b = fk["un"][ia].astype(np.float64), fm["un"][ib].astype(np.float64)
    T_cm_ck = T_ci @ np.linalg.inv(Pm) @ Pkk @ T_IC0
    # epipolar check with the known relative pose
    R, t = T_cm_ck[:3, :3], T_cm_ck[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    ah, bh = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    l = ah @ E.T
    epi = np.abs(np.sum(bh * l, 1)) / np.linalg.norm(l[:, :2], axis=1)
    Xh = cv2.triangulatePoints(np.eye(3, 4), T_cm_ck[:3], a.T, b.T)
    Xc = (Xh[:3] / Xh[3]).T
    Xm = Xc @ R.T + t
    ok = (epi < 2.0 / F) & (Xc[:, 2] > 0.3) & (Xc[:, 2] < 40) & (Xm[:, 2] > 0.3)
    ok &= np.linalg.norm(Xc[:, :2] / Xc[:, 2:] - a, axis=1) < 2.0 / F
    ok &= np.linalg.norm(Xm[:, :2] / Xm[:, 2:] - b, axis=1) < 2.0 / F
    # parallax >= 1 deg so depth is well conditioned
    cosang = np.sum(Xc * (Xc - (-R.T @ t)), 1) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(Xc + R.T @ t, axis=1))
    ok &= cosang < np.cos(np.radians(1.0))
    ok &= np.isnan(fk["X"][ia, 0])
    fk["X"][ia[ok]] = Xc[ok]


def pnp(fa, fb):
    """Pose of keyframe b's cam0 relative to a's cam0 (T_a_b) from a's 3D points and b's 2D features."""
    if fa["desc"] is None or fb["desc"] is None:
        return None, 0
    has3d = np.where(~np.isnan(fa["X"][:, 0]))[0]
    if len(has3d) < args.min_inliers:
        return None, 0
    m = _match(fa["desc"][has3d], fb["desc"])
    if len(m) < args.min_inliers:
        return None, 0
    obj = fa["X"][has3d[[i for i, _ in m]]].astype(np.float64)
    img = fb["un"][[j for _, j in m]].astype(np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, np.eye(3), None, reprojectionError=2.0 / F,
                                             iterationsCount=500, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < args.min_inliers:
        return None, 0 if inl is None else len(inl)
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl[:, 0]], img[inl[:, 0]], np.eye(3), None, rvec, tvec)
    T_b_a = np.eye(4)
    T_b_a[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_b_a[:3, 3] = tvec.ravel()
    return np.linalg.inv(T_b_a), len(inl)


def n3d(f):
    return int(np.sum(~np.isnan(f["X"][:, 0]))) if len(f["X"]) else 0


def verify_loop(fi, fj):
    """Rig-frame T_i_j if PnP i->j and j->i agree. If only one keyframe has enough 3D points (e.g. parked
    right after a turn, no triangulation partner), accept a one-way PnP with twice the inlier count."""
    T1, n1 = pnp(fi, fj)
    T2, n2 = pnp(fj, fi)
    if T1 is None and T2 is None:
        return None, max(n1, n2)
    if T1 is None or T2 is None:
        one_way_ok = (T1 is not None and n1 >= 2 * args.min_inliers and n3d(fj) < args.min_inliers) or \
                     (T2 is not None and n2 >= 2 * args.min_inliers and n3d(fi) < args.min_inliers)
        if not one_way_ok:
            return None, max(n1, n2)
        T = T1 if T1 is not None else np.linalg.inv(T2)
        return T_IC0 @ T @ np.linalg.inv(T_IC0), max(n1, n2)
    E = T1 @ T2  # should be identity
    if np.linalg.norm(E[:3, 3]) > 0.10 or Rotation.from_matrix(E[:3, :3]).magnitude() > np.radians(1.5):
        return None, min(n1, n2)
    return T_IC0 @ T1 @ np.linalg.inv(T_IC0), min(n1, n2)


# ---------------- 4-DoF pose graph ----------------
def gravity_rp(imu, t_ns, win_s=2.0):
    """Roll/pitch rotation (body -> level, no yaw) per time from the low-passed accelerometer: over +-win_s
    the mower's own accelerations and the vibration average out, leaving gravity."""
    acc_bias = np.array(calib["calib_accel_bias"][:3])
    ti = imu[:, 0]
    cs = np.vstack([np.zeros(3), np.cumsum(imu[:, 1:4] - acc_bias, axis=0)])
    out = []
    for t in t_ns:
        a, b = np.searchsorted(ti, [t - win_s * 1e9, t + win_s * 1e9])
        g = (cs[b] - cs[a]) / max(b - a, 1)
        R, _ = Rotation.align_vectors([[0, 0, 1]], [g / np.linalg.norm(g)])
        out.append(R.as_matrix())
    return np.array(out)


def rel4(Ti, Tj, Rrp_i=None):
    """4-DoF measurement from full poses: translation in i's yaw-free (gravity) frame, yaw difference.
    Rrp_i: roll/pitch of i (body -> level); defaults to the odometry's own."""
    if Rrp_i is None:
        Rrp_i = rz(-yaw_of(Ti[:3, :3])) @ Ti[:3, :3]
    tij = Ti[:3, :3].T @ (Tj[:3, 3] - Ti[:3, 3])
    return Rrp_i @ tij, wrap(yaw_of(Tj[:3, :3]) - yaw_of(Ti[:3, :3]))


def meas4(Ti, Tij, Rrp_i=None):
    """4-DoF measurement from i's odometry pose and a measured relative pose T_ij."""
    return rel4(Ti, Ti @ Tij, Rrp_i)


args_max_nfev = 200


def optimise(x0, edges, n, k=1.0, x_init=None):
    """edges: list of (i, j, t_meas(3), yaw_meas, sigma_t, sigma_yaw, is_loop, active).
    Loop edges get a Cauchy kernel with scale k (in sigmas); x0[:4] anchors the first node."""
    act = [e for e in edges if e[7]]
    m = len(act)
    J = lil_matrix((4 * m + 4, 4 * n), dtype=int)
    for k, (i, j, *_r) in enumerate(act):
        J[4 * k:4 * k + 4, 4 * i:4 * i + 4] = 1
        J[4 * k:4 * k + 4, 4 * j:4 * j + 4] = 1
    J[4 * m:, 0:4] = 1
    I = np.array([e[0] for e in act]); Jn = np.array([e[1] for e in act])
    tm = np.array([e[2] for e in act]); ym = np.array([e[3] for e in act])
    st = np.array([e[4] for e in act]); sy = np.array([e[5] for e in act])
    loop = np.array([e[6] for e in act])

    def res(x):
        X = x.reshape(n, 4)
        yi = X[I, 3]
        c, s = np.cos(yi), np.sin(yi)
        d = X[Jn, :3] - X[I, :3]
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
        r_t = (local - tm) / st[:, None]
        r_y = wrap(X[Jn, 3] - yi - ym) / sy
        r = np.concatenate([r_t, r_y[:, None]], 1)
        # Cauchy on loop edges (applied per edge on the whitened norm)
        nrm = np.linalg.norm(r, axis=1)
        w = np.where(loop, k * np.sqrt(np.log1p((nrm / k) ** 2)) / np.maximum(nrm, 1e-9), 1.0)
        return np.concatenate([(r * w[:, None]).ravel(), (X[0] - x0[:4]) * 1e3])

    sol = least_squares(res, x0 if x_init is None else x_init, jac_sparsity=J, method="trf", x_scale="jac", max_nfev=args_max_nfev)
    optimise.last = sol
    return sol.x.reshape(n, 4)


def edge_error(X, e):
    i, j, tm, ym, st, sy = e[:6]
    c, s = np.cos(X[i, 3]), np.sin(X[i, 3])
    d = X[j, :3] - X[i, :3]
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])
    return np.linalg.norm(local - tm), abs(wrap(X[j, 3] - X[i, 3] - ym))


# ---------------- per bag ----------------
def process(tag):
    fe = Path(args.fe_dir)
    z = np.load(fe / f"{tag}_fe.npz")
    kf_idx = z["kf_idx"]
    P, n_sub = build_odometry(z["pose_in"], z["pose_st"], args.odom, t_ns=z["t"])
    t = z["t"] * 1e-9
    Pk = P[kf_idx]
    n = len(kf_idx)
    path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pk[:, :3, 3], axis=0), axis=1))])

    Rrp = gravity_rp(z["imu"], z["t"][kf_idx]) if args.gravity else \
        np.array([rz(-yaw_of(M[:3, :3])) @ M[:3, :3] for M in Pk])
    edges = []
    for k in range(n - 1):
        tm, ym = rel4(Pk[k], Pk[k + 1], Rrp[k])
        d = path[k + 1] - path[k]
        edges.append((k, k + 1, tm, ym, 0.005 + 0.02 * d, np.radians(0.05 + 0.3 * d), False, True))

    loops, n_cand = [], 0
    if not args.no_loops:
        feat_file = fe / f"{tag}_feat2.pkl"
        if feat_file.exists():
            feats = pickle.load(open(feat_file, "rb"))
        else:
            kfs = pickle.load(open(fe / f"{tag}_kf.pkl", "rb"))
            orb = cv2.ORB_create(3000)
            feats = [kf_features(l, r, orb) for l, r in kfs]
            pickle.dump(feats, open(feat_file, "wb"))
        if args.points == "temporal":
            temporal_points(feats, Pk)
        tiny_file = fe / f"{tag}_tiny.npy"
        if tiny_file.exists():
            tiny = np.load(tiny_file)
        else:
            kfs = pickle.load(open(fe / f"{tag}_kf.pkl", "rb"))
            tiny = np.array([tiny_descriptor(l) for l, _ in kfs])
            np.save(tiny_file, tiny)
        pos, yaw = Pk[:, :3, 3], np.array([yaw_of(M[:3, :3]) for M in Pk])
        tk = t[kf_idx]
        for j in range(n):
            older = np.where(tk < tk[j] - args.min_gap_s)[0]
            if not len(older):
                continue
            dist = np.linalg.norm(pos[older, :2] - pos[j, :2], axis=1)
            rad = np.minimum(args.radius + args.drift * (path[j] - path[older]), args.radius_max)
            ok = (dist < rad) & (np.abs(wrap(yaw[older] - yaw[j])) < np.radians(args.max_yaw_deg))
            if args.rank == "tiny":
                cand = older[ok][np.argsort(-(tiny[older[ok]] @ tiny[j]))]
            else:
                cand = older[ok][np.argsort(dist[ok])]
            tried, found = [], 0
            for i in cand:
                if len(tried) >= args.cands or found >= args.loops_per_query:
                    break
                if any(abs(i - k) < 5 for k in tried):
                    continue
                tried.append(i)
                n_cand += 1
                Tij, ninl = verify_loop(feats[i], feats[j])
                if Tij is not None:
                    tm, ym = meas4(Pk[i], Tij, Rrp[i])
                    loops.append([i, j, tm, ym, 0.05, np.radians(0.5), True, True, ninl])
                    found += 1

    # initial values from odometry, then optimise with iterative loop rejection
    x0 = np.array([[*M[:3, 3], yaw_of(M[:3, :3])] for M in Pk]).ravel()
    X = x0.reshape(n, 4)
    all_edges = edges + [tuple(l[:8]) for l in loops]
    n_rej = 0
    if loops:
        # graduated non-convexity: a wide kernel first lets the true loops pull a metres-drifted graph into
        # place (a tight kernel would let the odometry win and reject them), then tighten
        X = x0.reshape(n, 4)
        for k in args.gnc:
            X = optimise(x0, all_edges, n, k=k, x_init=X.ravel())
        for idx, e in enumerate(all_edges):
            if e[6]:
                et, ey = edge_error(X, e)
                if et > args.reject_t or ey > np.radians(args.reject_yaw):
                    all_edges[idx] = e[:7] + (False,)
                    n_rej += 1
        X = optimise(x0, all_edges, n, k=args.gnc[-1], x_init=X.ravel())

    # keyframe trajectory: optimised x,y,z,yaw + odometry roll/pitch
    rows = []
    for k in range(n):
        q = Rotation.from_matrix(rz(X[k, 3]) @ Rrp[k]).as_quat()
        rows.append([t[kf_idx[k]], *X[k, :3], *q])
    rows = np.array(rows)
    suf = args.suffix
    np.savetxt(fe / f"{tag}_be{suf}.txt", rows, fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
    odo = np.array([[t[i], *P[i][:3, 3], *Rotation.from_matrix(P[i][:3, :3]).as_quat()] for i in kf_idx])
    np.savetxt(fe / f"{tag}_odo{suf}.txt", odo, fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
    stats = {"tag": tag, "kf": int(n), "guard_substituted_frames": int(n_sub),
             "both_implausible_frames": int(build_odometry.n_implausible), "loop_candidates": n_cand,
             "loops_verified": len(loops), "loops_rejected": n_rej,
             "closure_odo_m": float(np.linalg.norm(odo[-1, 1:4] - odo[0, 1:4])),
             "closure_be_m": float(np.linalg.norm(rows[-1, 1:4] - rows[0, 1:4]))}
    json.dump(stats, open(fe / f"{tag}_be{suf}.json", "w"), indent=1)
    np.savez(fe / f"{tag}_loops{suf}.npz", ij=np.array([l[:2] for l in loops], int).reshape(-1, 2),
             tm=np.array([l[2] for l in loops]).reshape(-1, 3), ym=np.array([l[3] for l in loops]),
             ninl=np.array([l[8] for l in loops]), active=np.array([e[7] for e in all_edges[len(edges):]], bool),
             path=path, odo4=x0.reshape(n, 4), opt4=X)
    return stats


if __name__ == "__main__":
    fe = Path(args.fe_dir)
    tags = sorted(p.name[:-7] for p in fe.glob("*_fe.npz")
                  if not (fe / "logs").exists() or "FE_DONE" in (fe / "logs" / f"{p.name[:-7]}.log").read_text())
    if args.only:
        tags = [t for t in tags if args.only in t]
    with Pool(args.jobs) as pool:
        for s in pool.imap_unordered(process, tags):
            print(f"{s['tag']}: kf {s['kf']} guard {s['guard_substituted_frames']} loops {s['loops_verified']}/"
                  f"{s['loop_candidates']} rej {s['loops_rejected']}  closure odo {s['closure_odo_m']:.3f} -> "
                  f"be {s['closure_be_m']:.3f}", flush=True)
