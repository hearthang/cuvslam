"""Loop-closure verification + robust SE(2) pose-graph re-optimisation on top of cuVSLAM.

cuVSLAM's own SLAM occasionally accepts a false loop closure on repetitive lawn texture, which bends
the whole trajectory. This back-end rebuilds the graph from
  * odometry edges: cuVSLAM visual odometry between consecutive SLAM keyframes (independent of loop closures)
  * loop edges:     cuVSLAM pose-graph edges between non-consecutive keyframes
checks every loop edge against the odometry chain (drift budget grows with travelled distance) and
against the other loop edges (pairwise consistency, PCM), keeps the largest mutually consistent set,
and solves a planar (x, y, yaw) least-squares problem with a Cauchy kernel on loop edges.
Per-frame poses are keyframe pose (optimised) + VO motion since that keyframe.

Usage: python pgo_robust.py RESULTS_DIR [--suffix _stereo_slam]   -> writes <tag>_pgo.txt
"""
import argparse
import glob
import os

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

# drift budget of the VO chain between two keyframes, as a function of travelled distance d [m]
TRANS_TOL = lambda d: 0.3 + 0.015 * d            # noqa: E731
YAW_TOL = lambda d: np.radians(3.0 + 0.08 * d)   # noqa: E731
# edges use cuVSLAM's own covariances, floored; a missing consecutive edge falls back to VO with these
SIGMA_MIN_T, SIGMA_MIN_Y = 0.01, np.radians(0.05)
SIGMA_VO_T_PER_M, SIGMA_VO_Y_PER_M = 0.02, np.radians(0.3)


def se2(x, y, th):
    return np.array([x, y, th])


def se2_from_row(row):
    """TUM row -> (x, y, yaw) in the z-up imu-aligned world."""
    return se2(row[1], row[2], Rotation.from_quat(row[4:8]).as_euler("zyx")[0])


def se2_from_mat(M):
    return se2(M[0, 3], M[1, 3], np.arctan2(M[1, 0], M[0, 0]))


def compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return se2(a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inv(a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return se2(-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def between(a, b):
    return compose(inv(a), b)


def wrap(t):
    return (t + np.pi) % (2 * np.pi) - np.pi


def mat(row):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(row[3:7]).as_matrix()
    M[:3, 3] = row[:3]
    return M


def run(prefix, reject=True, verbose=True):
    pg = np.load(f"{prefix}_posegraph.npz")
    slam, vo = np.loadtxt(f"{prefix}_slam.txt"), np.loadtxt(f"{prefix}_vo.txt")
    T = pg["T_rig_imu"]
    to_imu = lambda M: T.T @ M @ T  # noqa: E731  rig-frame pose/relative pose -> imu-aligned

    # keyframe timestamps: node poses coincide with frames of the final SLAM trajectory
    ids = pg["node_id"]
    pos = np.array([to_imu(mat(p))[:3, 3] for p in pg["node_pose"]])
    j = np.array([np.linalg.norm(slam[:, 1:4] - p, axis=1).argmin() for p in pos])
    t_node = slam[j, 0]
    order = np.argsort(t_node)
    ids, t_node = ids[order], t_node[order]
    idx_of = {int(i): k for k, i in enumerate(ids)}
    n = len(ids)

    # VO at keyframe times (nearest frame; keyframes are frames so this is exact)
    vo_k = np.array([se2_from_row(vo[np.abs(vo[:, 0] - t).argmin()]) for t in t_node])
    vo_xy = vo[:, 1:3]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(vo_xy, axis=0), axis=1))])
    dist_k = np.interp(t_node, vo[:, 0], cum)  # travelled distance at each keyframe

    # cuVSLAM covariance is (rotation, translation) in the rig frame; express in imu axes and keep the
    # vertical-axis rotation and horizontal translation
    Rr = T[:3, :3]

    def sigmas(cov):
        C = cov.reshape(6, 6)
        Cr, Ct = Rr.T @ C[:3, :3] @ Rr, Rr.T @ C[3:, 3:] @ Rr
        return max(np.sqrt(max(Ct[0, 0], Ct[1, 1])), SIGMA_MIN_T), max(np.sqrt(Cr[2, 2]), SIGMA_MIN_Y)

    odo, loops = {}, []
    for (a, b), tf, cov in zip(pg["edge_ft"], pg["edge_tf"], pg["edge_cov"]):
        ka, kb = idx_of.get(int(a)), idx_of.get(int(b))
        if ka is None or kb is None or ka == kb:
            continue
        meas = se2_from_mat(to_imu(mat(tf)))
        if ka > kb:
            ka, kb, meas = kb, ka, inv(meas)
        if kb - ka == 1:
            odo[ka] = (ka, kb, meas, *sigmas(cov))
        else:
            loops.append((ka, kb, meas, *sigmas(cov)))
    n_vo_fill = 0
    for k in range(n - 1):
        if k not in odo:
            d = dist_k[k + 1] - dist_k[k]
            odo[k] = (k, k + 1, between(vo_k[k], vo_k[k + 1]), SIGMA_MIN_T + SIGMA_VO_T_PER_M * d,
                      SIGMA_MIN_Y + SIGMA_VO_Y_PER_M * d)
            n_vo_fill += 1
    odo = [odo[k] for k in range(n - 1)]

    # 1) single-edge check against the VO chain
    def residual_vs_vo(ka, kb, meas):
        r = between(between(vo_k[ka], vo_k[kb]), meas)
        return np.hypot(r[0], r[1]), abs(r[2])

    ok1 = []
    for ka, kb, meas, *_ in loops:
        d = dist_k[kb] - dist_k[ka]
        et, ey = residual_vs_vo(ka, kb, meas)
        ok1.append(et <= TRANS_TOL(d) and ey <= YAW_TOL(d))
    cand = [l for l, o in zip(loops, ok1) if o]

    # 2) pairwise consistency (PCM): loop a, VO chain, loop b back, VO chain must close
    G = nx.Graph()
    G.add_nodes_from(range(len(cand)))
    for p in range(len(cand)):
        ia, ja, ma = cand[p][:3]
        for q in range(p + 1, len(cand)):
            ib, jb, mb = cand[q][:3]
            # cycle: ia -(la)-> ja -(vo)-> jb -(lb^-1)-> ib -(vo)-> ia
            cyc = compose(compose(compose(ma, between(vo_k[ja], vo_k[jb])), inv(mb)), between(vo_k[ib], vo_k[ia]))
            d = abs(dist_k[ja] - dist_k[jb]) + abs(dist_k[ia] - dist_k[ib])
            if np.hypot(cyc[0], cyc[1]) <= TRANS_TOL(d) and abs(cyc[2]) <= YAW_TOL(d):
                G.add_edge(p, q)
    best = max(nx.find_cliques(G), key=len) if len(cand) else []
    # isolated loops (no partner to test against) are kept only if they passed the single check
    # and do not conflict with the clique, i.e. are in a component without edges
    keep = set(best) | {v for v in G.nodes if G.degree(v) == 0 and len(best) <= 1}
    kept = [cand[i] for i in sorted(keep)] if reject else loops

    # 3) SE(2) least squares; node 0 fixed
    x0 = np.array(vo_k)  # initial guess: VO

    edges = odo + kept

    def fun(v):
        X = np.vstack([x0[:1], v.reshape(-1, 3)])
        r = np.empty(3 * len(edges))
        for e, (ka, kb, m, st, sy) in enumerate(edges):
            d = between(m, between(X[ka], X[kb]))
            r[3 * e:3 * e + 3] = d[0] / st, d[1] / st, wrap(d[2]) / sy
        return r

    sp = lil_matrix((3 * len(edges), 3 * (n - 1)), dtype=int)
    for e, (ka, kb, *_ ) in enumerate(edges):
        for k in (ka, kb):
            if k > 0:
                sp[3 * e:3 * e + 3, 3 * (k - 1):3 * k] = 1
    sol = least_squares(fun, x0[1:].ravel(), jac_sparsity=sp, loss="linear", x_scale="jac", max_nfev=200)
    X = np.vstack([x0[:1], sol.x.reshape(-1, 3)])

    # 4) per-frame poses: optimised keyframe + VO motion since it
    k_of = np.clip(np.searchsorted(t_node, vo[:, 0], side="right") - 1, 0, n - 1)
    out = []
    for row, k in zip(vo, k_of):
        p = compose(X[k], between(vo_k[k], se2_from_row(row)))
        q = Rotation.from_euler("z", p[2]) * Rotation.from_euler("zyx", [0, *Rotation.from_quat(row[4:8]).as_euler("zyx")[1:]])
        out.append([row[0], p[0], p[1], 0.0, *q.as_quat()])
    out = np.array(out)
    np.savetxt(f"{prefix}_pgo{'' if reject else '_all'}.txt", out, fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
    if verbose:
        c0 = 0.5 * np.sum(fun(x0[1:].ravel()) ** 2)
        print(f"{os.path.basename(prefix)}: {n} keyframes, {len(loops)} loop edges, {len(cand)} pass VO check, "
              f"{len(kept)} kept after PCM, {n_vo_fill} odometry gaps filled from VO; cost {c0:.0f} -> {sol.cost:.0f}, "
              f"max node shift {np.abs(X[:, :2] - x0[:, :2]).max():.2f} m")
    return len(loops), len(cand), len(kept)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--suffix", default="_stereo_slam")
    ap.add_argument("--no-reject", action="store_true", help="ablation: keep every loop edge (writes *_pgo_all.txt)")
    a = ap.parse_args()
    for f in sorted(glob.glob(f"{a.results}/*{a.suffix}_posegraph.npz")):
        run(f[: -len("_posegraph.npz")], reject=not a.no_reject)
