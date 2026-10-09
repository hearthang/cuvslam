"""Front-end experiment evaluation incl. the hybrid odometry (stereo+IMU rotation, stereo-only translation).

For each run_frontend.py output dir: closure error of the stereo+IMU tracker ("in"), the stereo tracker ("st") and
the hybrid ("hyb": per frame, rotation increment from the inertial tracker, translation increment from the stereo
tracker, both after the physical gate and the gyro rotation referee), plus each one's translation-only and
rotation-only error against Basalt (own translation with Basalt's rotation, and vice versa) and its peak speed.

Usage: python hybrid_eval.py DIR [DIR ...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

dirs = sys.argv[1:]
sys.argv = sys.argv[:1]
import backend as B  # noqa: E402

D = Path("/mnt/f/data_set/slam")


def basalt(tag):
    d, b = tag.split("_", 1)
    n = b.removeprefix("basalt_")
    return np.loadtxt(next(p for p in (D / d / f"vio_basalt_{n}.txt", D / d / b / f"vio_basalt_{n}.txt") if p.exists()))


def hybrid(Pr, Pt):
    P = [np.eye(4)]
    for k in range(len(Pr) - 1):
        inc = np.eye(4)
        inc[:3, :3] = Pr[k][:3, :3].T @ Pr[k + 1][:3, :3]
        inc[:3, 3] = Pt[k][:3, :3].T @ (Pt[k + 1][:3, 3] - Pt[k][:3, 3])
        P.append(P[-1] @ inc)
    return np.array(P)


def split(P, t, b):
    tt = np.clip(t, b[0, 0], b[-1, 0])
    Rb = Slerp(b[:, 0], Rotation.from_quat(b[:, 4:8]))(tt).as_matrix()
    pb = np.stack([np.interp(tt, b[:, 0], b[:, k]) for k in (1, 2, 3)], 1)
    pt, pr = np.zeros(3), np.zeros(3)
    for k in range(len(t) - 1):
        pt += Rb[k] @ (P[k][:3, :3].T @ (P[k + 1][:3, 3] - P[k][:3, 3]))
        pr += P[k][:3, :3] @ (Rb[k].T @ (pb[k + 1] - pb[k]))
    ref = pb[-1] - pb[0]
    return np.linalg.norm(pt - ref), np.linalg.norm(pr - ref)


for d in dirs:
    acc = {m: [] for m in ("in", "st", "hyb")}
    bas = []
    for f in sorted(Path(d).glob("*_fe.npz")):
        tag = f.name[:-7]
        z = np.load(f)
        t = z["t"] * 1e-9
        grv = B.gyro_rotvecs(z["imu"], z["t"])
        Pi = B.clean_chain(z["t"], z["pose_in"], grv)
        has_st = not np.isnan(z["pose_st"][:, 0]).all()
        Ps = B.clean_chain(z["t"], z["pose_st"], grv) if has_st else None
        b = basalt(tag)
        bas.append(np.linalg.norm(b[-1, 1:4] - b[0, 1:4]))
        cands = (("in", Pi), ("st", Ps), ("hyb", hybrid(Pi, Ps))) if has_st else (("in", Pi),)
        for m, P in cands:
            tr, rot = split(P, t, b)
            v = np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1) / np.diff(t)
            acc[m].append((np.linalg.norm(P[-1][:3, 3] - P[0][:3, 3]), tr, rot, v.max()))
    for m, rows in acc.items():
        if not rows:
            continue
        r = np.array(rows)
        print(f"{d:28s} {m:3s} n={len(r)}: closure mean {r[:, 0].mean():.2f} median {np.median(r[:, 0]):.2f} max {r[:, 0].max():.2f} "
              f"| translation-only {r[:, 1].mean():.2f} rotation-only {r[:, 2].mean():.2f} | peak speed {r[:, 3].max():.1f} m/s")
    print(f"{'':28s} basalt mean {np.mean(bas):.2f} median {np.median(bas):.2f}")
