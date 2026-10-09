"""Front-end (odometry only, no loop closure) accuracy of run_frontend.py outputs.

Reference: the loop-closed back-end trajectory results/fe/{tag}_be_v7.txt (agrees with Basalt to ~0.16 m).
For each output dir and tracker (in / st / guard) reports, at the last processed frame:
  horizontal drift |p_xy - p_ref_xy| [m] and yaw drift [deg] vs the reference, and the closure error
  |p_last - p_first| (meaningful only for full-length runs).

Usage: python fe_eval.py DIR [DIR ...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.argv, dirs = sys.argv[:1], sys.argv[1:]
import backend as B  # noqa: E402  (argparse in backend reads sys.argv)

REF = Path("results/fe")


def yaw_of(R):
    return np.arctan2(R[1, 0], R[0, 0])


def evaluate(d):
    out = {}
    for f in sorted(Path(d).glob("*_fe.npz")):
        tag = f.name[:-7]
        ref = np.loadtxt(REF / f"{tag}_be_v7.txt")
        z = np.load(f)
        t = z["t"] * 1e-9
        yref = np.unwrap([Rotation.from_quat(q).as_euler("zyx")[0] for q in ref[:, 4:8]])
        k = min(np.searchsorted(ref[:, 0], t[-1]), len(ref) - 1)  # reference keyframe at the end time
        i = np.searchsorted(t, ref[k, 0]).clip(0, len(t) - 1)
        for mode in ("in", "st", "guard"):
            col = {"in": "pose_in", "st": "pose_st"}.get(mode)
            if col and np.isnan(z[col][:, 0]).all():
                continue
            if mode == "guard" and (np.isnan(z["pose_in"][:, 0]).all() or np.isnan(z["pose_st"][:, 0]).all()):
                continue
            P, _ = B.build_odometry(z["pose_in"], z["pose_st"], mode, t_ns=z["t"])
            yo = np.unwrap([yaw_of(M[:3, :3]) for M in P[: i + 1: 8]] + [yaw_of(P[i][:3, :3])])
            out.setdefault(mode, []).append(dict(
                tag=tag, drift=np.linalg.norm(P[i][:2, 3] - (ref[k, 1:3] - ref[0, 1:3])),
                yaw=np.degrees((yo[-1] - yo[0]) - (yref[k] - yref[0])),
                closure=np.linalg.norm(P[-1][:3, 3] - P[0][:3, 3]), dur=t[i] - t[0]))
    return out


for d in dirs:
    res = evaluate(d)
    for mode, rows in res.items():
        dr = np.array([r["drift"] for r in rows]); yw = np.array([r["yaw"] for r in rows])
        cl = np.array([r["closure"] for r in rows])
        print(f"{d:45s} {mode:5s} n={len(rows)} drift med {np.median(dr):.2f} mean {dr.mean():.2f} | "
              f"yaw med {np.median(yw):+.2f} |yaw| med {np.median(np.abs(yw)):.2f} | closure med {np.median(cl):.2f} "
              f"mean {cl.mean():.2f} | {np.median([r['dur'] for r in rows]):.0f}s")
