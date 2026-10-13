"""Evaluate reference trajectories: closure error vs Basalt, and continuity at the same timestamps as Basalt.

Continuity (both trajectories interpolated to Basalt's 8 Hz timestamps, IMU clock):
  - per step, ours must not exceed 2x Basalt's step + a noise floor (TOL_T / TOL_R)
  - per step, rotation vs the integrated gyro (an independent reference): |ours - gyro| > 0.5 deg
Usage: python eval_ref.py SOURCE [SOURCE ...]
  SOURCE: slam (delivered slam_cuvslam_*.txt) | fe (FE_DIR/<tag>_frontend.txt) | a pattern with {tag},
          e.g. results/fe5/{tag}_ref_v11.txt
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

srcs = sys.argv[1:]
sys.argv = sys.argv[:1]
import backend as B  # noqa: E402

D = Path("/mnt/f/data_set/slam")
FE = Path(os.environ.get("FE_DIR", "results/fe5"))
TOL_T, TOL_R, GYRO_TOL = 0.01, 0.5, 0.5


def basalt_file(tag):
    date, bag = tag.split("_", 1)
    name = bag.removeprefix("basalt_")
    return next(p for p in (D / date / f"vio_basalt_{name}.txt", D / date / bag / f"vio_basalt_{name}.txt") if p.exists()), name


def load(src, tag):
    if src == "slam":
        b, name = basalt_file(tag)
        f = b.parent / f"slam_cuvslam_{name}.txt"
    elif src == "fe":
        f = FE / f"{tag}_frontend.txt"
    else:
        f = Path(src.format(tag=tag))
    return np.loadtxt(f) if f.exists() else None


def interp(a, t):
    p = np.stack([np.interp(t, a[:, 0], a[:, i]) for i in (1, 2, 3)], 1)
    return p, Slerp(a[:, 0], Rotation.from_quat(a[:, 4:8]))(t)


def steps(p, q):
    return np.linalg.norm(np.diff(p, axis=0), axis=1), np.degrees((q[:-1].inv() * q[1:]).magnitude())


def bag(tag):
    bf, _ = basalt_file(tag)
    b = np.loadtxt(bf)
    imu = np.load(FE / f"{tag}_fe.npz")["imu"]
    out = {"tag": tag, "basalt": float(np.linalg.norm(b[-1, 1:4] - b[0, 1:4]))}
    for src in srcs:
        a = load(src, tag)
        if a is None:
            continue
        lo, hi = max(a[0, 0], b[0, 0], imu[0, 0] * 1e-9 + 2), min(a[-1, 0], b[-1, 0], imu[-1, 0] * 1e-9)
        t = b[(b[:, 0] >= lo) & (b[:, 0] <= hi), 0]
        pa, qa = interp(a, t)
        pb, qb = interp(b, t)
        da, ra = steps(pa, qa)
        db, rb = steps(pb, qb)
        g = np.degrees(np.linalg.norm(B.gyro_rotvecs(imu, (t * 1e9).astype(np.int64)), axis=1))
        dt = np.diff(t)
        out[src] = dict(closure=float(np.linalg.norm(a[-1, 1:4] - a[0, 1:4])),
                        tr_bad=int((da > 2 * db + TOL_T).sum()), rot_bad=int((ra > 2 * rb + TOL_R).sum()),
                        tr_bad0=int((da > 2 * db).sum()), rot_bad0=int((ra > 2 * rb).sum()),
                        tr_exc=float((da - 2 * db).max()), rot_exc=float((ra - 2 * rb).max()),
                        gyro_bad=int((np.abs(ra - g) > GYRO_TOL).sum()), gyro_bad_basalt=int((np.abs(rb - g) > GYRO_TOL).sum()),
                        vmax=float((da / dt).max()), wmax=float((ra / dt).max()), n=len(da))
    return out


if __name__ == "__main__":
    tags = sorted(p.name[:-7] for p in FE.glob("*_fe.npz"))
    with Pool(7) as pool:
        rows = pool.map(bag, tags)
    b_all = np.array([r["basalt"] for r in rows])
    print(f"Basalt closure: mean {b_all.mean():.3f} median {np.median(b_all):.3f} max {b_all.max():.3f}  (n={len(rows)})")
    print(f"continuity on Basalt's timestamps: step > 2x Basalt + {TOL_T*100:.0f} cm / {TOL_R} deg (and with no floor); "
          f"|rot step - gyro| > {GYRO_TOL} deg")
    for src in srcs:
        rr = [r for r in rows if src in r]
        if not rr:
            continue
        c = np.array([r[src]["closure"] for r in rr])
        bb = np.array([r["basalt"] for r in rr])
        s = lambda k: sum(r[src][k] for r in rr)  # noqa: E731
        m = lambda k: max(r[src][k] for r in rr)  # noqa: E731
        print(f"\n== {src} (n={len(rr)})")
        print(f"  closure mean {c.mean():.3f} median {np.median(c):.3f} max {c.max():.3f}, better than Basalt {int((c < bb).sum())}/{len(rr)}"
              f"  (Basalt on these: {bb.mean():.3f} / {np.median(bb):.3f})")
        print(f"  translation > 2x Basalt + floor: {s('tr_bad')} steps in {sum(r[src]['tr_bad'] > 0 for r in rr)} bags "
              f"(no floor {s('tr_bad0')}), max excess {m('tr_exc')*100:.1f} cm")
        print(f"  rotation    > 2x Basalt + floor: {s('rot_bad')} steps in {sum(r[src]['rot_bad'] > 0 for r in rr)} bags "
              f"(no floor {s('rot_bad0')}), max excess {m('rot_exc'):.2f} deg")
        print(f"  rotation step off the gyro by > {GYRO_TOL} deg: {s('gyro_bad')} steps (Basalt: {s('gyro_bad_basalt')}) of {s('n')}")
        print(f"  max speed {m('vmax'):.2f} m/s, max turn rate {m('wmax'):.1f} deg/s")
        worst = sorted(rr, key=lambda r: -(r[src]["closure"] - r["basalt"]))[:5]
        print("  largest closure loss vs Basalt: " + ", ".join(f"{r['tag'][:16]} {r[src]['closure']:.2f}/{r['basalt']:.2f}" for r in worst))
