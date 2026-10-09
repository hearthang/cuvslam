"""Front-end experiments vs Basalt: closure error and yaw agreement with Basalt's trajectory.

Usage: python fe_eval_basalt.py DIR [DIR ...]      (modes evaluated: in, st)
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

dirs = sys.argv[1:]
sys.argv = sys.argv[:1]
import backend as B  # noqa: E402

D = Path("/mnt/f/data_set/slam")
bas_csv = {f"{r['date']}_{r['bag']}": float(r["closure_error_m"])
           for r in csv.DictReader(open(D / "basalt_closure_ratios_20260929.csv"))}


def basalt_traj(tag):
    d, b = tag.split("_", 1)
    n = b.removeprefix("basalt_")
    for c in (D / d / f"vio_basalt_{n}.txt", D / d / b / f"vio_basalt_{n}.txt"):
        if c.exists():
            return np.loadtxt(c)


for d in dirs:
    for mode in ("in", "st"):
        rows = []
        for f in sorted(Path(d).glob("*_fe.npz")):
            tag = f.name[:-7]
            z = np.load(f)
            col = {"in": "pose_in", "st": "pose_st"}[mode]
            if np.isnan(z[col][:, 0]).all():
                continue
            P = B.clean_chain(z["t"], z[col], B.gyro_rotvecs(z["imu"], z["t"]))
            t = z["t"] * 1e-9
            b = basalt_traj(tag)
            tq = np.linspace(max(t[0], b[0, 0]), min(t[-1], b[-1, 0]), 500)
            yo = np.interp(tq, t, np.unwrap([np.arctan2(M[1, 0], M[0, 0]) for M in P]))
            yb = np.interp(tq, b[:, 0], np.unwrap(Rotation.from_quat(b[:, 4:8]).as_euler("zyx")[:, 0]))
            dy = np.degrees((yo - yo[0]) - (yb - yb[0]))
            rows.append((np.linalg.norm(P[-1][:3, 3] - P[0][:3, 3]), bas_csv.get(tag, np.nan), dy[-1], np.abs(dy).max()))
        if not rows:
            continue
        r = np.array(rows)
        print(f"{d:32s} {mode}: closure mean {r[:, 0].mean():.3f} median {np.median(r[:, 0]):.3f} "
              f"(basalt {np.nanmean(r[:, 1]):.3f}/{np.nanmedian(r[:, 1]):.3f}) | yaw vs basalt end |.| median "
              f"{np.median(np.abs(r[:, 2])):.2f} max-in-run median {np.median(r[:, 3]):.2f} deg | n={len(r)}")
