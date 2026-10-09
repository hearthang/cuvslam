"""Front end (odometry only, no loop closure) closure error vs Basalt VIO, same assume-closed metric.

Usage: python fe_vs_basalt.py [FE_DIR=results/fe2] [MODE ...]   (modes: in st guard; default in guard)
"""
import csv
import sys
from pathlib import Path

import numpy as np

fe_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/fe2")
modes = sys.argv[2:] or ["in", "guard"]
sys.argv = sys.argv[:1]
import backend as B  # noqa: E402

basalt = {}
for r in csv.DictReader(open("/mnt/f/data_set/slam/basalt_closure_ratios_20260929.csv")):
    basalt[f"{r['date']}_{r['bag']}"] = float(r["closure_error_m"])


def basalt_closure(tag):
    if tag in basalt:
        return basalt[tag]
    date, bag = tag.split("_", 1)  # 9-29 is not in the csv: compute from its trajectory
    a = np.loadtxt(Path("/mnt/f/data_set/slam") / date / f"vio_basalt_{bag}.txt")
    return float(np.linalg.norm(a[-1, 1:4] - a[0, 1:4]))


rows = []
for f in sorted(fe_dir.glob("*_fe.npz")):
    tag = f.name[:-7]
    z = np.load(f)
    r = {"tag": tag, "basalt": basalt_closure(tag)}
    for m in modes:
        if m in ("vote", "vote_nogyro"):
            P = B.vote_odometry(z, gyro=(m == "vote"))
        elif m == "in_gyro":
            P = B.clean_chain(z["t"], z["pose_in"], B.gyro_rotvecs(z["imu"], z["t"]))
        else:
            P = B.build_odometry(z["pose_in"], z["pose_st"], m, t_ns=z["t"])[0]
        r[m] = float(np.linalg.norm(P[-1][:3, 3] - P[0][:3, 3]))
    rows.append(r)

b = np.array([r["basalt"] for r in rows])
print(f"{len(rows)} bags\n{'method':10s} {'mean':>7s} {'median':>7s} {'max':>7s}  wins")
print(f"{'basalt':10s} {b.mean():7.3f} {np.median(b):7.3f} {b.max():7.3f}")
for m in modes:
    e = np.array([r[m] for r in rows])
    print(f"{m:10s} {e.mean():7.3f} {np.median(e):7.3f} {e.max():7.3f}  {int((e < b).sum())}/{len(rows)}")
if "-v" in sys.argv or True:
    worst = sorted(rows, key=lambda r: r[modes[0]] - r["basalt"], reverse=True)[:8]
    print("\nlargest gaps to Basalt (" + modes[0] + "):")
    for r in worst:
        print(f"  {r['tag']:40s} basalt {r['basalt']:.3f}  " + "  ".join(f"{m} {r[m]:.3f}" for m in modes))
