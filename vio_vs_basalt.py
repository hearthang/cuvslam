"""Closure error of vio_fusion.py outputs vs Basalt VIO.  Usage: python vio_vs_basalt.py [FE_DIR] SUFFIX..."""
import csv
import sys
from pathlib import Path

import numpy as np

fe = Path(sys.argv[1])
sufs = sys.argv[2:] or [""]
bas = {f"{r['date']}_{r['bag']}": float(r["closure_error_m"])
       for r in csv.DictReader(open("/mnt/f/data_set/slam/basalt_closure_ratios_20260929.csv"))}


def basalt(tag):
    if tag not in bas:
        d, b = tag.split("_", 1)
        a = np.loadtxt(f"/mnt/f/data_set/slam/{d}/vio_basalt_{b}.txt")
        bas[tag] = float(np.linalg.norm(a[-1, 1:4] - a[0, 1:4]))
    return bas[tag]


def closure(f):
    a = np.loadtxt(f)
    return float(np.linalg.norm(a[-1, 1:4] - a[0, 1:4]))


tags = sorted(p.name[:-7] for p in fe.glob("*_fe.npz"))
b = np.array([basalt(t) for t in tags])
print(f"{'method':22s} {'mean':>6s} {'median':>6s} {'max':>6s} wins   (n={len(tags)})")
print(f"{'basalt':22s} {b.mean():6.3f} {np.median(b):6.3f} {b.max():6.3f}")
res = {}
for s in sufs:
    for kind in ("vio", "vio_smooth"):
        files = [fe / f"{t}_{kind}{s}.txt" for t in tags]
        if not all(f.exists() for f in files):
            continue
        e = np.array([closure(f) for f in files])
        res[kind + s] = e
        print(f"{kind + s:22s} {e.mean():6.3f} {np.median(e):6.3f} {e.max():6.3f} {int((e < b).sum()):2d}/{len(tags)}")
if res and "-v" in sys.argv[0:1] + sys.argv:
    pass
k = next(iter(res))
order = np.argsort(-(res[k] - b))[:6]
print("largest gaps (" + k + "):", ", ".join(f"{tags[i]} {res[k][i]:.2f} vs {b[i]:.2f}" for i in order))
