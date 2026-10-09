"""Closure error of backend.py outputs vs Basalt VIO (same assume-closed metric as basalt_closure_ratios csv).

Usage: python summarize_be.py SUFFIX [SUFFIX ...]     (e.g. _tin25; "" = default run)
"""
import json
import sys
from pathlib import Path

import numpy as np

FE = Path("results/fe")
DATA = Path("/mnt/f/data_set/slam")


def closure(path):
    a = np.loadtxt(path, ndmin=2)
    p = a[:, 1:4]
    return np.linalg.norm(p[-1] - p[0]), np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def basalt(tag):
    date, bag = tag.split("_", 1)
    name = bag.removeprefix("basalt_")
    for c in (DATA / date / f"vio_basalt_{name}.txt", DATA / date / bag / f"vio_basalt_{name}.txt"):
        if c.exists():
            return closure(c)[0]


sufs = sys.argv[1:] or [""]
tags = sorted(p.name[:-7] for p in FE.glob("*_fe.npz"))
rows = []
for tag in tags:
    r = {"tag": tag, "basalt": basalt(tag)}
    for s in sufs:
        f = FE / f"{tag}_be{s}.txt"
        if f.exists():
            r[s] = closure(f)[0]
            r[s + "_odo"] = closure(FE / f"{tag}_odo{s}.txt")[0]
            r[s + "_stats"] = json.load(open(FE / f"{tag}_be{s}.json"))
    rows.append(r)

full = [r for r in rows if all(s in r for s in sufs)]
print(f"{len(full)}/{len(rows)} bags with all results\n")
print(f"{'bag':40s} {'basalt':>7s} " + " ".join(f"{s or 'be':>14s}" for s in sufs))
for r in full:
    cells = []
    for s in sufs:
        mark = "*" if r[s] > r["basalt"] else " "
        cells.append(f"{r[s + '_odo']:6.2f}->{r[s]:5.2f}{mark}")
    print(f"{r['tag']:40s} {r['basalt']:7.3f} " + " ".join(cells))
b = np.array([r["basalt"] for r in full])
print(f"\n{'method':16s} {'mean':>7s} {'median':>7s} {'max':>7s} {'wins':>6s}")
print(f"{'basalt':16s} {b.mean():7.3f} {np.median(b):7.3f} {b.max():7.3f}")
for s in sufs:
    for k, lab in ((s + "_odo", f"odo{s}"), (s, f"slam{s}")):
        e = np.array([r[k] for r in full])
        print(f"{lab:16s} {e.mean():7.3f} {np.median(e):7.3f} {e.max():7.3f} {int((e < b).sum()):3d}/{len(full)}")
