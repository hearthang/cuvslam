"""Compare closure error (first/last assumed coincident) across run configurations.

Usage: python compare_configs.py [--bags BAG ...] DIR:SUFFIX[:LABEL] ...
  e.g. python compare_configs.py results/all:_stereo_slam:base results/exp:_stereo_slam_planar:planar
Each DIR:SUFFIX selects files <DIR>/<bagtag><SUFFIX>_{vo,slam}.txt; only bags present in every config are used.
"""
import argparse
import glob
import os

import numpy as np

from loop_check import closure_error, traj_first_last

ap = argparse.ArgumentParser()
ap.add_argument("configs", nargs="+")
ap.add_argument("--traj", nargs="+", default=["vo", "slam"])
args = ap.parse_args()

cfgs = []
for c in args.configs:
    d, suf, *label = c.split(":")
    files = glob.glob(f"{d}/*{suf}_summary.json")
    bags = {os.path.basename(f)[: -len(f"{suf}_summary.json")].replace("_stereo", "") for f in files}
    cfgs.append((label[0] if label else suf, d, suf, bags))
common = sorted(set.intersection(*[c[3] for c in cfgs]))
print(f"{len(common)} common bags\n")


def load(d, suf, bag, traj):
    stem = bag.replace("_2026", "_2026", 1)
    f = glob.glob(f"{d}/{stem}_stereo{suf.replace('_stereo', '', 1)}_{traj}.txt")
    return np.loadtxt(f[0])


res = {}
for label, d, suf, _ in cfgs:
    for traj in args.traj:
        vals = []
        for bag in common:
            _, exy, yaw = closure_error(traj_first_last(load(d, suf, bag, traj)), np.eye(4))
            vals.append((exy, yaw))
        res[(label, traj)] = np.array(vals)

hdr = f"{'bag':26s}" + "".join(f"{l + '/' + t:>18s}" for l, *_ in cfgs for t in args.traj)
print(hdr)
for i, bag in enumerate(common):
    print(f"{bag:26s}" + "".join(f"{res[(l, t)][i, 0]:9.2f}m {res[(l, t)][i, 1]:5.1f}°" for l, *_ in cfgs for t in args.traj))
for name, fn in [("median", np.median), ("mean", np.mean), ("max", np.max)]:
    print(f"{name:26s}" + "".join(f"{fn(res[(l, t)][:, 0]):9.2f}m {fn(res[(l, t)][:, 1]):5.1f}°" for l, *_ in cfgs for t in args.traj))
