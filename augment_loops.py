"""Verify cross-boundary visual revisits omitted by the limited online-style shortlist.

No zero-displacement or same-heading assumption is imposed. Only image-derived
relative poses passing independent forward/backward stereo PnP enter the graph.
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('fe_dir', nargs='?', default='results/fe5')
parser.add_argument('--only', default='')
parser.add_argument('--jobs', type=int, default=3)
parser.add_argument('--window', type=float, default=30)
parser.add_argument('--features', choices=['orb', 'sift'], default='sift')
parser.add_argument('--suffix', default='_boundary')
parser.add_argument('--stride', type=int, default=3)
parser.add_argument('--calib', default='/mnt/f/data_set/slam/calibration_front_small.json')
parser.add_argument('--cam1-fscale', type=float, default=-0.00038)
args = None
import backend as B


def configure(config):
    global args
    args = argparse.Namespace(**config)
    B.configure(args.calib, args.cam1_fscale)


def match(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    norm = cv2.NORM_HAMMING if d1.dtype == np.uint8 else cv2.NORM_L2
    pairs = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    # Require mutual nearest neighbours to suppress repeated grass patterns.
    reverse = cv2.BFMatcher(norm).match(d2, d1)
    back = {m.queryIdx: m.trainIdx for m in reverse}
    return [(m.queryIdx, m.trainIdx) for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]


def run(tag):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    B._match = match
    fe = Path(args.fe_dir)
    z = np.load(fe / f'{tag}_fe.npz')
    t = (z['t'][z['kf_idx']] - z['t'][0]) * 1e-9
    first = np.flatnonzero(t <= args.window)[::args.stride]
    last = np.unique(np.r_[np.flatnonzero(t >= t[-1] - args.window)[::args.stride], len(t) - 1])
    ids = np.unique(np.r_[first, last])
    if args.features == 'orb':
        ff = pickle.load(open(fe / f'{tag}_feat2.pkl', 'rb'))
        feats = {i: ff[i] for i in ids}
    else:
        imgs = pickle.load(open(fe / f'{tag}_kf.pkl', 'rb'))
        detector = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02)
        feats = {i: B.kf_features(*imgs[i], detector) for i in ids}
        del imgs
        for f in feats.values():
            if f['desc'] is not None:
                d = f['desc']
                f['desc'] = np.sqrt(d / np.maximum(d.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
    pairs, transforms, inliers, cycles = [], [], [], []
    for j in last:
        accepted = []
        for i in first:
            if t[j] - t[i] < 30:
                continue
            T1, n1 = B.pnp(feats[i], feats[j])
            if T1 is None:
                continue
            T2, n2 = B.pnp(feats[j], feats[i])
            if T2 is None:
                continue
            E = T1 @ T2
            et = np.linalg.norm(E[:3, 3])
            er = np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude())
            if et > 0.10 or er > 1.5:
                continue
            T2 = np.linalg.inv(T2)
            T = np.eye(4)
            T[:3, 3] = 0.5 * (T1[:3, 3] + T2[:3, 3])
            T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([T1[:3, :3], T2[:3, :3]]))([0.5]).as_matrix()[0]
            T = B.T_IC0 @ T @ np.linalg.inv(B.T_IC0)
            accepted.append((min(n1, n2), i, T, et, er))
        # Cap correlated constraints at each query and prefer the strongest evidence.
        for nin, i, T, et, er in sorted(accepted, key=lambda a: -a[0])[:3]:
            pairs.append((i, j)); transforms.append(T); inliers.append(nin); cycles.append((et, er))
    dest = fe / f'{tag}_extra{args.suffix}.npz'
    np.savez(dest, ij=np.array(pairs, dtype=int).reshape(-1, 2), T=np.array(transforms).reshape(-1, 4, 4),
             ninl=np.array(inliers), cycles=np.array(cycles).reshape(-1, 2), config=json.dumps(vars(args)))
    print(f'{tag}: {len(first)} x {len(last)} candidates, {len(pairs)} verified {args.features} boundary loops', flush=True)


if __name__ == '__main__':
    configure(vars(parser.parse_args()))
    tags = [p.name[:-7] for p in sorted(Path(args.fe_dir).glob('*_fe.npz')) if args.only in p.name]
    with Pool(args.jobs, initializer=configure, initargs=(vars(args),)) as pool:
        pool.map(run, tags)
