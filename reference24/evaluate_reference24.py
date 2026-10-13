"""Read-only trajectory audit on common timestamps. Basalt is a comparison, not ground truth.

Example: python evaluate_reference.py --source v10=slam --source candidate=_candidate
All trajectories are evaluated on the same overlapping Basalt timestamps per bag.
Reports closure, path length, derivatives and full SO(3) gyro residuals, without
aligning or modifying the evaluated trajectories.
"""
import argparse
import json
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_trajectory(path):
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != 8 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError(f'Invalid trajectory: {path}')
    if np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError(f'Non-increasing timestamps: {path}')
    if not np.allclose(np.linalg.norm(a[:, 4:8], axis=1), 1, atol=1e-4):
        raise ValueError(f'Non-unit quaternions: {path}')
    return a


def interpolate(a, t):
    return (np.stack([np.interp(t, a[:, 0], a[:, i]) for i in (1, 2, 3)], axis=1),
            Slerp(a[:, 0], Rotation.from_quat(a[:, 4:8]))(t))


def gyro_steps(imu, t, bias):
    ti = (imu[:, 0] - imu[0, 0]) * 1e-9
    tq = t - imu[0, 0] * 1e-9
    static = ti < 1.5
    if static.sum() > 10 and imu[static, 1:4].std(axis=0).max() < 0.15:
        bias = imu[static, 4:7].mean(axis=0)
    w = imu[:, 4:7] - bias
    cum = np.vstack([np.zeros(3), np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(ti)[:, None], axis=0)])
    c = np.stack([np.interp(tq, ti, cum[:, d]) for d in range(3)], axis=1)
    return Rotation.from_rotvec(np.diff(c, axis=0))


def score(a, t, gyro, bstep, brot, endpoint_t=None, gyro_valid=None):
    p, r = interpolate(a, t)
    dt = np.diff(t)
    dp = np.diff(p, axis=0)
    step = np.linalg.norm(dp, axis=1)
    rel = r[:-1].inv() * r[1:]
    rot = np.degrees(rel.magnitude())
    vel = dp / dt[:, None]
    tv = (t[1:] + t[:-1]) * 0.5
    acc = np.diff(vel, axis=0) / np.diff(tv)[:, None]
    ta = (tv[1:] + tv[:-1]) * 0.5
    jerk = np.diff(acc, axis=0) / np.diff(ta)[:, None]
    ge = np.degrees((gyro.inv() * rel).magnitude())
    if gyro_valid is None:
        gyro_valid = np.ones(len(ge), dtype=bool)
    ge = ge[gyro_valid]
    endpoint_t = [t[0], t[-1]] if endpoint_t is None else endpoint_t
    ep, _ = interpolate(a, endpoint_t)
    return dict(closure_m=float(np.linalg.norm(ep[-1]-ep[0])),
                full_endpoint_m=float(np.linalg.norm(a[-1, 1:4] - a[0, 1:4])),
                endpoint_t_s=list(map(float, endpoint_t)),
                path_m=float(step.sum()), n_steps=len(step), duration_s=float(t[-1] - t[0]),
                max_speed=float((step / dt).max()), max_turn_rate=float((rot / dt).max()),
                acc_p95=float(np.quantile(np.linalg.norm(acc, axis=1), .95)),
                jerk_p95=float(np.quantile(np.linalg.norm(jerk, axis=1), .95)),
                gyro_p95_deg=float(np.quantile(ge, .95)) if len(ge) else None,
                gyro_bad=int((ge > .5).sum()), gyro_excluded_steps=int((~gyro_valid).sum()),
                tr_bad=int((step > 2 * bstep + .01).sum()),
                rot_bad=int((rot > 2 * brot + .5).sum()))


def bag(job):
    tag, sources, fe, data, bias = job
    date, name = tag.split('_', 1)
    basename = name.removeprefix('basalt_')
    candidates = [data / date / f'vio_basalt_{basename}.txt', data / date / name / f'vio_basalt_{basename}.txt']
    bf = next((p for p in candidates if p.exists()), None)
    if bf is None:
        raise FileNotFoundError(f'Missing comparison trajectory for {tag}')
    trajectories = {'basalt': load_trajectory(bf)}
    paths={'basalt':bf}
    for label, source in sources.items():
        if source == 'slam':
            p = bf.parent / f'slam_cuvslam_{basename}.txt'
        elif source.startswith('_'):
            p = fe / f'{tag}_ref{source}.txt'
        else:
            p = Path(source.format(tag=tag))
        trajectories[label] = load_trajectory(p)
        paths[label]=p
    imu = np.load(fe / f'{tag}_fe.npz')['imu']
    endpoint_t = [max(a[0, 0] for a in trajectories.values()), min(a[-1, 0] for a in trajectories.values())]
    lo = max(endpoint_t[0], imu[0, 0] * 1e-9 + 2)
    hi = min(endpoint_t[1], imu[-1, 0] * 1e-9)
    t = trajectories['basalt'][:, 0]
    t = t[(t >= lo) & (t <= hi)]
    if len(t) < 4:
        raise ValueError(f'Insufficient common overlap for {tag}')
    g = gyro_steps(imu, t, bias)
    from imu_constraints24 import covered_gyro_intervals
    _, gyro_valid, _ = covered_gyro_intervals(imu, np.rint(t*1e9).astype(np.int64), bias)
    bp, br = interpolate(trajectories['basalt'], t)
    bs = np.linalg.norm(np.diff(bp, axis=0), axis=1)
    brot = np.degrees((br[:-1].inv() * br[1:]).magnitude())
    return dict(tag=tag, metrics={label: score(a, t, g, bs, brot, endpoint_t, gyro_valid) for label, a in trajectories.items()},
                input_sha256={label:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for label,path in paths.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fe-dir', type=Path, default=Path('results/fe5'))
    ap.add_argument('--data', type=Path, default=Path('/mnt/f/data_set/slam'))
    ap.add_argument('--calib', type=Path, default=Path('/mnt/f/data_set/slam/calibration_front_small.json'))
    ap.add_argument('--source', action='append', required=True, help='LABEL=SUFFIX or LABEL=path/{tag}.txt')
    ap.add_argument('--only', default='')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sources = dict(s.split('=', 1) for s in args.source)
    if 'basalt' in sources or len(sources) != len(args.source):
        ap.error('Source labels must be unique and must not be basalt')
    calib = json.loads(args.calib.read_text())['value0']
    tags = [p.name[:-7] for p in sorted(args.fe_dir.glob('*_fe.npz')) if args.only in p.name]
    if not tags:
        ap.error('No input bags found')
    jobs = [(tag, sources, args.fe_dir, args.data, np.array(calib['calib_gyro_bias'][:3])) for tag in tags]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(bag, jobs))
    summary = {}
    for label in ['basalt', *sources]:
        r = [row['metrics'][label] for row in rows]
        closure = np.array([a['closure_m'] for a in r])
        summary[label] = dict(n=len(r), closure_mean_m=float(closure.mean()), closure_median_m=float(np.median(closure)),
                              closure_p90_m=float(np.quantile(closure, .9)), closure_max_m=float(closure.max()),
                              acc_p95_median=float(np.median([a['acc_p95'] for a in r])),
                              jerk_p95_median=float(np.median([a['jerk_p95'] for a in r])),
                              gyro_p95_median_deg=float(np.median([a['gyro_p95_deg'] for a in r if a['gyro_p95_deg'] is not None])),
                              **{k: sum(a[k] for a in r) for k in ['n_steps', 'tr_bad', 'rot_bad', 'gyro_bad', 'gyro_excluded_steps']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, bags=rows, sources=sources,
                                           calibration_sha256=hashlib.sha256(args.calib.read_bytes()).hexdigest(),
                                           note='closure_m is endpoint separation on the common trajectory interval; full_endpoint_m uses each original file. Neither is absolute error. Physical return only known approximately within 0.2m. Gyro gaps >15ms excluded. Basalt is not ground truth.'), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
