"""One-command offline reference generation with provenance, logs and resumable stages.

Generation never reads Basalt or enforces equal endpoints. Existing front-end
caches can be used with --fe-dir, or create them with one or more --bag options.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fe-dir', type=Path, required=True)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--bag', action='append', type=Path, default=[])
    ap.add_argument('--frontend-python', default=sys.executable)
    ap.add_argument('--backend-suffix', default='_reference')
    ap.add_argument('--reuse-backend', action='store_true', help='explicitly import an existing backend solution and loops')
    ap.add_argument('--resume', action='store_true', help='continue this exact run if its manifest and input hashes match')
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error('--jobs must be positive')
    root = Path(__file__).resolve().parent
    args.fe_dir, args.calib, args.output = [p.resolve() for p in (args.fe_dir, args.calib, args.output)]
    if not args.calib.is_file():
        ap.error('Calibration file does not exist')
    args.fe_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'; logs.mkdir(exist_ok=True)
    state_path = args.output / 'manifest.json'
    settings = {k: ([str(p.resolve()) for p in v] if k == 'bag' else str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items() if k != 'resume'}
    scripts = ['backend.py', 'run_frontend.py', 'augment_loops.py', 'reference.py', 'translation_refine.py', 'offline_reference.py']
    identity = dict(settings=settings, source_sha256={f: digest(root/f) for f in scripts}, calibration_sha256=digest(args.calib))
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not args.resume or state.get('identity') != identity:
            ap.error('Output belongs to another run. Use a new output directory, or --resume with unchanged code and settings.')
    else:
        state = dict(identity=identity, started=time.time(), stages={}, status='running')
        atomic_json(state_path, state)
        src = args.output / 'source'; src.mkdir(exist_ok=True)
        for f in scripts:
            shutil.copy2(root/f, src/f)
        shutil.copy2(args.calib, args.output/'calibration.json')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def stage(name, command, outputs):
        previous = state['stages'].get(name)
        if previous and previous.get('status') == 'complete':
            if all(p.exists() and previous['sha256'].get(str(p)) == digest(p) for p in outputs):
                print(f'{name}: verified cached result', flush=True)
                return
            raise RuntimeError(f'{name}: cached outputs changed; use a new output directory')
        entry = dict(command=[str(c) for c in command], status='running', started=time.time())
        state['stages'][name] = entry
        atomic_json(state_path, state)
        print(f'{name}: running', flush=True)
        with open(logs/f'{name}.log', 'w') as log:
            result = subprocess.run(entry['command'], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0 or not all(p.exists() for p in outputs):
            entry.update(status='failed', returncode=result.returncode)
            state['status'] = 'failed'; atomic_json(state_path, state)
            raise RuntimeError(f'{name} failed; see {logs/name}.log')
        entry.update(status='complete', finished=time.time(), sha256={str(p): digest(p) for p in outputs})
        atomic_json(state_path, state)

    for b in args.bag:
        b = b.resolve(); tag = f'{b.parent.name}_{b.name}'
        stage(f'frontend_{tag}', [args.frontend_python, root/'run_frontend.py', b, '--calib', args.calib, '--out', args.fe_dir],
              [args.fe_dir/f'{tag}_fe.npz', args.fe_dir/f'{tag}_kf.pkl'])
    tags = [p.name[:-7] for p in sorted(args.fe_dir.glob('*_fe.npz')) if args.only in p.name]
    if not tags:
        raise RuntimeError('No front-end caches found')
    inputs = {str(args.fe_dir/f'{tag}{suffix}'): digest(args.fe_dir/f'{tag}{suffix}')
              for tag in tags for suffix in ['_fe.npz', '_kf.pkl']}
    if state.get('input_sha256') not in (None, inputs):
        raise RuntimeError('Input data changed; use a new output directory')
    state['input_sha256'] = inputs
    atomic_json(state_path, state)
    common = [args.fe_dir, '--calib', args.calib, '--jobs', str(args.jobs), '--only', args.only]
    backend_outputs = [args.fe_dir/f'{tag}_{kind}{args.backend_suffix}.{ext}'
                       for tag in tags for kind, ext in [('be', 'txt'), ('loops', 'npz')]]
    if args.reuse_backend:
        imported = {str(p): digest(p) for p in backend_outputs}
        if state.get('imported_backend_sha256') not in (None, imported):
            raise RuntimeError('Imported backend changed; use a new output directory')
        state['imported_backend_sha256'] = imported
        atomic_json(state_path, state)
    else:
        stage('backend', [sys.executable, root/'backend.py', *common, '--odom', 'fe_in', '--suffix', args.backend_suffix], backend_outputs)
    # Name intermediates by run identity to keep simultaneous runs separate.
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    suffix = '_reference_' + token
    stage('visual_revisits', [sys.executable, root/'augment_loops.py', *common, '--suffix', suffix],
          [args.fe_dir/f'{tag}_extra{suffix}.npz' for tag in tags])
    stage('global_optimization', [sys.executable, root/'reference.py', *common, '--be-suffix', args.backend_suffix,
          '--extra-suffix', suffix, '--suffix', suffix, '--vo-t', '0.01', '--vo-r-deg', '0.15',
          '--gnc', '10', '2', '--position-acc', '1.0'],
          [args.fe_dir/f'{tag}_ref{suffix}.{ext}' for tag in tags for ext in ['txt', 'json']])
    # Validate before publishing standalone outputs. No comparison trajectory is required.
    import numpy as np
    published = {}
    traj_dir = args.output/'trajectories'; traj_dir.mkdir(exist_ok=True)
    quality_dir = args.output/'quality'; quality_dir.mkdir(exist_ok=True)
    for tag in tags:
        source = args.fe_dir/f'{tag}_ref{suffix}.txt'
        a = np.loadtxt(source, ndmin=2)
        if a.shape[1] != 8 or not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) <= 0) or not np.allclose(np.linalg.norm(a[:, 4:8], axis=1), 1, atol=1e-4):
            raise ValueError(f'Invalid generated trajectory: {tag}')
        dst = traj_dir/f'{tag}.txt'; shutil.copy2(source, dst)
        stats = json.loads((args.fe_dir/f'{tag}_ref{suffix}.json').read_text())
        stats['quality_note'] = 'Offline sensor-constrained reference, not ground truth. Handling assumes replacement at pickup location.'
        atomic_json(quality_dir/f'{tag}.json', stats)
        published[tag] = dict(path=str(dst.relative_to(args.output)), sha256=digest(dst), frames=len(a), closure_m=stats['closure_ref_m'])
    state.update(status='complete', finished=time.time(), trajectories=published)
    atomic_json(state_path, state)
    print(f'Completed {len(tags)} reference trajectories: {traj_dir}', flush=True)


if __name__ == '__main__':
    main()
