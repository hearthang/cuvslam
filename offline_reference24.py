"""Run the frozen offline reference generator from the nv_slam repository root."""
from pathlib import Path
import argparse
import runpy

if __name__ == '__main__':
    # A subset run must not replace the manifest of an existing full delivery.
    guard=argparse.ArgumentParser(add_help=False)
    guard.add_argument('--repo',type=Path,default=Path.cwd())
    guard.add_argument('--output',type=Path,default=Path('results/reference_v2'))
    guard.add_argument('--only',default='')
    args,_=guard.parse_known_args()
    output=args.output if args.output.is_absolute() else args.repo/args.output
    if args.only and (output/'pipeline.json').exists():
        raise SystemExit('Subset --only requires a new output folder; existing manifests are retained.')
    runpy.run_path(str(Path(__file__).resolve().parent/'reference24'/'offline_reference24.py'),run_name='__main__')
