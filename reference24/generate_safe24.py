"""Reproduce v1 settings with the isolated IMU interval coverage correction."""
import argparse,json
from pathlib import Path
import reference_candidate24 as R
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('tag');ap.add_argument('--suffix',default='_reference24safe')
args=ap.parse_args();cfg=json.load(open(Path('results/reference_v1/quality')/f'{args.tag}.json'))['config']
cfg.update(suffix=args.suffix,jobs=1);R.configure(cfg);R.run(args.tag)
