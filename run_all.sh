#!/bin/bash
# Run cuVSLAM stereo+SLAM on every bag under DATA, JOBS at a time, then summarise.
# Usage: ./run_all.sh [DATA=/home/cj/data/slam] [JOBS=3] [extra run_bag.py args...]
set -u
DATA=${1:-/home/cj/data/slam}
JOBS=${2:-3}
shift $(( $# < 2 ? $# : 2 ))
OUT=${OUT:-results/all}
cd "$(dirname "$0")"
mkdir -p $OUT/logs

run_one() {
  bag=$1; shift
  tag="$(basename "$(dirname "$bag")")_$(basename "$bag")"
  ls $OUT/${tag}_*_summary.json >/dev/null 2>&1 && { echo "skip $tag"; return; }
  echo "start $tag"
  conda run --no-capture-output -n cuvslam python run_bag.py "$bag" --slam --calib "$DATA/calibration_front_small.json" \
    --out $OUT "$@" > $OUT/logs/$tag.log 2>&1 && echo "done  $tag" || echo "FAIL  $tag (see $OUT/logs/$tag.log)"
}
export -f run_one
export OUT DATA

find "$DATA" -name metadata.yaml -printf '%h\n' | sort | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "$@"
conda run --no-capture-output -n cuvslam python summarize.py $OUT
