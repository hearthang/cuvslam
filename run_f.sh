#!/bin/bash
# cuVSLAM over every bag in F:/data_set/slam for the Basalt comparison, JOBS in parallel.
# Job queue is ordered config-major so the primary config (A) finishes first.
# Usage: ./run_f.sh [JOBS=12]
set -u
DATA=/mnt/f/data_set/slam
JOBS=${1:-12}
OUT=results/f_cmp
cd "$(dirname "$0")"
mkdir -p $OUT/logs

run_one() {
  suffix=$1 bag=$2; shift 2
  tag="$(basename "$(dirname "$bag")")_$(basename "$bag")_$suffix"
  ls $OUT/*"$(basename "$bag")"*"$suffix"_summary.json >/dev/null 2>&1 && { echo "skip $tag"; return; }
  echo "start $tag"
  conda run --no-capture-output -n cuvslam python -u run_bag.py "$bag" --slam --calib "$DATA/calibration_front_small.json" \
    --out $OUT --tag-suffix "_$suffix" "$@" > $OUT/logs/$tag.log 2>&1 && echo "done  $tag" || echo "FAIL  $tag"
}
export -f run_one
export OUT DATA

bags=$(find "$DATA" -name metadata.yaml -printf '%h\n' | sort)
{
  for b in $bags; do echo "A $b --mode inertial --imu-noise-scale 0.3"; done
  for b in $bags; do echo "B $b --mode inertial --imu-noise-scale 0.3 --rig-frame level-cam"; done
  for b in $bags; do echo "C $b --mode stereo"; done
} | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _
echo ALL_DONE
