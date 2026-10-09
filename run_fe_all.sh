#!/bin/bash
# Front end (run_frontend.py) over every bag in F:/data_set/slam, JOBS in parallel.  Usage: ./run_fe_all.sh [JOBS]
DATA=/mnt/f/data_set/slam; JOBS=${1:-10}; OUT=${OUT:-results/fe}
cd "$(dirname "$0")"; mkdir -p $OUT/logs
run_one() {
  bag=$1; tag="$(basename "$(dirname "$bag")")_$(basename "$bag")"
  [ -f $OUT/${tag}_fe.npz ] && { echo "skip $tag"; return; }
  conda run --no-capture-output -n cuvslam python -u run_frontend.py "$bag" --calib $DATA/calibration_front_small.json \
    --out $OUT > $OUT/logs/$tag.log 2>&1 && echo "done  $tag" || echo "FAIL  $tag"
}
export -f run_one; export OUT DATA
find $DATA -name metadata.yaml -printf '%h\n' | sort | xargs -P $JOBS -I{} bash -c 'run_one {}'
echo ALL_DONE
