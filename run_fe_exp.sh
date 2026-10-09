#!/bin/bash
# Front-end experiments on the local bag subset: ./run_fe_exp.sh NAME JOBS [run_frontend.py args...]
# Writes results/fe_exp/NAME/*_fe.npz (no keyframe images).
DATA=${DATA:-/home/cj/data/slam_sub}; NAME=$1; JOBS=$2; shift 2
OUT=results/fe_exp/$NAME; mkdir -p $OUT/logs
cd "$(dirname "$0")"
find $DATA -name metadata.yaml -printf '%h\n' | sort | xargs -P $JOBS -I{} bash -c \
  'b={}; tag=$(basename $(dirname $b))_$(basename $b); conda run --no-capture-output -n cuvslam python -u run_frontend.py $b \
   --calib '$DATA'/calibration_front_small.json --out '$OUT' --no-kf '"$*"' > '$OUT'/logs/$tag.log 2>&1 || echo FAIL $tag'
echo "done $NAME"
