#!/usr/bin/env bash
# Task 3.1/3.2 on one Kaggle GPU: optimization ladder + profiler traces for one model size.
#   MODEL=s20m bash kaggle/run_ladder.sh
set -euo pipefail
source kaggle/gpu_setup.sh
MODEL=${MODEL:-s20m}
STEPS=${STEPS:-300}

python -c "from sensorfm.utils.budget import estimate; estimate(0.5, 1, 'ladder ${MODEL}: 7 rungs x ${STEPS} steps + 2 noise seeds + profiles')"

python -m sensorfm.pretrain.bench --model "$MODEL" --shards "$SHARD_DIR" --steps "$STEPS" \
  --warmup 30 --micro-batch 64 --workers 2 --noise-seeds 2 --profile \
  --out "results/profiles/ladder_${MODEL}.json"
cp "results/profiles/ladder_${MODEL}.json" results/profiles/ladder.json

# Optional Nsight Systems timeline of the best rung, if nsys exists in the image.
if command -v nsys >/dev/null; then
  nsys profile -o "results/profiles/nsys_${MODEL}" --force-overwrite true \
    python -m sensorfm.pretrain.bench --model "$MODEL" --shards "$SHARD_DIR" --steps 60 \
      --warmup 10 --rungs 5 --out "results/profiles/nsys_${MODEL}_rung.json" || true
fi
mkdir -p /kaggle/working/results && cp -r results/. /kaggle/working/results/
