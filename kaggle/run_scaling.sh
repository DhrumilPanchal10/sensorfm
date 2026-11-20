#!/usr/bin/env bash
# Task 3.3 scaling study on Kaggle "GPU T4 x2" (free). 1 vs 2 GPUs x {DDP, FSDP2} x {m60m, l150m}.
set -euo pipefail
source kaggle/gpu_setup.sh
MODELS=${MODELS:-"m60m l150m"}; MB=${MB:-32}

python -c "from sensorfm.utils.budget import estimate; estimate(0.6, 2, 'scaling study: 2 models x (1 GPU + DDP x2 + FSDP x2), 60 steps each')"

for M in $MODELS; do
  python scripts/scaling_bench.py --model "$M" --strategy none --shards "$SHARD_DIR" --micro-batch "$MB"
  for S in ddp fsdp; do
    torchrun --nproc_per_node=2 scripts/scaling_bench.py --model "$M" --strategy "$S" \
      --shards "$SHARD_DIR" --micro-batch "$MB"
  done
done
mkdir -p /kaggle/working/results && cp -r results/. /kaggle/working/results/
