#!/usr/bin/env bash
# Common setup for Kaggle GPU sessions. Source it from the repo root:
#   source kaggle/gpu_setup.sh
# Expects the data built by kaggle/build_data.sh attached as a Kaggle Dataset input.
set -euo pipefail

DATA_IN=${DATA_IN:-$(ls -d /kaggle/input/*/shards/main 2>/dev/null | head -1 || true)}
if [ -z "${DATA_IN}" ]; then
  echo "No shards found under /kaggle/input/*/shards/main — attach the sensorfm-data dataset." >&2
  return 1 2>/dev/null || exit 1
fi
export SHARD_DIR="$DATA_IN"
export DOWNSTREAM_ROOT=$(dirname "$(dirname "$DATA_IN")")/downstream
export GPU_PRICE_PER_HOUR=0
export MLFLOW_TRACKING_URI=file:/kaggle/working/mlruns

if [ ! -x .venv/bin/python ]; then  # idempotent: notebooks source this more than once
  pip install -q uv
  # Reuse the image's CUDA torch (multi-GB) instead of downloading it every session.
  uv venv -q --system-site-packages .venv
  uv pip install -q --python .venv/bin/python -e ".[data,track,eval,export]"
fi
export PATH="$PWD/.venv/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpus',torch.cuda.device_count())"
