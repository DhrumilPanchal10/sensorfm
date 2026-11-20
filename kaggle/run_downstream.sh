#!/usr/bin/env bash
# Phase 4 on one Kaggle GPU: fine-tune/evaluate a pretrained checkpoint and its from-scratch twin,
# plus zero-shot foundation-model baselines on the same forecasting protocol.
#   CKPT=/kaggle/input/<runs-dataset>/runs/s20m_main/checkpoints/step_00020000 MODEL=s20m bash kaggle/run_downstream.sh
set -euo pipefail
source kaggle/gpu_setup.sh
: "${CKPT:?set CKPT to a pretraining checkpoint directory}"
MODEL=${MODEL:-s20m}; TAG=${TAG:-$MODEL}
uv pip install -q --python .venv/bin/python "chronos-forecasting>=2.3" || echo "chronos install failed (recorded as skipped)"

python -c "from sensorfm.utils.budget import estimate; estimate(4.0, 1, 'downstream: RUL 4 subsets x 3 regimes x 3 seeds (pre + scratch), bearing, forecasting, UCR')"

python scripts/eval_rul.py --tag "${TAG}_pre" --ckpt "$CKPT" --root "$DOWNSTREAM_ROOT/cmapss"
python scripts/eval_rul.py --tag "${TAG}_scratch" --model "$MODEL" --root "$DOWNSTREAM_ROOT/cmapss"
python scripts/eval_forecast_baselines.py --methods sensorfm,chronos2,chronos_bolt_small \
  --ckpt "$CKPT" --root "$DOWNSTREAM_ROOT/ett" --out "results/eval/forecast_${TAG}.json"
mkdir -p /kaggle/working/results && cp -r results/. /kaggle/working/results/
