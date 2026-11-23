#!/usr/bin/env bash
# Phase 4 on one Kaggle GPU: fine-tune/evaluate a pretrained checkpoint and its from-scratch twin,
# plus zero-shot foundation-model baselines on the same forecasting protocol.
#   CKPT=/kaggle/input/<runs-dataset>/runs/s20m_main/checkpoints/step_00020000 MODEL=s20m bash kaggle/run_downstream.sh
set -euo pipefail
source kaggle/gpu_setup.sh
if [ -z "${CKPT:-}" ]; then  # newest checkpoint of the attached pretraining output
  CKPT=$(find -L /kaggle/input -maxdepth 8 -type d -path "*/runs/${MODEL:-s20m}_main/checkpoints/step_*" 2>/dev/null | sort | tail -1 || true)
fi
: "${CKPT:?set CKPT to a pretraining checkpoint directory}"
MODEL=${MODEL:-s20m}; TAG=${TAG:-$MODEL}
uv pip install -q --python .venv/bin/python "chronos-forecasting>=2.3" || echo "chronos install failed (recorded as skipped)"

python -c "from sensorfm.utils.budget import estimate; estimate(4.0, 1, 'downstream: RUL 4 subsets x 3 regimes x 3 seeds (pre + scratch), bearing, forecasting, UCR')"

python scripts/eval_rul.py --tag "${TAG}_pre" --ckpt "$CKPT" --root "$DOWNSTREAM_ROOT/cmapss"
python scripts/eval_rul.py --tag "${TAG}_scratch" --model "$MODEL" --root "$DOWNSTREAM_ROOT/cmapss"
python scripts/eval_forecast_baselines.py --methods sensorfm,chronos2,chronos_bolt_small \
  --ckpt "$CKPT" --root "$DOWNSTREAM_ROOT/ett" --out "results/eval/forecast_${TAG}.json"
python scripts/eval_bearing.py --tag "${TAG}_pre" --ckpt "$CKPT" --root "$DOWNSTREAM_ROOT/cwru"
python scripts/eval_bearing.py --tag "${TAG}_scratch" --model "$MODEL" --root "$DOWNSTREAM_ROOT/cwru"
python scripts/eval_ucr.py --ckpt "$CKPT" --tag "${TAG}_zeroshot" --root "$DOWNSTREAM_ROOT/ucr"
python scripts/eval_ucr.py --ckpt "$CKPT" --tag "${TAG}_prefix200" --finetune-steps 200 --root "$DOWNSTREAM_ROOT/ucr"
python scripts/analyze_rul.py results/eval/cmapss_baselines.json "results/eval/rul_${TAG}_pre.json" \
  "results/eval/rul_${TAG}_scratch.json" --compare "${TAG}_pre/full:${TAG}_scratch/full" \
  --compare "${TAG}_pre/few_shot:${TAG}_scratch/few_shot" --compare "${TAG}_pre/full:gbm"
mkdir -p /kaggle/working/results && cp -r results/. /kaggle/working/results/
