#!/usr/bin/env bash
# Pretraining on Kaggle (1 GPU, or both T4s with NPROC=2). Resumable across sessions:
#   MODEL=s20m STEPS=20000 RUN_ID=s20m_main bash kaggle/pretrain.sh            # first session
#   MODEL=s20m STEPS=20000 RUN_ID=s20m_main RESUME=true bash kaggle/pretrain.sh # later sessions
# Attach the previous session's output (runs/) as input and copy it back before resuming.
set -euo pipefail
source kaggle/gpu_setup.sh
MODEL=${MODEL:-s20m}; STEPS=${STEPS:-20000}; MB=${MB:-64}; NPROC=${NPROC:-1}
RUN_ID=${RUN_ID:-${MODEL}_$(date +%Y%m%d)}; RESUME=${RESUME:-false}
STRATEGY=${STRATEGY:-$([ "$NPROC" -gt 1 ] && echo ddp || echo none)}
WALL=${WALL_MIN:-690}  # stop cleanly with a checkpoint before Kaggle's 12 h session limit

if [ "$RESUME" = true ] && [ ! -d "runs/$RUN_ID" ]; then  # previous session attached as input
  PREV=$(find -L /kaggle/input -maxdepth 6 -type d -path "*/runs/$RUN_ID" 2>/dev/null | head -1 || true)
  [ -n "$PREV" ] || { echo "RESUME=true but no runs/$RUN_ID under /kaggle/input" >&2; exit 1; }
  mkdir -p runs && cp -r "$PREV" "runs/$RUN_ID" && ls "runs/$RUN_ID/checkpoints"
fi
python scripts/estimate_budget.py --model "$MODEL" --steps "$STEPS" --micro-batch "$MB" --gpus "$NPROC"

ARGS=(model="$MODEL" data=pretrain_main train=single_gpu train.strategy="$STRATEGY"
      train.max_steps="$STEPS" train.micro_batch="$MB" run_id="$RUN_ID" resume="$RESUME"
      max_wall_minutes="$WALL" use_mlflow=true)
if [ "$NPROC" -gt 1 ]; then
  torchrun --nproc_per_node="$NPROC" -m sensorfm.pretrain.train "${ARGS[@]}"
else
  python -m sensorfm.pretrain.train "${ARGS[@]}"
fi
mkdir -p /kaggle/working/runs && cp -r "runs/$RUN_ID" /kaggle/working/runs/
