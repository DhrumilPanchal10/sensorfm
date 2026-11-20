#!/usr/bin/env bash
# Task 4.4 ablations (each = short pretrain + C-MAPSS fine-tune), sized for the free quota.
# Every ablation pretrains a small model for ABL_STEPS steps so the comparisons are like-for-like;
# E1 (pretrained vs scratch), E2 (size) and E7 (regimes) come from the main runs and are not
# re-run here.
#   ABL_STEPS=4000 bash kaggle/run_ablations.sh E3 E4 E6
set -euo pipefail
source kaggle/gpu_setup.sh
STEPS=${ABL_STEPS:-4000}; MODEL=${MODEL:-s20m}

declare -A RUNS=(
  [E3_p8]="model.patch_len=8 model.patch_stride=8"
  [E3_p32]="model.patch_len=32 model.patch_stride=32"
  [E4_m02]="objective.mask_ratio=0.2"
  [E4_m06]="objective.mask_ratio=0.6"
  [E6_a0]="objective.alpha=0.0"
  [E6_a1]="objective.alpha=1.0"
)
# E5 (synthetic mix) needs a second shard build with --synthetic-frac 0.15 (CPU session):
#   python scripts/build_corpus.py --synthetic-frac 0.15 ... --out /kaggle/working/shards/syn15
# then: SHARD_DIR=.../syn15 ABL=E5 bash kaggle/run_ablations.sh E5
RUNS[E5_syn15]=""

WANT=("$@")
n=0; for k in "${!RUNS[@]}"; do for w in "${WANT[@]:-all}"; do [[ $w == all || $k == $w* ]] && n=$((n+1)); done; done
python scripts/estimate_budget.py --model "$MODEL" --steps "$STEPS" --micro-batch 64 \
  --assume-tokens-per-s 100000 >/dev/null || true
python -c "from sensorfm.utils.budget import estimate; estimate(0.75 * $n, 1, '$n ablation runs x ($STEPS pretrain steps + FD001 fine-tune)')"

for k in "${!RUNS[@]}"; do
  sel=0; for w in "${WANT[@]:-all}"; do [[ $w == all || $k == $w* ]] && sel=1; done
  [ $sel -eq 1 ] || continue
  python -m sensorfm.pretrain.train model="$MODEL" data=pretrain_main train=single_gpu \
    train.max_steps="$STEPS" run_id="abl_$k" ${RUNS[$k]}
  CK=$(ls -d "runs/abl_$k/checkpoints/step_"* | tail -1)
  python scripts/eval_rul.py --tag "abl_$k" --ckpt "$CK" --subsets FD001 --regimes full,few_shot \
    --root "$DOWNSTREAM_ROOT/cmapss"
done
mkdir -p /kaggle/working/results && cp -r results/. /kaggle/working/results/
