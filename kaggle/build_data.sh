#!/usr/bin/env bash
# Build downstream data, decontamination reference, and pretraining shards on Kaggle.
# Run from the repo root inside a Kaggle notebook (Internet ON). CPU session is enough.
set -euo pipefail

WORK=${WORK:-/kaggle/working}
TMP=${TMP_CACHE:-/kaggle/tmp/hf}
mkdir -p "$WORK" "$TMP"

pip install -q uv
uv venv -q --system-site-packages .venv
uv pip install -q -e ".[data]" --python .venv/bin/python
PY=.venv/bin/python

echo "== downstream datasets + reference =="
$PY scripts/prepare_downstream.py --root "$WORK/downstream"

echo "== pretraining shards =="
$PY scripts/build_corpus.py \
  --selection configs/data/lotsa_selection.yaml \
  --reference "$WORK/downstream/reference" \
  --cache-dir "$TMP" \
  --out "$WORK/shards/main"

mkdir -p "$WORK/results"
cp -r results/. "$WORK/results/"
echo "done. manifest: $WORK/shards/main/manifest.json"
