#!/usr/bin/env bash
# Tasks 5.1/5.2 on one Kaggle GPU: fine-tune the deployed RUL model (FD001), export ONNX, ORT INT8
# variants (CPU latency on the Kaggle CPU), TensorRT FP16/INT8 engines (T4 latency).
#   CKPT=/kaggle/input/<...>/runs/s20m_main/checkpoints/step_N TAG=s20m bash kaggle/run_deploy.sh
set -euo pipefail
source kaggle/gpu_setup.sh
if [ -z "${CKPT:-}" ]; then  # newest checkpoint of the attached pretraining output
  CKPT=$(find -L /kaggle/input -maxdepth 8 -type d -path "*/runs/${MODEL:-s20m}_main/checkpoints/step_*" 2>/dev/null | sort | tail -1 || true)
fi
: "${CKPT:?set CKPT to a pretraining checkpoint directory}"
TAG=${TAG:-s20m}
python -c "from sensorfm.utils.budget import estimate; estimate(0.5, 1, 'deploy: fine-tune FD001 + ONNX/INT8 + TensorRT FP16/INT8 benchmarks')"
# TensorRT 11 is out (PyPI, checked 2026-11-22); trt_build.py uses the TRT 10 implicit-INT8
# calibrator API (deprecated in 10), so stay on 10.x until it is ported to explicit Q/DQ.
uv pip install -q --python .venv/bin/python onnx onnxruntime "tensorrt-cu12>=10,<11"
python -c "import tensorrt as t; print('tensorrt', t.__version__)"
python scripts/deploy.py --tag "$TAG" --ckpt "$CKPT" --root "$DOWNSTREAM_ROOT/cmapss" --gpu
mkdir -p /kaggle/working/results/bench /kaggle/working/exports
cp results/bench/deploy_"$TAG"_*.json /kaggle/working/results/bench/
cp exports/"$TAG"/rul.pt exports/"$TAG"/*.onnx /kaggle/working/exports/  # engines are GPU-specific
