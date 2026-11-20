"""Check an exported model against PyTorch: same inputs, outputs within tolerance.

FP32 ONNX should match to ~1e-4 relative; INT8 variants are *not* expected to match closely —
their acceptance criterion is the task-metric delta (Task 5.2), so this module also reports the
error statistics instead of only pass/fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn


def ort_session(path: Path, threads: int = 0) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_ort(sess: Any, inputs: dict[str, npt.NDArray[Any]]) -> npt.NDArray[np.float32]:
    out: npt.NDArray[np.float32] = sess.run(None, inputs)[0]
    return out


def compare(
    module: nn.Module, onnx_path: Path, inputs: dict[str, npt.NDArray[Any]]
) -> dict[str, float]:
    module.eval()
    with torch.no_grad():
        ref = module(*(torch.as_tensor(v) for v in inputs.values())).numpy()
    got = run_ort(ort_session(onnx_path), inputs)
    diff = np.abs(ref - got)
    denom = np.abs(ref).mean() + 1e-12
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_mean": float(diff.mean() / denom),
    }
