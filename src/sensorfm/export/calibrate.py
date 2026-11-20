"""Calibration data for INT8 (ORT static and TensorRT).

Representative windows are drawn from the *training* split of the downstream task (never test),
512 by default as in the plan. The TensorRT calibrator subclasses ``IInt8EntropyCalibrator2``
(KL-divergence range selection) and caches the computed scales so rebuilding an engine doesn't
recalibrate. GPU-only (needs ``tensorrt`` and CUDA via torch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def sample_calibration(
    arrays: dict[str, npt.NDArray[Any]], n: int = 512, seed: int = 0
) -> dict[str, npt.NDArray[Any]]:
    """Random subset of ``n`` rows (same rows for every input)."""
    total = len(next(iter(arrays.values())))
    idx = np.random.default_rng(seed).choice(total, size=min(n, total), replace=False)
    return {k: v[np.sort(idx)] for k, v in arrays.items()}


def make_trt_calibrator(inputs: dict[str, npt.NDArray[Any]], batch: int, cache: Path) -> Any:
    import tensorrt as trt
    import torch

    class Calibrator(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.names = list(inputs)
            self.n = len(inputs[self.names[0]])
            self.i = 0
            self.dev: dict[str, torch.Tensor] = {}

        def get_batch_size(self) -> int:
            return batch

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self.i + batch > self.n:
                return None
            ptrs = []
            for name in names:
                t = torch.as_tensor(inputs[name][self.i : self.i + batch]).cuda().contiguous()
                self.dev[name] = t  # keep alive until the next call
                ptrs.append(int(t.data_ptr()))
            self.i += batch
            return ptrs

        def read_calibration_cache(self) -> bytes | None:
            return cache.read_bytes() if cache.exists() else None

        def write_calibration_cache(self, data: bytes) -> None:
            cache.write_bytes(data)

    return Calibrator()
