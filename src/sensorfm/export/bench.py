"""Latency timing shared by ``bench/latency.py`` and ``scripts/deploy.py`` (Task 5.2).

Every runtime (PyTorch, ONNX Runtime, TensorRT) is timed through the same ``time_fn`` so the
numbers are comparable: fixed inputs, warm-up iterations discarded, wall-clock per call, p50 / p99
and throughput at the given batch size. GPU runners must synchronize inside ``fn`` (the TensorRT
runner does), otherwise only the launch is timed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def time_fn(
    fn: Callable[[Any], Any], inputs: Any, batch: int, iters: int, warmup: int = 20
) -> dict[str, float]:
    for _ in range(warmup):
        fn(inputs)
    ts = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        fn(inputs)
        ts[i] = time.perf_counter() - t0
    ms = ts * 1000
    return {"p50_ms": float(np.percentile(ms, 50)), "p99_ms": float(np.percentile(ms, 99)),
            "throughput_per_s": float(batch * 1000 / ms.mean())}  # fmt: skip


def time_session(sess: Any, inputs: dict[str, Any], batch: int, iters: int) -> dict[str, float]:
    """ONNX Runtime session."""
    return time_fn(lambda inp: sess.run(None, inp), inputs, batch, iters)
