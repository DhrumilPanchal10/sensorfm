"""Step-time breakdown: how long the training loop waited for data vs computed.

"Data wait" is the time spent blocked in ``next(loader)``; if it is a noticeable fraction of the
step, the GPU is starved and optimizing the model is pointless until the loader is fixed (Task
3.1). CUDA work is asynchronous, so compute time is only meaningful after ``synchronize``; we sync
once per *logged* step, not every step, to avoid distorting throughput.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class StepTimer:
    device: torch.device
    data_s: float = 0.0
    compute_s: float = 0.0
    steps: int = 0
    _t: float = field(default_factory=time.perf_counter)

    def mark_data_start(self) -> None:
        self._t = time.perf_counter()

    def mark_data_end(self) -> None:
        now = time.perf_counter()
        self.data_s += now - self._t
        self._t = now

    def mark_step_end(self, sync: bool) -> None:
        if sync and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        self.compute_s += now - self._t
        self._t = now
        self.steps += 1

    def pop(self) -> dict[str, float]:
        total = self.data_s + self.compute_s
        out = {
            "time/step_s": total / max(1, self.steps),
            "time/data_frac": self.data_s / total if total > 0 else 0.0,
        }
        self.data_s = self.compute_s = 0.0
        self.steps = 0
        return out
