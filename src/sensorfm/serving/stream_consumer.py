"""Streaming consumer: readings -> per-asset windows -> batched ONNX scoring -> alerts + drift.

Debug mode uses an in-process ``asyncio.Queue`` and a simulated producer that replays held-out
data (a C-MAPSS test engine cycle by cycle, or a UCR series point by point). A Kafka/Redpanda
source can replace the queue without touching the scoring path (same ``Reading`` messages).

Scorers run ONNX Runtime sessions (INT8 by default) — no PyTorch at serving time:

- ``RulScorer``: window ``(C, W)`` -> remaining useful life; alert when RUL < threshold.
- ``AnomalyScorer``: window of ``context + P`` points; forecasts the last P points from the first
  ``context`` and scores them by |actual - median| / (q90 - q10); alert when the score exceeds a
  threshold calibrated on the anomaly-free training prefix.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from sensorfm.monitoring.drift import FEATURES, DriftMonitor
from sensorfm.serving.metrics import ServiceMetrics
from sensorfm.serving.windowing import SlidingWindows, Window

F32 = npt.NDArray[np.float32]


@dataclass
class Reading:
    asset: str
    values: list[float]
    ts: float = 0.0


@dataclass
class Alert:
    asset: str
    task: str
    score: float
    end_index: int
    message: str


class RulScorer:
    task = "rul"

    def __init__(self, session: Any, threshold: float = 30.0) -> None:
        self.sess = session
        self.threshold = threshold

    def score(self, windows: list[Window]) -> F32:
        x = np.stack([w.values for w in windows]).astype(np.float32)
        mask = np.ones((len(windows), x.shape[-1]), dtype=bool)
        out: F32 = self.sess.run(None, {"x": x, "mask": mask})[0].reshape(-1)
        return out

    def is_alert(self, score: float) -> bool:
        return score < self.threshold


class AnomalyScorer:
    task = "anomaly"

    def __init__(self, session: Any, context: int, horizon: int, threshold: float) -> None:
        self.sess = session
        self.context = context
        self.horizon = horizon
        self.threshold = threshold

    def score(self, windows: list[Window]) -> F32:
        v = np.stack([w.values[0] for w in windows]).astype(np.float32)  # (B, c + P)
        q = self.sess.run(None, {"context": v[:, : self.context]})[0]  # (B, P, Q)
        nq = q.shape[-1]
        actual = v[:, self.context :]
        err = np.abs(actual - q[..., nq // 2]) / (np.abs(q[..., -1] - q[..., 0]) + 1e-6)
        out: F32 = err.mean(-1).astype(np.float32)
        return out

    def is_alert(self, score: float) -> bool:
        return score > self.threshold


@dataclass
class StreamConsumer:
    scorer: RulScorer | AnomalyScorer
    windows: SlidingWindows
    metrics: ServiceMetrics
    drift: DriftMonitor | None = None
    max_batch: int = 32
    alerts: list[Alert] = field(default_factory=list)
    scored: int = 0

    def _score(self, batch: list[Window]) -> None:
        t0 = time.perf_counter()
        scores = self.scorer.score(batch)
        self.metrics.latency.observe(time.perf_counter() - t0)
        self.metrics.windows.labels(self.scorer.task).inc(len(batch))
        self.scored += len(batch)
        for w, s in zip(batch, scores, strict=True):
            if self.drift is not None:
                self.drift.update(w.values[0])
            if self.scorer.is_alert(float(s)):
                msg = f"{self.scorer.task} alert on {w.asset}: score {float(s):.2f}"
                a = Alert(w.asset, self.scorer.task, float(s), w.end_index, msg)
                self.alerts.append(a)
                self.metrics.alerts.labels(self.scorer.task).inc()
        if self.drift is not None:
            for f, v in self.drift.scores().items():
                self.metrics.drift.labels(f).set(v)

    async def run(self, source: AsyncIterator[Reading]) -> None:
        pending: list[Window] = []
        async for r in source:
            w = self.windows.push(r.asset, r.values)
            if w is not None:
                pending.append(w)
            if len(pending) >= self.max_batch:
                self._score(pending)
                pending = []
        if pending:
            self._score(pending)

    def drift_scores(self) -> dict[str, float]:
        return self.drift.scores() if self.drift else {f: 0.0 for f in FEATURES}


async def queue_source(q: asyncio.Queue[Reading | None]) -> AsyncIterator[Reading]:
    while True:
        r = await q.get()
        if r is None:
            return
        yield r


async def replay(q: asyncio.Queue[Reading | None], asset: str, rows: Iterable[Iterable[float]],
                 rate_hz: float = 0.0) -> None:  # fmt: skip
    """Simulated producer: push rows (one reading each) then a sentinel."""
    for row in rows:
        await q.put(Reading(asset, [float(v) for v in row], time.time()))
        if rate_hz:
            await asyncio.sleep(1.0 / rate_hz)
    await q.put(None)
