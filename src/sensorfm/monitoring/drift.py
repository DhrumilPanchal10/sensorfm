"""Input drift detection with the Population Stability Index (PSI).

For each scored window we compute three cheap, physically meaningful features — mean, standard
deviation, and spectral energy above the median frequency (vibration content) — and compare their
distribution in a recent sliding buffer to a reference distribution captured from training data.

    PSI = sum_bins (p_recent - p_ref) * ln(p_recent / p_ref)

with bins at reference deciles. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 major
shift. A shift in operating condition (e.g. switching C-MAPSS from FD001-like sea-level operation
to FD002's six conditions) moves mean/std and trips the alarm; the serving demo shows this.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
FEATURES = ("mean", "std", "hf_energy")


def window_features(w: npt.ArrayLike) -> F64:
    x = np.asarray(w, np.float64).reshape(-1)
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    half = len(spec) // 2
    hf = spec[half:].sum() / (spec.sum() + 1e-12)
    return np.array([x.mean(), x.std(), hf])


def psi(reference: F64, recent: F64, edges: F64, eps: float = 1e-4) -> float:
    p_ref = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)
    p_new = np.histogram(recent, bins=edges)[0] / max(len(recent), 1)
    p_ref, p_new = np.clip(p_ref, eps, None), np.clip(p_new, eps, None)
    return float(np.sum((p_new - p_ref) * np.log(p_new / p_ref)))


@dataclass
class DriftMonitor:
    reference: F64  # (n_ref, 3) features of training windows
    buffer_size: int = 500
    threshold: float = 0.25
    _recent: deque[F64] = field(default_factory=deque)
    _edges: list[F64] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.reference) < 20:
            raise ValueError("need >= 20 reference windows")
        for j in range(self.reference.shape[1]):
            e = np.unique(np.quantile(self.reference[:, j], np.linspace(0, 1, 11)))
            e[0], e[-1] = -np.inf, np.inf
            self._edges.append(e)

    @classmethod
    def from_windows(cls, windows: list[npt.ArrayLike], **kw: float) -> DriftMonitor:
        return cls(np.stack([window_features(w) for w in windows]), **kw)  # type: ignore[arg-type]

    def update(self, window: npt.ArrayLike) -> None:
        self._recent.append(window_features(window))
        while len(self._recent) > self.buffer_size:
            self._recent.popleft()

    def scores(self) -> dict[str, float]:
        if len(self._recent) < 20:
            return {f: 0.0 for f in FEATURES}
        rec = np.stack(list(self._recent))
        return {
            f: psi(self.reference[:, j], rec[:, j], self._edges[j]) for j, f in enumerate(FEATURES)
        }

    def drifted(self) -> bool:
        return max(self.scores().values()) > self.threshold
