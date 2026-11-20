"""Paired bootstrap for "is model A really better than model B?".

Both models are scored on the *same* evaluation units (series, engines, ...). We resample units
with replacement and recompute the difference of the aggregate metric each time. Pairing matters:
per-unit difficulty varies far more than the model difference, and resampling the pairs together
cancels that shared variance. The claim "A < B" is reported only if the 95% interval of
``metric(A) - metric(B)`` excludes 0.

``aggregate`` lets non-mean metrics be bootstrapped correctly, e.g. RMSE = sqrt(mean(sq err)) is
computed from resampled squared errors rather than by averaging per-unit RMSEs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PairedResult:
    diff: float  # aggregate(A) - aggregate(B) on the full sample
    lo: float
    hi: float
    p_a_better: float  # fraction of resamples where A beats B (lower is better)
    n_units: int

    @property
    def significant(self) -> bool:
        return self.hi < 0 or self.lo > 0


def paired_bootstrap(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    aggregate: Callable[[F64], float] = lambda v: float(np.mean(v)),
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedResult:
    """``a``/``b``: per-unit values (e.g. squared errors or per-series MASE), aligned by unit."""
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("a and b must be 1-D and aligned per unit")
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.array([aggregate(x[i]) - aggregate(y[i]) for i in idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return PairedResult(aggregate(x) - aggregate(y), float(lo), float(hi),
                        float(np.mean(diffs < 0)), n)  # fmt: skip


def rmse_from_sq(v: F64) -> float:
    return float(np.sqrt(np.mean(v)))
