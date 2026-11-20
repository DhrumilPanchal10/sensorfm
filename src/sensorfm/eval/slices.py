"""Metric slices: break an aggregate down by domain, frequency, horizon, operating condition, or
true-RUL bucket, so a headline number can't hide a regime where the model fails (Task 4.3/4.5)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

RUL_BUCKETS = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 10_000))


def by_key(
    keys: Sequence[Any],
    pred: npt.ArrayLike,
    true: npt.ArrayLike,
    metric: Callable[[npt.NDArray[Any], npt.NDArray[Any]], float],
) -> dict[str, dict[str, float]]:
    p, t = np.asarray(pred), np.asarray(true)
    k = np.asarray([str(x) for x in keys])
    out = {}
    for key in sorted(set(k.tolist())):
        sel = k == key
        out[key] = {"n": int(sel.sum()), "metric": float(metric(p[sel], t[sel]))}
    return out


def rul_bucket(true_rul: npt.ArrayLike) -> list[str]:
    """Label each unit by its true-RUL bucket (late life = small RUL, where errors matter most)."""
    out = []
    for r in np.asarray(true_rul, np.float64):
        for lo, hi in RUL_BUCKETS:
            if lo <= r < hi:
                out.append(f"{lo}-{hi if hi < 10_000 else 'inf'}")
                break
    return out
