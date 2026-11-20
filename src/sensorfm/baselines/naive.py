"""Seasonal-naive forecaster with empirical quantiles.

Point forecast: repeat the last season, ``y[t+h] = y[t+h-m*k]`` with the smallest k >= 1 that
reaches into the context. Quantiles: the point forecast plus empirical quantiles of the
seasonal-naive errors observed *inside the context* — a calibrated-by-construction, assumption-free
baseline that any foundation model must beat. It is also the MASE scale, so its MASE is ~1 by
definition on in-sample data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


def seasonal_naive(context: npt.ArrayLike, horizon: int, season: int) -> F64:
    x = np.asarray(context, np.float64)
    m = season if len(x) >= season else 1
    reps = int(np.ceil(horizon / m))
    out: F64 = np.tile(x[-m:], reps)[:horizon]
    return out


def seasonal_naive_quantiles(
    context: npt.ArrayLike, horizon: int, season: int, quantiles: tuple[float, ...]
) -> F64:
    """(horizon, Q) quantile forecast."""
    x = np.asarray(context, np.float64)
    point = seasonal_naive(x, horizon, season)
    m = season if len(x) > season else 1
    resid = x[m:] - x[:-m]
    if len(resid) < 10:
        resid = np.diff(x) if len(x) > 2 else np.zeros(1)
    qs = np.quantile(resid, quantiles)
    # Errors compound with the number of seasons ahead: scale by sqrt(seasons ahead).
    k = 1 + np.arange(horizon) // m
    out: F64 = point[:, None] + np.sqrt(k)[:, None] * qs[None, :]
    return out
