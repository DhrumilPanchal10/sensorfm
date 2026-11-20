"""UCR anomaly scoring with forecast error.

For every position in the test part, the model forecasts the next patch from the preceding
context (stride = one patch); the anomaly score of a point is its absolute error divided by the
predicted spread (q0.9 - q0.1), i.e. "how surprising is this value given what the model expected".
The raw score is then smoothed with a moving average of one patch and normalized by a rolling
median/MAD (*rolling normalization*, plan Task 4.2) so slow regime changes don't dominate. The
predicted anomaly location is the argmax over the test part (UCR protocol: exactly one anomaly).

``finetune_on_prefix`` optionally adapts the model to the anomaly-free training prefix first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from sensorfm.finetune.forecast import finetune_forecaster, forecast_quantiles
from sensorfm.model.sensorfm import SensorFM

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


def rolling_robust_z(x: F64, window: int) -> F64:
    """(x - rolling median) / rolling MAD over a trailing window (NaN-safe)."""
    from pandas import Series

    s = Series(x)
    med = s.rolling(window, min_periods=max(5, window // 4)).median()
    mad = (s - med).abs().rolling(window, min_periods=max(5, window // 4)).median()
    z: F64 = ((s - med) / (1.4826 * mad + 1e-9)).to_numpy(np.float64)
    return z


def anomaly_scores(
    model: SensorFM,
    values: F32,
    train_end: int,
    device: torch.device,
    context: int | None = None,
    norm_window: int = 2000,
) -> F64:
    """Per-point score (length n, NaN where undefined). 0-based indexing."""
    p = model.cfg.patch_len
    ctx_len = context or (model.cfg.context_length - p)
    n = len(values)
    start = max(ctx_len, train_end - ctx_len)  # include a stretch of training data for warm-up
    origins = np.arange(start, n - p + 1, p)
    if len(origins) == 0:
        return np.full(n, np.nan)
    ctx = np.stack([values[o - ctx_len : o] for o in origins]).astype(np.float32)
    q = forecast_quantiles(model, ctx, p, device)  # (k, P, Q)
    nq = q.shape[-1]
    med, lo, hi = q[..., nq // 2], q[..., 0], q[..., -1]
    actual = np.stack([values[o : o + p] for o in origins])
    err = np.abs(actual - med) / (np.abs(hi - lo) + 1e-6 * (np.abs(med).mean() + 1.0))
    raw = np.full(n, np.nan)
    for o, e in zip(origins, err, strict=True):
        raw[o : o + p] = e
    smooth = np.convolve(np.nan_to_num(raw, nan=0.0), np.ones(p) / p, mode="same")
    smooth[np.isnan(raw)] = np.nan
    return rolling_robust_z(smooth, norm_window)


def finetune_on_prefix(
    model: SensorFM, values: F32, train_end: int, steps: int, seed: int, device: torch.device
) -> list[float]:
    return finetune_forecaster(
        model, [values[:train_end]], model.cfg.patch_len, steps, seed, device
    )
