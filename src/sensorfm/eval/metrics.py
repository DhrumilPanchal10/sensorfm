"""Task metrics. One implementation each, shared by baselines, fine-tuning and deployment checks.

- RUL: RMSE and the NASA/PHM08 asymmetric score (late predictions penalized more than early ones,
  because predicting failure too late is the dangerous error).
- Forecasting: MASE (scaled by in-sample seasonal-naive MAE, so it is comparable across series and
  datasets) and CRPS approximated by the mean pinball loss over quantiles 0.1..0.9 (times 2, which
  makes it converge to CRPS as the quantile grid gets dense). ``wql`` is the scale-free weighted
  variant used by Chronos-style leaderboards.
- Classification: macro-F1 (bearing classes are balanced-ish but we don't want accuracy to hide a
  class that is never predicted).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ArrayF = npt.ArrayLike
QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def rmse(pred: ArrayF, true: ArrayF) -> float:
    p, t = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def nasa_score(pred: ArrayF, true: ArrayF) -> float:
    """Sum over units of exp(-d/13)-1 for early (d<0) and exp(d/10)-1 for late (d>=0)."""
    d = np.asarray(pred, np.float64) - np.asarray(true, np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def seasonal_naive_scale(insample: ArrayF, season: int) -> float:
    x = np.asarray(insample, np.float64)
    m = season if len(x) > season else 1
    scale = float(np.mean(np.abs(x[m:] - x[:-m])))
    return scale if scale > 0 else 1.0


def mase(pred: ArrayF, true: ArrayF, insample: ArrayF, season: int) -> float:
    err = np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(true, np.float64)))
    return float(err / seasonal_naive_scale(insample, season))


def pinball(
    qpred: ArrayF, true: ArrayF, quantiles: tuple[float, ...] = QUANTILES
) -> npt.NDArray[np.float64]:
    """Pinball loss per quantile; ``qpred`` shape (..., Q), ``true`` shape (...)."""
    qp = np.asarray(qpred, np.float64)
    t = np.asarray(true, np.float64)[..., None]
    q = np.asarray(quantiles, np.float64)
    diff = t - qp
    loss: npt.NDArray[np.float64] = np.maximum(q * diff, (q - 1.0) * diff)
    return loss


def crps_quantile(qpred: ArrayF, true: ArrayF, quantiles: tuple[float, ...] = QUANTILES) -> float:
    """CRPS approximated as 2 x mean pinball loss over the quantile grid."""
    return float(2.0 * pinball(qpred, true, quantiles).mean())


def wql(qpred: ArrayF, true: ArrayF, quantiles: tuple[float, ...] = QUANTILES) -> float:
    """Weighted quantile loss: 2 * sum pinball / (Q * sum |y|)."""
    pl = pinball(qpred, true, quantiles)
    denom = np.abs(np.asarray(true, np.float64)).sum() * len(quantiles)
    return float(2.0 * pl.sum() / max(denom, 1e-12))


def macro_f1(pred: ArrayF, true: ArrayF, n_classes: int) -> float:
    p, t = np.asarray(pred), np.asarray(true)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((p == c) & (t == c))
        fp = np.sum((p == c) & (t != c))
        fn = np.sum((p != c) & (t == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))
