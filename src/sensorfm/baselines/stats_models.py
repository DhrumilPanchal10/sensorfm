"""Classical statistical baseline: ETS (exponential smoothing state-space model).

ETS(A, Ad, A): additive errors, additive damped trend, additive seasonality with the dataset's
season length, refit on every forecast context (as a foundation model sees only the context).
Quantiles come from the model's analytic prediction intervals (Gaussian errors), converted to the
quantile grid via the interval half-width. Fits that fail or diverge fall back to seasonal naive;
the fallback count is reported so the baseline is never silently weakened. ARIMA is not run on
the full benchmark: auto-ARIMA refit per window is ~100x slower than ETS on these context lengths
for no expected gain on strongly seasonal sensor data (documented deviation from plan §4.1).
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from sensorfm.baselines.naive import seasonal_naive_quantiles

F64 = npt.NDArray[np.float64]


def ets_quantiles(
    context: npt.ArrayLike, horizon: int, season: int, quantiles: tuple[float, ...]
) -> tuple[F64, bool]:
    """(horizon, Q) quantile forecast and whether ETS succeeded (False -> naive fallback)."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    x = np.asarray(context, np.float64)
    use_season = season if len(x) >= 2 * season else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pandas input: statsmodels 0.15 get_prediction crashes on plain ndarrays
            # ('numpy.ndarray' object has no attribute 'index').
            model = ETSModel(
                pd.Series(x),
                error="add",
                trend="add",
                damped_trend=True,
                seasonal="add" if use_season else None,
                seasonal_periods=use_season,
            )
            res = model.fit(disp=False, maxiter=200)
            pred = res.get_prediction(start=len(x), end=len(x) + horizon - 1)
            frame = pred.summary_frame(alpha=0.2)
        mean = frame["mean"].to_numpy()
        half = (frame["pi_upper"].to_numpy() - frame["pi_lower"].to_numpy()) / 2.0
        sigma = half / norm.ppf(0.9)
        out: F64 = mean[:, None] + sigma[:, None] * norm.ppf(np.asarray(quantiles))[None, :]
        scale = np.abs(x).max() + 1e-9
        if not np.all(np.isfinite(out)) or np.abs(out).max() > 100 * scale:
            raise ValueError("diverged")
        return out, True
    except Exception:
        return seasonal_naive_quantiles(x, horizon, season, quantiles), False
