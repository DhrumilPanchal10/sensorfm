"""Zero-shot public time-series foundation models, for the size-vs-quality comparison.

Verified 2026-11-20 on PyPI / Hugging Face: ``chronos-forecasting`` 2.3.2 exposes
``Chronos2Pipeline.from_pretrained("amazon/chronos-2")`` (Apache-2.0) with
``predict_df(df, prediction_length, quantile_levels, id_column, timestamp_column, target)``;
Chronos-Bolt checkpoints (``amazon/chronos-bolt-{tiny,mini,small,base}``) load through
``BaseChronosPipeline`` with ``predict_quantiles``. Other candidates (``uni2ts`` 2.0.0 for Moirai,
``granite-tsfm`` 0.3.12 for TTM, ``timesfm`` 3.0.2) exist on PyPI; their APIs are checked when they
are run on Kaggle, and any model that fails to install or run is recorded as skipped with the
error — never silently dropped (plan Task 4.1).

All wrappers take a batch of univariate contexts and return quantiles (n, horizon, Q) on the
original scale, matching ``sensorfm.finetune.forecast.forecast_quantiles``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]

PARAMS_M = {  # published parameter counts (millions), for the size axis of the comparison plot
    "chronos2": 120.0,
    "chronos_bolt_tiny": 9.0,
    "chronos_bolt_small": 48.0,
    "chronos_bolt_base": 205.0,
}


class Chronos2:
    name = "chronos2"

    def __init__(self, device: str = "cuda") -> None:
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def forecast(self, contexts: F32, horizon: int, quantiles: tuple[float, ...]) -> F32:
        import pandas as pd

        n, c = contexts.shape
        ts = pd.date_range("2000-01-01", periods=c, freq="h")
        df = pd.DataFrame({
            "id": np.repeat(np.arange(n), c),
            "timestamp": np.tile(ts, n),
            "target": contexts.reshape(-1).astype(np.float64),
        })  # fmt: skip
        pred: Any = self.pipe.predict_df(df, prediction_length=horizon,
                                         quantile_levels=list(quantiles), id_column="id",
                                         timestamp_column="timestamp", target="target")  # fmt: skip
        pred = pred.sort_values(["id", "timestamp"])
        cols = [str(q) for q in quantiles]
        missing = [c for c in cols if c not in pred.columns]
        if missing:
            raise RuntimeError(
                f"Chronos-2 output lacks quantile columns {missing}: {list(pred.columns)}"
            )
        return pred[cols].to_numpy(np.float32).reshape(n, horizon, len(quantiles))


class ChronosBolt:
    def __init__(self, size: str = "small", device: str = "cuda") -> None:
        from chronos import BaseChronosPipeline

        self.name = f"chronos_bolt_{size}"
        self.pipe = BaseChronosPipeline.from_pretrained(
            f"amazon/chronos-bolt-{size}", device_map=device
        )

    def forecast(self, contexts: F32, horizon: int, quantiles: tuple[float, ...]) -> F32:
        import torch

        q, _mean = self.pipe.predict_quantiles(torch.as_tensor(contexts), prediction_length=horizon,
                                               quantile_levels=list(quantiles))  # fmt: skip
        return q.cpu().numpy().astype(np.float32)


def load(name: str, device: str = "cuda") -> Chronos2 | ChronosBolt:
    if name == "chronos2":
        return Chronos2(device)
    if name.startswith("chronos_bolt_"):
        return ChronosBolt(name.removeprefix("chronos_bolt_"), device)
    raise ValueError(f"unknown foundation model {name!r}")
