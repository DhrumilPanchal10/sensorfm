"""Forecasting benchmark on public data excluded from pretraining.

Datasets: ETTh1, ETTh2, ETTm1, ETTm2 (electricity transformer temperature; Zhou et al. 2021;
``github.com/zhouhaoyi/ETDataset``, CC BY-ND 4.0). Verified 2026-11-20: no ETT subset exists in
LOTSA, and the window-hash decontamination removes any renamed copies anyway.

Protocol (the standard ETT one): chronological 12/4/4-month train/val/test split (hourly rows:
8640/2880/2880; 15-min rows x4). Each of the 7 columns is treated as a univariate series. Test
forecasts use rolling origins inside the test span with stride ``H``; the context may reach back
into earlier splits (as in the standard ETT loaders), the targets never do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
SEASON = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96}


def split_borders(name: str) -> tuple[int, int, int]:
    """End indices (exclusive) of train / val / test for the standard ETT split."""
    per_month = 30 * 24 * (4 if name.startswith("ETTm") else 1)
    return 12 * per_month, 16 * per_month, 20 * per_month


@dataclass
class ForecastSeries:
    dataset: str
    column: str
    values: npt.NDArray[np.float32]
    train_end: int
    val_end: int
    test_end: int
    season: int


def load_dataset(root: str | Path, name: str) -> list[ForecastSeries]:
    path = Path(root) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/prepare_downstream.py")
    df = pd.read_csv(path)
    tr, va, te = split_borders(name)
    if len(df) < te:
        raise ValueError(f"{name}: {len(df)} rows < expected {te}")
    return [
        ForecastSeries(name, c, df[c].to_numpy(np.float32), tr, va, te, SEASON[name])
        for c in df.columns
        if c != "date"
    ]


def rolling_windows(
    s: ForecastSeries, context: int, horizon: int, split: str = "test", stride: int | None = None
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """(contexts (n, context), targets (n, horizon)) with forecast origins inside ``split``."""
    lo, hi = {"val": (s.train_end, s.val_end), "test": (s.val_end, s.test_end)}[split]
    stride = stride or horizon
    origins = np.arange(max(lo, context), hi - horizon + 1, stride)
    if len(origins) == 0:
        raise ValueError("no forecast origins fit; reduce context/horizon")
    ctx = np.stack([s.values[o - context : o] for o in origins])
    tgt = np.stack([s.values[o : o + horizon] for o in origins])
    return ctx, tgt
