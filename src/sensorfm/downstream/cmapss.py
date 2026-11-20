"""NASA C-MAPSS turbofan remaining-useful-life (RUL) benchmark, FD001-FD004.

Source (verified 2026-11-20): NASA PCoE data repository, ``6. Turbofan Engine Degradation Simulation
Data Set.zip`` on ``phm-datasets.s3.amazonaws.com`` -> inner ``CMAPSSData.zip`` with
``{train,test,RUL}_FD00x.txt``. 26 space-separated columns: unit, cycle, 3 operational settings,
21 sensors. (The readme says FD004 has 248 train units; the file has 249 — we use the file.)

Standard protocol implemented here:
- **Labels:** piecewise-linear RUL, ``min(max_cycle - cycle, 125)`` on training units. Test targets
  are the official ``RUL_FD00x.txt`` values for the **last** cycle of each test unit, *uncapped* by
  default (``cap_test=True`` reproduces papers that also cap test targets; always reported).
- **Operating conditions:** FD002/FD004 run under 6 conditions. Rounding the settings to
  ``(op1: 0 dp, op2: 2 dp, op3: 0 dp)`` yields exactly 6 clusters on FD002/FD004 and 1 on
  FD001/FD003 (verified on the real files), so no KMeans is needed. Sensors are z-normalized
  **per condition** with statistics from the training units only.
- **Sensors:** sensors constant (within condition) on the training set are dropped.
- **Splits:** validation holds out whole training *units* (never windows).
- **Windows:** length ``W`` ending at each cycle; shorter histories are left-padded with a mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def read_txt(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def condition_ids(df: pd.DataFrame) -> npt.NDArray[np.object_]:
    """Operating-condition key per row from rounded settings (stable across splits)."""
    keys = (
        (df["op1"].round(0) + 0.0).astype(str)
        + "|" + (df["op2"].round(2) + 0.0).astype(str)
        + "|" + (df["op3"].round(0) + 0.0).astype(str)
    )  # fmt: skip
    return keys.to_numpy(dtype=object)


@dataclass
class CmapssData:
    subset: str
    sensors: list[str]
    train: pd.DataFrame  # normalized sensors + 'rul' (capped)
    test: pd.DataFrame  # normalized sensors
    test_rul: npt.NDArray[np.float32]  # true RUL at the last cycle of each test unit (sorted unit)


def load_subset(root: str | Path, subset: str, cap: int = RUL_CAP) -> CmapssData:
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset}")
    root = Path(root)
    for f in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
        if not (root / f).exists():
            raise FileNotFoundError(f"missing {root / f}; run scripts/prepare_downstream.py")
    train, test = read_txt(root / f"train_{subset}.txt"), read_txt(root / f"test_{subset}.txt")
    test_rul = np.loadtxt(root / f"RUL_{subset}.txt", dtype=np.float32).reshape(-1)
    if len(test_rul) != test["unit"].nunique():
        raise ValueError("RUL file length does not match number of test units")

    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = np.minimum(max_cycle - train["cycle"], cap).astype(np.float32)

    train["cond"], test["cond"] = condition_ids(train), condition_ids(test)
    unseen = set(test["cond"]) - set(train["cond"])
    if unseen:
        raise ValueError(f"test has operating conditions unseen in train: {unseen}")
    stats = train.groupby("cond")[SENSORS].agg(["mean", "std"])
    keep = [s for s in SENSORS if (stats[(s, "std")] > 1e-4).any()]
    for df in (train, test):
        mu = stats.loc[df["cond"], [(s, "mean") for s in keep]].to_numpy()
        sd = stats.loc[df["cond"], [(s, "std") for s in keep]].to_numpy()
        df[keep] = (df[keep].to_numpy() - mu) / np.where(sd > 1e-4, sd, 1.0)
    return CmapssData(subset, keep, train, test, test_rul)


def split_units(
    units: npt.ArrayLike, val_frac: float, seed: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Train/val split of engine *units* (never windows)."""
    u = np.unique(np.asarray(units))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(u)
    n_val = max(1, round(val_frac * len(u)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def unit_windows(
    df: pd.DataFrame, sensors: list[str], window: int, last_only: bool
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """Windows ``(n, C, W)``, masks ``(n, W)`` and unit ids, ending at each cycle (or the last)."""
    xs, ms, us = [], [], []
    for unit, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        arr = g[sensors].to_numpy(np.float32).T  # (C, T)
        T = arr.shape[1]
        ends = [T] if last_only else range(1, T + 1)
        for e in ends:
            s = max(0, e - window)
            w = np.zeros((len(sensors), window), np.float32)
            m = np.zeros(window, bool)
            w[:, window - (e - s) :] = arr[:, s:e]
            m[window - (e - s) :] = True
            xs.append(w)
            ms.append(m)
            us.append(int(np.asarray(unit).item()))
    return np.stack(xs), np.stack(ms), np.asarray(us, np.int64)


def train_windows(
    data: CmapssData, units: npt.ArrayLike, window: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    df = data.train[data.train["unit"].isin(np.asarray(units))]
    x, m, _ = unit_windows(df, data.sensors, window, last_only=False)
    y = df.sort_values(["unit", "cycle"])["rul"].to_numpy(np.float32)
    return x, m, y


def test_windows(
    data: CmapssData, window: int, cap_test: bool = False
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    x, m, _ = unit_windows(data.test, data.sensors, window, last_only=True)
    y = np.minimum(data.test_rul, RUL_CAP) if cap_test else data.test_rul
    return x, m, y.astype(np.float32)
