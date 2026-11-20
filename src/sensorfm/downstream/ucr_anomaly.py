"""UCR Time Series Anomaly Archive (Wu & Keogh; KDD Cup 2021), 250 univariate series.

Source (verified 2026-11-20): ``https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
UCR_TimeSeriesAnomalyDatasets2021.zip`` -> ``.../FilesAreInHere/UCR_Anomaly_FullData/*.txt``.
File name: ``<id>_UCR_Anomaly_<name>_<trainEnd>_<begin>_<end>.txt``; points ``1..trainEnd`` are
anomaly-free training data, and exactly one anomaly lies in ``[begin, end]`` (1-based, inclusive).
Most files have one value per line; a few put all values on one line — both are parsed.

Scoring (verified from the archive's own ``UCR_AnomalyDataSets.pptx``, "Scoring Function"):
the algorithm outputs one location ``P`` per series (recommended: the anomaly *center*); with
``L = end - begin + 1`` it is correct iff
``min(begin - L, begin - 100) < P < max(end + L, end + 100)``.
Accuracy = fraction of the 250 series scored correct. **No point-adjusted F1** — that metric is
the main flaw the archive was created to avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
_NAME = re.compile(r"^(\d{3})_UCR_Anomaly_(.+)_(\d+)_(\d+)_(\d+)\.txt$")


@dataclass
class UcrSeries:
    id: int
    name: str
    train_end: int  # 1-based inclusive end of anomaly-free prefix
    begin: int  # 1-based inclusive
    end: int  # 1-based inclusive
    values: npt.NDArray[np.float32]

    @property
    def train(self) -> npt.NDArray[np.float32]:
        return self.values[: self.train_end]


def parse_name(filename: str) -> tuple[int, str, int, int, int]:
    m = _NAME.match(Path(filename).name)
    if not m:
        raise ValueError(f"not a UCR anomaly file name: {filename}")
    i, name, tr, b, e = m.groups()
    return int(i), name, int(tr), int(b), int(e)


def load_file(path: str | Path) -> UcrSeries:
    i, name, tr, b, e = parse_name(str(path))
    values = np.array(Path(path).read_text().split(), dtype=np.float32)
    if not (0 < tr < b <= e <= len(values)):
        raise ValueError(
            f"{path}: inconsistent indices train_end={tr} begin={b} end={e} n={len(values)}"
        )
    return UcrSeries(i, name, tr, b, e, values)


def load_archive(root: str | Path) -> list[UcrSeries]:
    files = sorted(Path(root).rglob("*_UCR_Anomaly_*.txt"))
    files = [f for f in files if "UCR_Anomaly_FullData" in str(f)] or files
    if not files:
        raise FileNotFoundError(
            f"no UCR anomaly files under {root}; run scripts/prepare_downstream.py"
        )
    return [load_file(f) for f in files]


def is_correct(pred_1based: int, begin: int, end: int) -> bool:
    """Official UCR rule (strict inequalities, 1-based indices)."""
    length = end - begin + 1
    return min(begin - length, begin - 100) < pred_1based < max(end + length, end + 100)


def predict_location(scores: npt.ArrayLike, train_end: int) -> int:
    """1-based location of the max anomaly score in the test part (``scores`` is 0-based, full
    length; NaN where undefined)."""
    s = np.asarray(scores, dtype=np.float64).copy()
    s[:train_end] = -np.inf
    s[~np.isfinite(s)] = -np.inf
    return int(np.argmax(s)) + 1


def accuracy(series: list[UcrSeries], preds_1based: dict[int, int]) -> float:
    if not series:
        raise ValueError("empty archive")
    hits = [is_correct(preds_1based[s.id], s.begin, s.end) for s in series]
    return float(np.mean(hits))
