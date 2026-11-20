"""Cleaning rules applied to every raw series before sharding.

Rules (and why):
- Trim leading/trailing NaNs: they carry no signal and would bias the missing fraction.
- Drop if > ``max_missing_frac`` missing: heavily imputed series teach the model to draw lines.
- Linearly impute gaps of length <= ``max_gap``: short dropouts are common in real sensors and
  linear interpolation is the least-assuming fill.
- Split at gaps longer than ``max_gap`` instead of imputing them: a long straight line is an
  artifact, but the segments on either side are good data.
- Drop segments shorter than ``min_length`` (cannot form enough patches) and (near-)constant
  segments (no information; also produce zero std in RevIN).
- Drop non-finite values (inf) by treating them as missing.

Every drop is counted by reason so the dataset card can report it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


@dataclass
class DropStats:
    """Counts of series/segments dropped per reason plus kept totals."""

    reasons: Counter[str] = field(default_factory=Counter)
    kept_series: int = 0
    kept_obs: int = 0
    imputed_points: int = 0

    def merge(self, other: DropStats) -> None:
        self.reasons.update(other.reasons)
        self.kept_series += other.kept_series
        self.kept_obs += other.kept_obs
        self.imputed_points += other.imputed_points

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "kept_series": self.kept_series,
            "kept_obs": self.kept_obs,
            "imputed_points": self.imputed_points,
            "dropped": dict(sorted(self.reasons.items())),
        }


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Start/end (exclusive) of each run of True in ``mask``."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def is_constant(x: F32, rel_tol: float = 1e-6) -> bool:
    scale = max(float(np.max(np.abs(x))), 1e-12)
    return float(np.std(x)) <= rel_tol * scale


def clean_series(
    values: F32,
    min_length: int,
    max_missing_frac: float,
    max_gap: int,
    stats: DropStats | None = None,
) -> list[F32]:
    """Return zero or more clean, fully-observed float32 segments from one raw series."""
    st = stats if stats is not None else DropStats()
    x = np.asarray(values, dtype=np.float32).copy()
    x[~np.isfinite(x)] = np.nan
    obs = ~np.isnan(x)
    if not obs.any():
        st.reasons["all_missing"] += 1
        return []
    first, last = int(np.argmax(obs)), len(x) - int(np.argmax(obs[::-1]))
    x, obs = x[first:last], obs[first:last]
    if len(x) < min_length:
        st.reasons["too_short"] += 1
        return []
    if 1.0 - obs.mean() > max_missing_frac:
        st.reasons["too_missing"] += 1
        return []

    # Split at long gaps; impute short ones.
    segments: list[tuple[int, int]] = []
    start = 0
    for g0, g1 in _runs(~obs):
        if g1 - g0 > max_gap:
            segments.append((start, g0))
            start = g1
    segments.append((start, len(x)))

    out: list[F32] = []
    idx = np.arange(len(x))
    for s0, s1 in segments:
        seg, seg_obs = x[s0:s1], obs[s0:s1]
        if len(seg) < min_length:
            st.reasons["segment_too_short"] += 1
            continue
        n_missing = int((~seg_obs).sum())
        if n_missing:
            seg = seg.copy()
            seg[~seg_obs] = np.interp(idx[s0:s1][~seg_obs], idx[s0:s1][seg_obs], seg[seg_obs])
            st.imputed_points += n_missing
        if is_constant(seg):
            st.reasons["constant"] += 1
            continue
        out.append(seg.astype(np.float32, copy=False))
        st.kept_series += 1
        st.kept_obs += len(seg)
    return out
