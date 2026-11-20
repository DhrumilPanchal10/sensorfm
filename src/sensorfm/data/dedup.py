"""Decontamination: keep downstream evaluation data out of the pretraining corpus.

Two independent filters:

1. **Source-name blocklist.** Any pretraining subset whose name matches a downstream dataset
   pattern (ETT, C-MAPSS, UCR, CWRU, ...) is rejected outright.
2. **Window hashing.** Copies of downstream data can hide under other names (re-uploads,
   aggregations). We z-normalize short windows, quantize them, and hash. Downstream series are
   hashed at **every offset** (stride 1); pretraining series only at **aligned** blocks (stride W).
   Any shared stretch of length >= 2W must fully contain one aligned pretraining block, whose hash
   is then guaranteed to be in the downstream set — so overlap is detected regardless of shift,
   while pretraining costs only one hash per W points. Scale/offset changes are absorbed by the
   z-normalization. Quantization can in rare cases split near-identical values across a bucket
   boundary; that only causes misses on noisy copies, never false positives on unrelated data.

Implementation: the reference side is ~20M windows (UCR alone), so hashing is fully vectorized. A
quantized window ``q`` (int64 vector) is hashed as ``sum(q * c) mod 2^64`` with fixed random odd
64-bit coefficients ``c`` — a random linear hash, whose collision probability for two distinct
windows is ~2^-64, so false positives across ~1e7 x 1e8 comparisons stay negligible. Hashes live
in a sorted ``uint64`` array; lookups are ``np.searchsorted``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

DOWNSTREAM_NAME_PATTERNS: tuple[str, ...] = (
    r"\bett", r"etth", r"ettm", r"cmapss", r"c-mapss", r"turbofan", r"ucr", r"cwru",
    r"paderborn", r"bearing", r"electricity_transformer",
)  # fmt: skip

U64 = npt.NDArray[np.uint64]


def name_is_blocked(name: str, patterns: Iterable[str] = DOWNSTREAM_NAME_PATTERNS) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in patterns)


class WindowIndex:
    """Sorted set of hashes of all z-normalized windows of the downstream data."""

    def __init__(self, window: int = 64, decimals: int = 2, chunk: int = 200_000) -> None:
        if window < 8:
            raise ValueError("window too small to be discriminative")
        self.window = window
        self.min_distinct = max(4, window // 4)
        self.scale = 10.0**decimals
        self.chunk = chunk
        rng = np.random.default_rng(0x5EED)
        self._coef = rng.integers(1, 2**63, size=window, dtype=np.uint64) | np.uint64(1)
        self._parts: list[U64] = []
        self._sorted: U64 | None = None

    def _hash_rows(self, wins: npt.NDArray[np.float64]) -> U64:
        """Hashes of the non-constant rows of ``wins`` (n, W); constant rows are dropped."""
        mean = wins.mean(axis=1, keepdims=True)
        sd = wins.std(axis=1, keepdims=True)
        keep = (sd[:, 0] >= 1e-8) & np.isfinite(wins).all(axis=1)
        if not keep.any():
            return np.empty(0, np.uint64)
        z = (wins[keep] - mean[keep]) / sd[keep]
        qi = np.rint(z * self.scale).astype(np.int64)
        # Low-information windows (e.g. a flat run with one spike) z-normalize to the same pattern
        # in unrelated datasets and caused false "leaks" (found on real PEMS04 / BDG-2 data), so
        # windows with fewer than ``min_distinct`` distinct quantized values are not hashed.
        srt = np.sort(qi, axis=1)
        distinct = 1 + (np.diff(srt, axis=1) != 0).sum(axis=1)
        qi = qi[distinct >= self.min_distinct]
        if len(qi) == 0:
            return np.empty(0, np.uint64)
        q = qi.view(np.uint64)
        with np.errstate(over="ignore"):
            h: U64 = (q * self._coef).sum(axis=1, dtype=np.uint64)
        return h

    def add_reference(self, series: npt.ArrayLike) -> None:
        x = np.asarray(series, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < self.window:
            return
        wins = np.lib.stride_tricks.sliding_window_view(x, self.window)
        for s in range(0, len(wins), self.chunk):
            self._parts.append(self._hash_rows(np.asarray(wins[s : s + self.chunk])))
        self._sorted = None

    def _table(self) -> U64:
        if self._sorted is None:
            allh = np.concatenate(self._parts) if self._parts else np.empty(0, np.uint64)
            self._sorted = np.unique(allh)
            self._parts = [self._sorted]
        return self._sorted

    def overlaps(self, series: npt.ArrayLike, min_hits: int = 1) -> bool:
        """True if at least ``min_hits`` aligned blocks of ``series`` appear in the reference."""
        table = self._table()
        x = np.asarray(series, dtype=np.float64)
        n_blocks = len(x) // self.window
        if n_blocks == 0 or len(table) == 0:
            return False
        h = self._hash_rows(x[: n_blocks * self.window].reshape(n_blocks, self.window))
        if len(h) == 0:
            return False
        pos = np.clip(np.searchsorted(table, h), 0, len(table) - 1)
        return int((table[pos] == h).sum()) >= min_hits

    def __len__(self) -> int:
        return len(self._table())
