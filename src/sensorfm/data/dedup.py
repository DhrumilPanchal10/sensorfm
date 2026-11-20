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
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

DOWNSTREAM_NAME_PATTERNS: tuple[str, ...] = (
    r"\bett", r"etth", r"ettm", r"cmapss", r"c-mapss", r"turbofan", r"ucr", r"cwru",
    r"paderborn", r"bearing", r"electricity_transformer",
)  # fmt: skip


def name_is_blocked(name: str, patterns: Iterable[str] = DOWNSTREAM_NAME_PATTERNS) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in patterns)


def _window_hash(w: npt.NDArray[np.float64], decimals: int) -> bytes | None:
    sd = w.std()
    if sd < 1e-8:
        return None  # constant windows are uninformative and would collide everywhere
    z = np.round((w - w.mean()) / sd, decimals) + 0.0  # +0.0 folds -0.0 into 0.0
    return hashlib.blake2b(z.astype(np.float32).tobytes(), digest_size=12).digest()


class WindowIndex:
    """Set of hashes of all z-normalized windows of the downstream data."""

    def __init__(self, window: int = 64, decimals: int = 2) -> None:
        if window < 8:
            raise ValueError("window too small to be discriminative")
        self.window = window
        self.decimals = decimals
        self.hashes: set[bytes] = set()

    def add_reference(self, series: npt.ArrayLike) -> None:
        x = np.asarray(series, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < self.window:
            return
        wins = np.lib.stride_tricks.sliding_window_view(x, self.window)
        for w in wins:
            h = _window_hash(w, self.decimals)
            if h is not None:
                self.hashes.add(h)

    def overlaps(self, series: npt.ArrayLike, min_hits: int = 1) -> bool:
        """True if at least ``min_hits`` aligned blocks of ``series`` appear in the reference."""
        x = np.asarray(series, dtype=np.float64)
        hits = 0
        for s in range(0, len(x) - self.window + 1, self.window):
            w = x[s : s + self.window]
            if not np.all(np.isfinite(w)):
                continue
            h = _window_hash(w, self.decimals)
            if h is not None and h in self.hashes:
                hits += 1
                if hits >= min_hits:
                    return True
        return False

    def __len__(self) -> int:
        return len(self.hashes)
