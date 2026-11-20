"""Per-asset sliding windows over a stream of (possibly multichannel) sensor readings.

Each asset (an engine, a bearing, a sensor) gets a fixed-size ring buffer of its last ``W``
readings (each reading a vector of ``C`` channels). Once the buffer is full, a window is emitted
every ``hop`` readings. Ring buffers make each push O(C) with no reallocation, so the hot path
stays cheap at high message rates; the emitted window is a copy in chronological order
``(C, W)``, safe to hand to a batching scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


@dataclass
class _Ring:
    buf: F32  # (C, W)
    pos: int = 0
    count: int = 0
    since_emit: int = 0


@dataclass
class Window:
    asset: str
    values: F32  # (C, W), oldest first
    end_index: int  # number of readings seen for this asset when emitted


@dataclass
class SlidingWindows:
    window: int
    hop: int
    channels: int = 1
    _rings: dict[str, _Ring] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window < 1 or self.hop < 1:
            raise ValueError("window and hop must be >= 1")

    def push(self, asset: str, reading: npt.ArrayLike) -> Window | None:
        r = self._rings.get(asset)
        if r is None:
            r = self._rings[asset] = _Ring(np.zeros((self.channels, self.window), np.float32))
        v = np.asarray(reading, dtype=np.float32).reshape(self.channels)
        r.buf[:, r.pos] = v
        r.pos = (r.pos + 1) % self.window
        r.count += 1
        r.since_emit += 1
        if r.count >= self.window and (r.count == self.window or r.since_emit >= self.hop):
            r.since_emit = 0
            ordered = np.concatenate([r.buf[:, r.pos :], r.buf[:, : r.pos]], axis=1)
            return Window(asset, ordered, r.count)
        return None

    def assets(self) -> list[str]:
        return sorted(self._rings)
