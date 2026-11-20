"""Corpus build: sources -> caps -> blocklist -> decontamination -> cleaning -> records.

Kept separate from the I/O-heavy LOTSA reader so the whole pipeline is unit-testable with
synthetic sources and runs identically for the debug corpus and the real one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sensorfm.data.dedup import WindowIndex, name_is_blocked
from sensorfm.data.filter import DropStats, clean_series
from sensorfm.data.lotsa import RawSeries
from sensorfm.data.shard import ShardRecord
from sensorfm.data.synthetic_physics import generate


@dataclass
class BuildStats:
    per_dataset: dict[str, DropStats] = field(default_factory=dict)
    raw_series: Counter[str] = field(default_factory=Counter)
    decontaminated: Counter[str] = field(default_factory=Counter)
    blocked_datasets: list[str] = field(default_factory=list)
    capped_datasets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_series": dict(sorted(self.raw_series.items())),
            "decontaminated_series": dict(sorted(self.decontaminated.items())),
            "blocked_datasets": sorted(self.blocked_datasets),
            "capped_datasets": sorted(self.capped_datasets),
            "cleaning": {k: v.to_dict() for k, v in sorted(self.per_dataset.items())},
        }


def build_records(
    sources: Iterable[tuple[str, int | None, Iterable[RawSeries]]],
    min_length: int,
    max_missing_frac: float,
    max_gap: int,
    reference: WindowIndex | None = None,
    stats: BuildStats | None = None,
) -> Iterator[ShardRecord]:
    """Yield clean records. ``sources`` items are (dataset name, max_obs cap, raw series)."""
    st = stats if stats is not None else BuildStats()
    for name, cap, raw_iter in sources:
        if name_is_blocked(name):
            st.blocked_datasets.append(name)
            continue
        ds_stats = st.per_dataset.setdefault(name, DropStats())
        for raw in raw_iter:
            if cap is not None and ds_stats.kept_obs >= cap:
                st.capped_datasets.append(name)
                break
            st.raw_series[name] += 1
            if reference is not None and len(reference) and reference.overlaps(raw.values):
                st.decontaminated[name] += 1
                continue
            for seg in clean_series(raw.values, min_length, max_missing_frac, max_gap, ds_stats):
                yield ShardRecord(
                    key=f"{raw.dataset}/{raw.item_id}",
                    dataset=raw.dataset,
                    domain=raw.domain,
                    freq=raw.freq,
                    values=seg,
                )


def synthetic_source(n: int, length: int, seed: int) -> Iterator[RawSeries]:
    """Synthetic physics series as a RawSeries source (debug corpus and ablation E5)."""
    for i, s in enumerate(generate(n, length, seed)):
        yield RawSeries(
            dataset=f"synthetic_{s.kind}",
            item_id=str(i),
            freq=f"{s.fs:.6g}Hz",
            values=s.values,
            domain="synthetic",
        )


def fake_lotsa_source(name: str, domain: str, n: int, seed: int) -> Iterator[RawSeries]:
    """Messy LOTSA-like series (gaps, constants, short ones) for pipeline tests."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        length = int(rng.integers(64, 3000))
        x = np.cumsum(rng.standard_normal(length)).astype(np.float32)
        if i % 7 == 0:
            x[:] = 3.0
        if i % 5 == 0:
            g = int(rng.integers(0, length // 2))
            x[g : g + int(rng.integers(1, 80))] = np.nan
        yield RawSeries(name, f"item{i}", "H", x, domain)
