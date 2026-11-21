"""Read selected LOTSA subsets into univariate float32 series.

LOTSA (the Moirai pretraining collection) is ~925 GB on the Hub — far more than free-tier disk or
GPU time can use — so the corpus is a *selection* (``configs/data/lotsa_selection.yaml``) with
per-subset caps on files and observations. Each subset is a folder of Hugging Face ``datasets``
arrow files (IPC *stream* format) with columns ``item_id, start, freq, target``; ``target`` is
``list<float>`` (univariate) or ``list<list<float>>`` (multivariate, variates first). We read the
arrow files directly with ``pyarrow`` instead of depending on ``datasets``: it is lighter and lets
us stop reading as soon as a cap is hit.

Multivariate rows are split into one univariate series per variate, since the model is univariate.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import yaml


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    domain: str
    max_files: int | None = None
    max_obs: int | None = None


@dataclass(frozen=True)
class SelectionConfig:
    repo_id: str
    subsets: list[SubsetSpec]
    default_max_obs: int
    min_length: int
    max_missing_frac: float
    max_gap: int
    val_frac: float
    shard_bytes: int
    seed: int


@dataclass
class RawSeries:
    """One univariate series as read from a source, before cleaning."""

    dataset: str
    item_id: str
    freq: str
    values: npt.NDArray[np.float32]
    domain: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def load_selection(path: str | Path) -> SelectionConfig:
    raw = yaml.safe_load(Path(path).read_text())
    subsets = [SubsetSpec(**s) for s in raw["subsets"]]
    names = [s.name for s in subsets]
    if len(set(names)) != len(names):
        raise ValueError("duplicate subset names in selection")
    return SelectionConfig(
        repo_id=raw["repo_id"],
        subsets=subsets,
        default_max_obs=int(raw["default_max_obs"]),
        min_length=int(raw["min_length"]),
        max_missing_frac=float(raw["max_missing_frac"]),
        max_gap=int(raw["max_gap"]),
        val_frac=float(raw["val_frac"]),
        shard_bytes=int(raw["shard_bytes"]),
        seed=int(raw["seed"]),
    )


def iter_arrow_file(path: str | Path, dataset: str, domain: str = "") -> Iterator[RawSeries]:
    """Yield univariate series from one LOTSA arrow file, splitting multivariate rows."""
    with pa.memory_map(str(path)) as source:
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            item_ids = batch.column("item_id").to_pylist()
            freqs = batch.column("freq").to_pylist()
            targets = batch.column("target")
            for i, (item_id, freq) in enumerate(zip(item_ids, freqs, strict=True)):
                tgt = targets[i].values
                if pa.types.is_list(tgt.type) or pa.types.is_large_list(tgt.type):
                    for v, var in enumerate(targets[i].as_py()):
                        arr = np.asarray(var, dtype=np.float32)  # None -> nan
                        yield RawSeries(dataset, f"{item_id}#v{v}", str(freq), arr, domain)
                else:
                    arr = tgt.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                    yield RawSeries(dataset, str(item_id), str(freq), arr, domain)


@functools.lru_cache(maxsize=4)
def _repo_files(repo_id: str) -> tuple[str, ...]:
    """One listing of the whole repo (~6.7k files), reused for every subset: listing per subset
    would hit the Hub 40+ times and risks rate limits on an unauthenticated Kaggle session."""
    from huggingface_hub import HfApi

    return tuple(HfApi().list_repo_files(repo_id, repo_type="dataset"))


def list_subset_files(repo_id: str, subset: str, max_files: int | None) -> list[str]:
    """Arrow files of one subset on the Hub, sorted, truncated to ``max_files``."""
    files = sorted(
        f for f in _repo_files(repo_id) if f.startswith(f"{subset}/") and f.endswith(".arrow")
    )
    if not files:
        raise FileNotFoundError(f"no arrow files for subset {subset!r} in {repo_id}")
    return files[:max_files] if max_files is not None else files


def iter_subset(
    repo_id: str, spec: SubsetSpec, cache_dir: str | Path | None = None
) -> Iterator[RawSeries]:
    """Download (lazily, file by file) and stream one subset. Caps are applied by the caller."""
    from huggingface_hub import hf_hub_download

    for fname in list_subset_files(repo_id, spec.name, spec.max_files):
        local = hf_hub_download(repo_id, fname, repo_type="dataset", cache_dir=cache_dir)
        yield from iter_arrow_file(local, spec.name, spec.domain)
