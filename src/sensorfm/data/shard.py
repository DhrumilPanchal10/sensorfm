"""Fixed-size streaming shards of univariate series.

Format (per split directory, e.g. ``shards/train/``):

- ``shard_00000.bin`` — all series of the shard concatenated as raw little-endian float32. Read via
  ``np.memmap`` so a worker touches only the pages of the windows it samples.
- ``shard_00000.idx.npz`` — ``offsets`` (int64), ``lengths`` (int64), ``dataset`` (int32 index into
  the manifest's dataset list), ``freq`` (int32 index into the manifest's freq list).
- ``manifest.json`` (one level up) — datasets, domains, freqs, per-split/per-domain/per-dataset
  counts, cleaning drop stats, sha256 of every file, and the config that produced it. Its hash is
  logged with every training run as the "data manifest hash".

Why raw memmap instead of Arrow/Parquet: the loader's only access pattern is "random window of a
random series", which a memmap serves with zero decoding and zero copies, and the format needs no
library to read. Series are globally shuffled before writing so every shard mixes domains; the
loader additionally shuffles at shard and buffer level.

Train/val split is **by series key** (``dataset/item_id``, so all segments of one source series
land on the same side) via a stable hash — never by window, which would leak.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


@dataclass
class ShardRecord:
    key: str  # "dataset/item_id" — split assignment uses this
    dataset: str
    domain: str
    freq: str
    values: F32


def split_of(key: str, val_frac: float) -> str:
    """Stable train/val assignment from the series key."""
    h = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "big")
    return "val" if (h % 1_000_000) < val_frac * 1_000_000 else "train"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(
    records: Sequence[ShardRecord],
    out_dir: Path,
    shard_bytes: int,
    datasets: list[str],
    freqs: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Shuffle ``records`` and write them into ~``shard_bytes`` shards. Returns shard metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_idx = {d: i for i, d in enumerate(datasets)}
    fq_idx = {f: i for i, f in enumerate(freqs)}
    order = np.random.default_rng(seed).permutation(len(records))
    shards: list[dict[str, Any]] = []
    buf: list[ShardRecord] = []
    buf_bytes = 0

    def flush() -> None:
        nonlocal buf, buf_bytes
        if not buf:
            return
        sid = len(shards)
        bin_path = out_dir / f"shard_{sid:05d}.bin"
        idx_path = out_dir / f"shard_{sid:05d}.idx.npz"
        lengths = np.array([len(r.values) for r in buf], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        tmp = bin_path.with_suffix(".bin.tmp")
        with tmp.open("wb") as f:
            for r in buf:
                f.write(np.ascontiguousarray(r.values, dtype="<f4").tobytes())
        tmp.replace(bin_path)
        np.savez(
            idx_path,
            offsets=offsets,
            lengths=lengths,
            dataset=np.array([ds_idx[r.dataset] for r in buf], dtype=np.int32),
            freq=np.array([fq_idx[r.freq] for r in buf], dtype=np.int32),
        )
        shards.append({
            "bin": bin_path.name,
            "idx": idx_path.name,
            "n_series": len(buf),
            "n_obs": int(lengths.sum()),
            "sha256_bin": sha256_file(bin_path),
            "sha256_idx": sha256_file(idx_path),
        })  # fmt: skip
        buf, buf_bytes = [], 0

    for i in order:
        r = records[int(i)]
        buf.append(r)
        buf_bytes += 4 * len(r.values)
        if buf_bytes >= shard_bytes:
            flush()
    flush()
    return shards


def write_corpus(
    records: Iterable[ShardRecord],
    root: Path,
    shard_bytes: int,
    val_frac: float,
    seed: int,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Split, shard and write a manifest. Returns the manifest dict.

    Refuses to write into a split directory that already holds shards, so a rebuild can never
    silently mix stale shards from an earlier build. Use a fresh output directory instead.
    """
    for split in ("train", "val"):
        if (root / split).exists() and any((root / split).glob("shard_*")):
            raise FileExistsError(f"{root / split} already has shards; use a fresh --out")
    recs = list(records)
    datasets = sorted({r.dataset for r in recs})
    freqs = sorted({r.freq for r in recs})
    domain_of = {r.dataset: r.domain for r in recs}
    manifest: dict[str, Any] = {
        "format": "sensorfm-shards-v1",
        "datasets": datasets,
        "domains": domain_of,
        "freqs": freqs,
        "val_frac": val_frac,
        "seed": seed,
        "splits": {},
    }
    for split in ("train", "val"):
        part = [r for r in recs if split_of(r.key, val_frac) == split]
        shards = write_split(part, root / split, shard_bytes, datasets, freqs, seed)
        per_domain: Counter[str] = Counter()
        per_dataset: Counter[str] = Counter()
        for r in part:
            per_domain[r.domain] += len(r.values)
            per_dataset[r.dataset] += len(r.values)
        manifest["splits"][split] = {
            "n_series": len(part),
            "n_obs": int(sum(per_dataset.values())),
            "obs_per_domain": dict(sorted(per_domain.items())),
            "obs_per_dataset": dict(sorted(per_dataset.items())),
            "shards": shards,
        }
    if extra_manifest:
        manifest.update(extra_manifest)
    body = json.dumps(manifest, indent=2, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


class ShardReader:
    """Zero-copy random access to the series of one shard."""

    def __init__(self, split_dir: Path, meta: dict[str, Any]) -> None:
        idx = np.load(split_dir / meta["idx"])
        self.offsets: npt.NDArray[np.int64] = idx["offsets"]
        self.lengths: npt.NDArray[np.int64] = idx["lengths"]
        self.dataset: npt.NDArray[np.int32] = idx["dataset"]
        self.freq: npt.NDArray[np.int32] = idx["freq"]
        self.values: np.memmap[Any, np.dtype[np.float32]] = np.memmap(
            split_dir / meta["bin"], dtype="<f4", mode="r"
        )

    def __len__(self) -> int:
        return len(self.lengths)

    def series(self, i: int) -> F32:
        o, n = int(self.offsets[i]), int(self.lengths[i])
        return np.asarray(self.values[o : o + n])

    def window(self, i: int, start: int, length: int) -> F32:
        o = int(self.offsets[i]) + start
        return np.asarray(self.values[o : o + length])


def load_manifest(root: Path) -> dict[str, Any]:
    m: dict[str, Any] = json.loads((root / "manifest.json").read_text())
    if m.get("format") != "sensorfm-shards-v1":
        raise ValueError(f"unknown shard format in {root}")
    return m
