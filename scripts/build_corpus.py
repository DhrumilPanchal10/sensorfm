"""Build the sharded pretraining corpus.

    # Debug corpus (local/CI, synthetic only, seconds):
    python scripts/build_corpus.py --synthetic-only --out data/shards/debug

    # Real corpus (run on Kaggle — needs ~10-20 GB of temporary download space):
    python scripts/build_corpus.py --selection configs/data/lotsa_selection.yaml \
        --reference data/downstream/reference --out /kaggle/working/shards/main

``--reference`` points at a directory of ``*.npy`` downstream series (written by Task 1.5's
loaders); every pretraining series overlapping them is removed (see ``data/dedup.py``).
``--synthetic-frac`` mixes synthetic physics series into the corpus (ablation E5).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import typer

from sensorfm.data.build import BuildStats, build_records, synthetic_source
from sensorfm.data.dedup import WindowIndex
from sensorfm.data.lotsa import RawSeries, iter_subset, load_selection
from sensorfm.data.shard import write_corpus

app = typer.Typer(add_completion=False)


def _reference_index(ref_dir: Path | None) -> WindowIndex | None:
    if ref_dir is None:
        return None
    idx = WindowIndex()
    files = sorted(ref_dir.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no reference *.npy under {ref_dir}")
    for f in files:
        idx.add_reference(np.load(f))
    typer.echo(f"reference: {len(files)} series, {len(idx)} window hashes")
    return idx


@app.command()
def main(
    out: Path = typer.Option(..., help="output root (gets train/, val/, manifest.json)"),
    selection: Path = Path("configs/data/lotsa_selection.yaml"),
    reference: Path | None = None,
    synthetic_only: bool = False,
    synthetic_series: int = 2000,
    synthetic_length: int = 4096,
    synthetic_frac: float = 0.0,
    cache_dir: Path | None = None,
    shard_bytes: int | None = None,
    val_frac: float | None = None,
) -> None:
    cfg = load_selection(selection)
    t0 = time.time()
    stats = BuildStats()
    sources: list[tuple[str, int | None, Iterable[RawSeries]]] = []
    if not synthetic_only:
        for spec in cfg.subsets:
            cap = spec.max_obs if spec.max_obs is not None else cfg.default_max_obs
            sources.append((spec.name, cap, iter_subset(cfg.repo_id, spec, cache_dir)))
    records = list(
        build_records(sources, cfg.min_length, cfg.max_missing_frac, cfg.max_gap,
                      _reference_index(reference), stats)
    )  # fmt: skip
    real_obs = sum(len(r.values) for r in records)
    if synthetic_only or synthetic_frac > 0:
        n_syn = synthetic_series
        if not synthetic_only:
            # synthetic share of total = frac  ->  syn_obs = frac/(1-frac) * real_obs
            n_syn = int(synthetic_frac / (1 - synthetic_frac) * real_obs / synthetic_length) + 1
        syn = [("synthetic", None, synthetic_source(n_syn, synthetic_length, cfg.seed))]
        records += list(build_records(syn, cfg.min_length, 1.0, cfg.max_gap, None, stats))
    manifest = write_corpus(
        records,
        out,
        shard_bytes or cfg.shard_bytes,
        cfg.val_frac if val_frac is None else val_frac,
        cfg.seed,
        extra_manifest={
            "build": stats.to_dict(),
            "selection_file": str(selection),
            "synthetic_only": synthetic_only,
            "synthetic_frac": synthetic_frac,
            "build_seconds": round(time.time() - t0, 1),
        },
    )
    summary = {
        s: {k: manifest["splits"][s][k] for k in ("n_series", "n_obs", "obs_per_domain")}
        for s in ("train", "val")
    }
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"manifest sha256: {manifest['manifest_sha256']}")


if __name__ == "__main__":
    app()
