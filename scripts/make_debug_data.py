"""Generate the synthetic debug corpus (CPU, seconds) and a preview plot.

Writes ``data/raw/synthetic_debug.npz`` (values, fs, kind) and a JSON manifest. Task 1.3's shard
writer will consume the same in-memory series for debug shards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

from sensorfm.data.synthetic_physics import generate


def main(
    n: int = 256,
    length: int = 2048,
    seed: int = 0,
    out: Path = Path("data/raw/synthetic_debug.npz"),
    plot: Path = Path("data/raw/synthetic_debug_preview.png"),
) -> None:
    series = list(generate(n, length, seed))
    values = np.stack([s.values for s in series])
    fs = np.array([s.fs for s in series])
    kinds = np.array([s.kind for s in series])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, values=values, fs=fs, kind=kinds)
    manifest = {
        "n_series": n,
        "length": length,
        "seed": seed,
        "n_observations": int(values.size),
        "per_kind": {k: int((kinds == k).sum()) for k in sorted(set(kinds.tolist()))},
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    for ax, s in zip(axes, series[:4], strict=True):
        ax.plot(np.arange(length) / s.fs, s.values, lw=0.6)
        ax.set_title(f"{s.kind}  fs={s.fs:.1f} Hz  snr={s.params['snr_db']:.0f} dB", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot, dpi=110)
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    typer.run(main)
