"""Loader throughput: samples/s for several num_workers settings (Task 1.4 / 3.1).

    python scripts/bench_loader.py --root data/shards/debug --tag local_cpu

Writes ``results/bench/loader_<tag>.json``. The training loop must consume slower than this; the
comparison is made in Task 3.1 with the measured model consumption rate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from sensorfm.data.loader import ShardWindowDataset, make_loader
from sensorfm.utils.hardware import hardware_info


def main(
    root: Path = Path("data/shards/debug"),
    tag: str = "local",
    context_length: int = 512,
    batch_size: int = 64,
    batches: int = 200,
    workers: str = "0,1,2,4",
) -> None:
    results = []
    for nw in [int(w) for w in workers.split(",")]:
        ds = ShardWindowDataset(root, context_length=context_length, seed=0)
        dl = make_loader(ds, batch_size=batch_size, num_workers=nw, pin_memory=False)
        it = iter(dl)
        next(it)  # exclude worker start-up from the steady-state rate
        t0 = time.perf_counter()
        for _ in range(batches):
            next(it)
        dt = time.perf_counter() - t0
        sps = batches * batch_size / dt
        results.append({"num_workers": nw, "samples_per_s": round(sps, 1),
                        "per_worker": round(sps / max(nw, 1), 1)})  # fmt: skip
        typer.echo(results[-1])
        del it, dl
    out = Path("results/bench") / f"loader_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "context_length": context_length, "batch_size": batch_size, "batches": batches,
        "root": str(root), "results": results, "hardware": hardware_info(),
    }, indent=2))  # fmt: skip
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
