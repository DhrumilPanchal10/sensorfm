"""Latency / throughput benchmark of deployed variants (Task 5.2).

    python bench/latency.py --variant ort_fp32=exports/rul_fp32.onnx --variant ort_int8_dyn=... \
        --task rul --out results/bench/rul_cpu.json

For each variant: p50 / p99 latency at batch 1 and batch 32 (after warm-up, single-threaded and
all-threads), throughput (windows/s at batch 32), model size, and — when ``--eval`` inputs and
targets are given — the task metric and its delta vs the first variant (the FP32 reference).
The full deployment study (fine-tuned model, task-metric deltas, TensorRT) is
``scripts/deploy.py``; this script times arbitrary ONNX files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer

from sensorfm.export.bench import time_session
from sensorfm.export.verify import ort_session
from sensorfm.utils.hardware import hardware_info

app = typer.Typer(add_completion=False)


def rul_inputs(channels: int, window: int):  # type: ignore[no-untyped-def]
    def make(b: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        return {"x": rng.standard_normal((b, channels, window)).astype(np.float32),
                "mask": np.ones((b, window), dtype=bool)}  # fmt: skip

    return make


def forecast_inputs(context: int):  # type: ignore[no-untyped-def]
    def make(b: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        return {"context": rng.standard_normal((b, context)).astype(np.float32)}

    return make


@app.command()
def main(
    variant: list[str] = typer.Option(..., help="name=path.onnx, first is the FP32 reference"),
    task: str = "rul",
    channels: int = 15,
    window: int = 64,
    context: int = 416,
    iters: int = 200,
    threads: str = "1,0",
    out: Path = Path("results/bench/latency.json"),
) -> None:
    make = rul_inputs(channels, window) if task == "rul" else forecast_inputs(context)
    rows = []
    for v in variant:
        name, path = v.split("=", 1)
        p = Path(path)
        row: dict[str, Any] = {"variant": name, "path": path, "size_mb": p.stat().st_size / 1e6}
        for th in (int(t) for t in threads.split(",")):
            sess = ort_session(p, threads=th)
            key = f"threads_{th or 'all'}"
            row[key] = {f"b{b}": time_session(sess, make(b), b, iters) for b in (1, 32)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"task": task, "variants": rows, "hardware": hardware_info()}, indent=2)
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
