"""Latency / throughput benchmark of deployed variants (Task 5.2).

    python bench/latency.py --variant ort_fp32=exports/rul_fp32.onnx --variant ort_int8_dyn=... \
        --task rul --out results/bench/rul_cpu.json

For each variant: p50 / p99 latency at batch 1 and batch 32 (after warm-up, single-threaded and
all-threads), throughput (windows/s at batch 32), model size, and — when ``--eval`` inputs and
targets are given — the task metric and its delta vs the first variant (the FP32 reference).
TensorRT engines are benchmarked by ``kaggle/run_deploy.sh`` on the GPU with the same JSON schema.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import typer

from sensorfm.export.verify import ort_session
from sensorfm.utils.hardware import hardware_info

app = typer.Typer(add_completion=False)


def time_session(
    sess: Any, make_inputs: Any, batch: int, iters: int, warmup: int = 20
) -> dict[str, float]:
    inp = make_inputs(batch)
    for _ in range(warmup):
        sess.run(None, inp)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, inp)
        ts.append(time.perf_counter() - t0)
    a = np.array(ts) * 1000
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99)),
            "throughput_per_s": float(batch * 1000 / a.mean())}  # fmt: skip


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
            row[key] = {f"b{b}": time_session(sess, make, b, iters) for b in (1, 32)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"task": task, "variants": rows, "hardware": hardware_info()}, indent=2)
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
