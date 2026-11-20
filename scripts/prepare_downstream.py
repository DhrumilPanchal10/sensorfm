"""Download + verify all downstream datasets and write the decontamination reference.

    python scripts/prepare_downstream.py --root data/downstream          # ~0.6 GB on disk

Layout produced under ``--root``::

    cmapss/  (train|test|RUL)_FD00x.txt
    ucr/     .../UCR_Anomaly_FullData/*.txt
    cwru/    <n>.mat
    ett/     ETTh1.csv ...
    reference/<dataset>/<name>.npy   one file per univariate downstream series (dedup input)

Also writes ``results/eval/cwru_label_check.json`` (physics check of every CWRU label).
Already-present files are not downloaded again.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import numpy as np
import typer

from sensorfm.downstream import bearing, cmapss, forecast_bench, ucr_anomaly
from sensorfm.utils.hardware import hardware_info


def _get(url: str) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=600) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content


def fetch_cmapss(root: Path) -> Path:
    d = root / "cmapss"
    if not (d / "train_FD001.txt").exists() and not list(d.rglob("train_FD001.txt")):
        d.mkdir(parents=True, exist_ok=True)
        outer = zipfile.ZipFile(io.BytesIO(_get(cmapss.URL)))
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
        for n in inner.namelist():
            if n.endswith(".txt") and "/" not in n.strip("/"):
                (d / Path(n).name).write_bytes(inner.read(n))
    return next(d.rglob("train_FD001.txt")).parent


def fetch_ucr(root: Path) -> Path:
    d = root / "ucr"
    if not list(d.rglob("*_UCR_Anomaly_*.txt")):
        d.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(_get(ucr_anomaly.URL)))
        for n in z.namelist():
            if "UCR_Anomaly_FullData/" in n and n.endswith(".txt"):
                (d / Path(n).name).write_bytes(z.read(n))
    return d


def fetch_cwru(root: Path) -> Path:
    d = root / "cwru"
    d.mkdir(parents=True, exist_ok=True)
    for n in sorted(bearing.FILES):
        p = d / f"{n}.mat"
        if not p.exists():
            p.write_bytes(_get(bearing.BASE_URL.format(n=n)))
    return d


def fetch_ett(root: Path) -> Path:
    d = root / "ett"
    d.mkdir(parents=True, exist_ok=True)
    for name in forecast_bench.DATASETS:
        p = d / f"{name}.csv"
        if not p.exists():
            p.write_bytes(_get(forecast_bench.URL.format(name=name)))
    return d


def main(root: Path = Path("data/downstream"), skip_reference: bool = False) -> None:
    cm, ucr, cw, ett = fetch_cmapss(root), fetch_ucr(root), fetch_cwru(root), fetch_ett(root)
    typer.echo(f"downloaded to {root}")

    checks = [
        bearing.verify_file(bearing.load_recording(cw / f"{n}.mat", n))
        for n in sorted(bearing.FILES)
    ]
    out = Path("results/eval/cwru_label_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "rpm_ok": int(sum(bool(c["rpm_ok"]) for c in checks)),
        "signature_ok_by_kind": {
            k: [int(sum(bool(c["signature_ok"]) for c in checks if str(c["label"]).startswith(k))),
                int(sum(1 for c in checks if str(c["label"]).startswith(k)))]
            for k in ("normal", "inner", "outer", "ball")
        },
        "files": checks,
        "hardware": hardware_info(),
    }  # fmt: skip
    out.write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps({k: v for k, v in summary.items() if k != "files" and k != "hardware"}))

    if skip_reference:
        return
    ref = root / "reference"
    n_written = 0

    def put(ds: str, name: str, x: np.ndarray) -> None:
        nonlocal n_written
        (ref / ds).mkdir(parents=True, exist_ok=True)
        np.save(ref / ds / f"{name}.npy", np.asarray(x, np.float32))
        n_written += 1

    for fd in cmapss.SUBSETS:
        for split in ("train", "test"):
            df = cmapss.read_txt(cm / f"{split}_{fd}.txt")
            for unit, g in df.groupby("unit"):
                for s in cmapss.SENSORS:
                    put("cmapss", f"{fd}_{split}_{unit}_{s}", g[s].to_numpy())
    for s in ucr_anomaly.load_archive(ucr):
        put("ucr", f"{s.id:03d}_{s.name}", s.values)
    for n in sorted(bearing.FILES):
        put("cwru", str(n), bearing.load_recording(cw / f"{n}.mat", n).signal)
    for name in forecast_bench.DATASETS:
        for fs in forecast_bench.load_dataset(ett, name):
            put("ett", f"{name}_{fs.column}", fs.values)
    typer.echo(f"reference: {n_written} series under {ref}")


if __name__ == "__main__":
    typer.run(main)
