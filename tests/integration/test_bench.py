"""Ladder harness + report generator run end to end on CPU (tiny model, few steps)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sensorfm.data.build import build_records, synthetic_source
from sensorfm.data.shard import write_corpus

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_ladder_and_report(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    recs = list(build_records([("synthetic", None, synthetic_source(48, 2048, 0))], 128, 1.0, 32))
    write_corpus(recs, shards, shard_bytes=200_000, val_frac=0.1, seed=0)
    out = tmp_path / "ladder.json"
    r = subprocess.run([sys.executable, "-m", "sensorfm.pretrain.bench", "--shards", str(shards),
                        "--rungs", "0,1,6", "--steps", "8", "--warmup", "2", "--micro-batch", "8",
                        "--workers", "0", "--profile", "--out", str(out)],
                       cwd=REPO, capture_output=True, text=True, timeout=600)  # fmt: skip
    assert r.returncode == 0, r.stderr[-3000:]
    lad = json.loads(out.read_text())
    assert [x["rung"]["name"] for x in lad["results"]] == ["0_naive", "1_loader", "6_act_ckpt"]
    assert all(x["tokens_per_s"] > 0 for x in lad["results"])
    assert (tmp_path / "0_naive" / "trace.json").exists()
    md, png = tmp_path / "r.md", tmp_path / "r.png"
    report = "scripts/report_training_efficiency.py"
    r = subprocess.run([sys.executable, report, "--ladder", str(out),
                        "--out-md", str(md), "--out-png", str(png)],
                       cwd=REPO, capture_output=True, text=True, timeout=300)  # fmt: skip
    assert r.returncode == 0, r.stderr[-3000:]
    assert "0_naive" in md.read_text() and png.stat().st_size > 1000
