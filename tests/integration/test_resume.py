"""Task 2.3 acceptance: SIGKILL mid-run, resume, and reproduce the uninterrupted run exactly."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sensorfm.data.build import build_records, synthetic_source
from sensorfm.data.shard import write_corpus
from sensorfm.utils.io import read_metrics

REPO = Path(__file__).resolve().parents[2]
MAX_STEPS = 150


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shards")
    recs = list(build_records([("synthetic", None, synthetic_source(96, 2048, 0))], 128, 1.0, 32))
    write_corpus(recs, root, shard_bytes=200_000, val_frac=0.1, seed=0)
    return root


def _cmd(out: Path, shards: Path, *extra: str) -> list[str]:
    return [
        sys.executable, "-m", "sensorfm.pretrain.train",
        f"output_dir={out}", f"data.shard_dir={shards}",
        f"train.max_steps={MAX_STEPS}", "train.log_every=1", "train.ckpt_every=10",
        "train.val_every=0", "train.micro_batch=8", *extra,
    ]  # fmt: skip


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]


@pytest.mark.slow
def test_sigkill_and_resume_matches_uninterrupted_run(tmp_path: Path, shards: Path) -> None:
    ref, crash = tmp_path / "ref", tmp_path / "crash"
    _run(_cmd(ref, shards))

    proc = subprocess.Popen(_cmd(crash, shards), cwd=REPO, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)  # fmt: skip
    metrics = crash / "metrics.jsonl"
    deadline = time.time() + 120
    killed_after = None
    while time.time() < deadline:
        rows = read_metrics(metrics)
        if rows and max(rows) >= 25:
            os.kill(proc.pid, signal.SIGKILL)
            killed_after = max(rows)
            break
        if proc.poll() is not None:
            break
        time.sleep(0.005)
    proc.wait()
    assert killed_after is not None, "process finished before it could be killed"
    assert killed_after < MAX_STEPS
    assert proc.returncode == -signal.SIGKILL
    assert not (crash / "summary.json").exists()

    t0 = time.time()
    _run(_cmd(crash, shards, "resume=true"))
    recover_wall = time.time() - t0

    summary = json.loads((crash / "summary.json").read_text())
    assert summary["final_step"] == MAX_STEPS
    resumed_step = int(Path(summary["resumed_from"]).name.split("_")[1])
    assert resumed_step % 10 == 0 and resumed_step <= killed_after

    a, b = read_metrics(ref / "metrics.jsonl"), read_metrics(crash / "metrics.jsonl")
    assert sorted(a) == sorted(b) == list(range(1, MAX_STEPS + 1))  # step count consistent
    for step in range(1, MAX_STEPS + 1):
        for key in ("loss/total", "loss/masked", "loss/forecast", "lr", "grad_norm"):
            assert a[step][key] == b[step][key], (step, key, a[step][key], b[step][key])

    record = {"killed_after_step": killed_after, "resumed_from_step": resumed_step,
              "lost_steps": killed_after - resumed_step,
              "seconds_to_first_step_after_resume": summary["seconds_to_first_step"],
              "resume_process_wall_s": recover_wall}  # fmt: skip
    out = REPO / "results" / "resume_test.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2))


@pytest.mark.slow
def test_resume_refuses_changed_batch_size(tmp_path: Path, shards: Path) -> None:
    out = tmp_path / "run"
    _run(_cmd(out, shards, "train.max_steps=10"))
    r = subprocess.run(_cmd(out, shards, "resume=true", "train.micro_batch=4"), cwd=REPO,
                       capture_output=True, text=True, timeout=300)  # fmt: skip
    assert r.returncode != 0
    assert "cannot resume exactly" in r.stderr
