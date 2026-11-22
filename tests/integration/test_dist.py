"""Task 3.3 correctness on CPU (gloo, 2 processes): parity with single process, and exact resume
under DDP and FSDP2 (sharded DCP checkpoints)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sensorfm.data.build import build_records, synthetic_source
from sensorfm.data.shard import write_corpus
from sensorfm.utils.io import read_metrics

REPO = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
       "OMP_NUM_THREADS": "1"}  # fmt: skip
_port = [29600]


def torchrun(*args: str) -> list[str]:
    _port[0] += 1
    return [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
            "--master_addr=127.0.0.1", f"--master_port={_port[0]}", *args]  # fmt: skip


def _run(cmd: list[str], timeout: int = 300) -> None:
    r = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-4000:])


@pytest.mark.slow
def test_ddp_and_fsdp_match_single_process(tmp_path: Path) -> None:
    out = {}
    _run(
        [
            sys.executable,
            "scripts/dist_parity.py",
            "--strategy",
            "none",
            "--out",
            str(tmp_path / "none.json"),
        ]
    )
    for s in ("ddp", "fsdp"):
        _run(
            torchrun(
                "scripts/dist_parity.py", "--strategy", s, "--out", str(tmp_path / f"{s}.json")
            )
        )
    for s in ("none", "ddp", "fsdp"):
        out[s] = json.loads((tmp_path / f"{s}.json").read_text())["losses"]
    diffs = {
        s: max(abs(a - b) for a, b in zip(out["none"], out[s], strict=True))
        for s in ("ddp", "fsdp")
    }
    assert all(d < 1e-5 for d in diffs.values()), diffs
    rec = REPO / "results" / "scaling" / "parity_cpu_gloo.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps({"max_abs_loss_diff_vs_single": diffs, "steps": len(out["none"]),
                               "losses": out}, indent=2))  # fmt: skip


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shards")
    recs = list(build_records([("synthetic", None, synthetic_source(96, 2048, 0))], 128, 1.0, 32))
    write_corpus(recs, root, shard_bytes=200_000, val_frac=0.1, seed=0)
    return root


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_distributed_resume_is_exact(tmp_path: Path, shards: Path, strategy: str) -> None:
    def cmd(out: Path, *extra: str) -> list[str]:
        return torchrun("-m", "sensorfm.pretrain.train", f"output_dir={out}",
                        f"data.shard_dir={shards}", f"train.strategy={strategy}",
                        "train.device=cpu", "train.max_steps=30", "train.log_every=1",
                        "train.ckpt_every=10", "train.val_every=0", "train.micro_batch=8",
                        *extra)  # fmt: skip

    ref, split = tmp_path / "ref", tmp_path / "split"
    _run(cmd(ref))
    _run(cmd(split, "train.stop_at_step=20"))
    assert json.loads((split / "summary.json").read_text())["final_step"] == 20
    if strategy == "fsdp":
        assert (split / "checkpoints" / "step_00000020" / "dcp").is_dir()
    assert (split / "checkpoints" / "step_00000020" / "rng_rank1.pt").exists()
    _run(cmd(split, "resume=true"))
    # Downstream code loads checkpoints into a plain model: DDP must not leak "module." prefixes,
    # and sharded FSDP2 (DCP) checkpoints must be consolidated on load.
    import torch

    from sensorfm.finetune.common import load_encoder

    enc = load_encoder(split / "checkpoints" / "step_00000020", None, torch.device("cpu"))
    assert enc.cfg.d_model > 0
    a, b = read_metrics(ref / "metrics.jsonl"), read_metrics(split / "metrics.jsonl")
    assert sorted(b) == list(range(1, 31))
    for step in range(1, 31):
        for key in ("loss/total", "lr", "grad_norm"):
            assert a[step][key] == pytest.approx(b[step][key], rel=1e-6, abs=1e-7), (step, key)
