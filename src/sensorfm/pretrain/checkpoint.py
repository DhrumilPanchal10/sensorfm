"""Crash-safe checkpoints that allow exact resumption.

A checkpoint holds everything that determines the rest of the run: model, optimizer, LR scheduler,
GradScaler, step, all RNG states (python / numpy / torch CPU / every CUDA device) and the data
position (global batches consumed). A fingerprint (world size, micro-batch, grad accumulation,
num_workers, seed, context length) is stored and checked on resume, because the loader's resume
arithmetic is only exact if those are unchanged.

Writes are **atomic**: save to ``<name>.tmp`` then ``os.replace`` — a kill mid-write leaves the
previous checkpoint intact, never a truncated file named like a valid one. Only the last ``K`` are
kept. Sharded (FSDP) checkpoints via ``torch.distributed.checkpoint`` are added in Task 3.3.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PAT = re.compile(r"^step_(\d{8})\.pt$")


def rng_state() -> dict[str, Any]:
    st: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def list_checkpoints(ckpt_dir: Path) -> list[Path]:
    if not ckpt_dir.exists():
        return []
    return sorted(p for p in ckpt_dir.iterdir() if _PAT.match(p.name))


def latest(ckpt_dir: Path) -> Path | None:
    cks = list_checkpoints(ckpt_dir)
    return cks[-1] if cks else None


def save(ckpt_dir: Path, step: int, state: dict[str, Any], keep_last: int) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({**state, "step": step, "rng": rng_state()}, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    for old in list_checkpoints(ckpt_dir)[:-keep_last]:
        old.unlink()
    return path


def load(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    return state


def check_fingerprint(saved: dict[str, Any], current: dict[str, Any]) -> None:
    diffs = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    if diffs:
        raise RuntimeError(f"cannot resume exactly; run settings changed: {diffs}")
