"""Crash-safe checkpoints that allow exact resumption — single process, DDP and FSDP2.

A checkpoint is a directory ``step_XXXXXXXX/`` holding everything that determines the rest of the
run:

- model + optimizer: ``state.pt`` written by rank 0 (none / DDP: every rank holds the full,
  identical state), or a **sharded** ``torch.distributed.checkpoint`` (DCP) save for FSDP2, where
  each rank writes only its own parameter/optimizer shards — a plain ``torch.save`` of an FSDP2
  state dict would silently keep rank 0's shard only;
- ``meta.pt`` (rank 0): step, LR scheduler, GradScaler, run fingerprint, model config;
- ``rng_rank{r}.pt`` (every rank): python / numpy / torch / CUDA RNG states. Ranks have different
  dropout streams, so restoring rank 0's RNG everywhere would not resume exactly.

The data position is implied by the step (the loader resumes by batch count). The fingerprint
(world size, micro-batch, grad accumulation, num_workers, seed, context length) is checked on
resume, because that arithmetic is only exact if those are unchanged.

**Atomicity:** everything is written into ``step_XXXXXXXX.tmp/``; after a barrier rank 0 renames
it. A kill at any point leaves either the previous complete checkpoint or a ``.tmp`` directory
that is never loaded. Only the last ``K`` checkpoints are kept.
"""

from __future__ import annotations

import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sensorfm.utils.dist import DistInfo, barrier, unwrap

_PAT = re.compile(r"^step_(\d{8})$")


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
    return sorted(p for p in ckpt_dir.iterdir() if p.is_dir() and _PAT.match(p.name))


def latest(ckpt_dir: Path) -> Path | None:
    cks = list_checkpoints(ckpt_dir)
    return cks[-1] if cks else None


def step_of(path: Path) -> int:
    m = _PAT.match(path.name)
    if not m:
        raise ValueError(f"not a checkpoint directory: {path}")
    return int(m.group(1))


def _fsync_dir_files(d: Path) -> None:
    for f in d.rglob("*"):
        if f.is_file():
            with open(f, "rb") as fh:
                os.fsync(fh.fileno())


def save(
    ckpt_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    meta: dict[str, Any],
    info: DistInfo,
    sharded: bool,
    keep_last: int,
) -> Path:
    """Collective when distributed: every rank must call it."""
    final = ckpt_dir / f"step_{step:08d}"
    tmp = ckpt_dir / f"step_{step:08d}.tmp"
    if info.is_main:
        if tmp.exists():
            shutil.rmtree(tmp)  # leftover from a crash during a previous save of this step
        tmp.mkdir(parents=True)
    barrier()
    if sharded:
        from torch.distributed.checkpoint.state_dict import get_state_dict
        from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

        msd, osd = get_state_dict(model, optimizer)
        dcp_save({"model": msd, "optimizer": osd}, checkpoint_id=str(tmp / "dcp"))
    elif info.is_main:
        torch.save(
            # unwrap: a DDP state_dict prefixes every key with "module.", which would make the
            # checkpoint unloadable by anything but another DDP wrapper (e.g. downstream code).
            {"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict()},
            tmp / "state.pt",
        )
    if info.is_main:
        torch.save({**meta, "step": step, "world_size": info.world_size}, tmp / "meta.pt")
    torch.save(rng_state(), tmp / f"rng_rank{info.rank}.pt")
    barrier()
    if info.is_main:
        _fsync_dir_files(tmp)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        for old in list_checkpoints(ckpt_dir)[:-keep_last]:
            shutil.rmtree(old)
    barrier()
    return final


def load(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    info: DistInfo,
    sharded: bool,
) -> dict[str, Any]:
    """Restore model/optimizer in place and this rank's RNG; return the metadata dict."""
    meta: dict[str, Any] = torch.load(path / "meta.pt", map_location="cpu", weights_only=False)
    if sharded:
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

        msd, osd = get_state_dict(model, optimizer)
        state: dict[str, Any] = {"model": msd, "optimizer": osd}
        dcp_load(state, checkpoint_id=str(path / "dcp"))
        set_state_dict(
            model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"]
        )
    else:
        st = torch.load(path / "state.pt", map_location=info.device, weights_only=False)
        unwrap(model).load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
    rng_file = path / f"rng_rank{info.rank}.pt"
    set_rng_state(torch.load(rng_file, map_location="cpu", weights_only=False))
    return meta


def check_fingerprint(saved: dict[str, Any], current: dict[str, Any]) -> None:
    diffs = {k: (saved.get(k), v) for k, v in current.items() if saved.get(k) != v}
    if diffs:
        raise RuntimeError(f"cannot resume exactly; run settings changed: {diffs}")
