"""Distributed helpers: torchrun environment, DDP / FSDP2 wrapping, gradient-sync control.

Launch: ``torchrun --nproc_per_node=N -m sensorfm.pretrain.train train=ddp``. torchrun sets
RANK / WORLD_SIZE / LOCAL_RANK; without them everything degrades to a single process.

- **DDP** replicates the full model, optimizer state and gradients on every GPU and all-reduces
  gradients during backward (bucketed, overlapped with compute). Simple and fastest when the
  model + optimizer state fit on one GPU (true for all our sizes on 16 GB T4s).
- **FSDP2** (``torch.distributed.fsdp.fully_shard``) shards parameters, gradients and optimizer
  state across GPUs; each block all-gathers its parameters just before use and reduce-scatters
  gradients after. It trades extra communication for ~1/N memory of the model state. Each
  transformer block is its own FSDP unit so only one block's full parameters live at a time.
- Gradient accumulation must skip gradient sync on all but the last micro-step, otherwise every
  micro-step pays a full all-reduce: ``no_sync()`` for DDP, ``set_requires_gradient_sync``
  for FSDP2.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def init_distributed(prefer_cuda: bool = True) -> DistInfo:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = prefer_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return DistInfo(rank, world, local, device)


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


def wrap_model(model: nn.Module, strategy: str, info: DistInfo, precision: str) -> nn.Module:
    if strategy == "none" or not info.distributed:
        return model
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        ids = [info.local_rank] if info.device.type == "cuda" else None
        return DDP(model, device_ids=ids)
    if strategy == "fsdp":
        from torch.distributed.device_mesh import init_device_mesh

        try:  # torch >= 2.6 public API
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        except ImportError:  # torch 2.4/2.5 (e.g. an older Kaggle image)
            from torch.distributed._composable.fsdp import (  # type: ignore[attr-defined]
                MixedPrecisionPolicy,
                fully_shard,
            )

        # Explicit mesh: without it fully_shard auto-detects the accelerator, which on macOS
        # picks MPS and crashes (torch.mps.is_initialized missing) even for CPU/gloo runs.
        mesh = init_device_mesh(info.device.type, (info.world_size,))
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32) if dtype else None
        blocks: Any = getattr(model, "encoder").blocks  # noqa: B009
        for blk in [*blocks, model]:
            if mp is not None:
                fully_shard(blk, mesh=mesh, mp_policy=mp)
            else:
                fully_shard(blk, mesh=mesh)
        return model
    raise ValueError(f"unknown strategy {strategy!r}")


@contextlib.contextmanager
def maybe_no_sync(model: nn.Module, sync: bool) -> Iterator[None]:
    """Skip gradient synchronization for non-final accumulation micro-steps."""
    if sync:
        yield
        return
    m: Any = model
    if hasattr(m, "no_sync"):  # DDP
        with m.no_sync():
            yield
        return
    if hasattr(m, "set_requires_gradient_sync"):  # FSDP2
        m.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            m.set_requires_gradient_sync(True)
        return
    yield


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model  # type: ignore[return-value]
