"""Distributed correctness check (Task 3.3): DDP / FSDP2 vs single process, same global batch.

    python scripts/dist_parity.py --strategy none --out /tmp/none.json
    torchrun --nproc_per_node=2 scripts/dist_parity.py --strategy ddp --out /tmp/ddp.json
    torchrun --nproc_per_node=2 scripts/dist_parity.py --strategy fsdp --out /tmp/fsdp.json

Every step draws one *global* batch (and its pretraining masks) from a fixed seed; rank r trains
on rows ``[r*B/W, (r+1)*B/W)``. With dropout off, the averaged gradient over the global batch is
the same math in all three setups, so the global-mean loss per step must agree up to float
reduction-order noise. Runs on CPU with the gloo backend, so it needs no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from sensorfm.model.sensorfm import ModelConfig, SensorFM
from sensorfm.pretrain.objectives import ObjectiveConfig, compute_losses, make_hide_masks
from sensorfm.utils.dist import all_reduce_mean, cleanup, init_distributed, wrap_model
from sensorfm.utils.seed import seed_everything


def main(
    strategy: str = "none",
    out: Path = Path("results/scaling/parity.json"),
    steps: int = 10,
    global_batch: int = 16,
    seed: int = 0,
) -> None:
    info = init_distributed(prefer_cuda=False)
    seed_everything(seed)
    cfg = ModelConfig(context_length=128, dropout=0.0)
    core = SensorFM(cfg)
    model = wrap_model(core, strategy, info, "fp32")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    obj = ObjectiveConfig(mask_ratio=0.3, horizon=32, alpha=0.5)
    per_rank = global_batch // info.world_size
    lo, hi = info.rank * per_rank, (info.rank + 1) * per_rank
    losses = []
    for step in range(steps):
        g = torch.Generator().manual_seed(seed * 10_000 + step)
        x = torch.randn(global_batch, 128, generator=g).cumsum(-1)
        obs = torch.ones_like(x, dtype=torch.bool)
        masked, horizon = make_hide_masks(obs, cfg.patch_len, obj, g)
        xs, os_, ms, hs = x[lo:hi], obs[lo:hi], masked[lo:hi], horizon[lo:hi]
        out_ = model(xs, os_, ms | hs)
        loss = compute_losses(core, out_, xs, os_, ms, hs, obj)["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(all_reduce_mean(loss.detach())))
    if info.is_main:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "strategy": strategy,
                    "world_size": info.world_size,
                    "global_batch": global_batch,
                    "losses": losses,
                },
                indent=2,
            )
        )
    cleanup()


if __name__ == "__main__":
    typer.run(main)
