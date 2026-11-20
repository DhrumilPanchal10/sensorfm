"""Optimizer + LR schedule.

AdamW with beta2 = 0.95 (more responsive second-moment estimate; standard for transformer
pretraining, reduces loss spikes) and weight decay 0.1 applied only to matrices: decaying norms,
biases, embeddings, the mask token or positional table pulls them toward zero for no benefit.
Schedule: linear warmup, then cosine decay to ``min_lr_ratio * lr``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.dim() >= 2
        is_embedding_like = any(k in name for k in ("pos", "mask_token", "revin"))
        (decay if is_matrix and not is_embedding_like else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float, betas: tuple[float, float], fused: bool
) -> torch.optim.AdamW:
    kwargs: dict[str, object] = {}
    if fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas, **kwargs)  # type: ignore[arg-type]


def lr_lambda(warmup: int, total: int, min_lr_ratio: float = 0.1):  # type: ignore[no-untyped-def]
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f
