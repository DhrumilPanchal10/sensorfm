"""Masked per-window statistics shared by RevIN, losses and evaluation.

Kept in one place so the model, the metrics and the exported ONNX graph compute the exact same
mean/std (a mismatch here silently breaks de-normalized forecasts).
"""

from __future__ import annotations

import torch


def masked_mean_std(
    x: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std over observed entries along ``dim`` (keepdim). ``mask`` is bool/0-1.

    Fully-masked rows get mean 0, std 1 (identity normalization) instead of NaN.
    """
    m = mask.to(x.dtype)
    n = m.sum(dim=dim, keepdim=True)
    safe_n = n.clamp_min(1.0)
    mean = (x * m).sum(dim=dim, keepdim=True) / safe_n
    var = (((x - mean) * m) ** 2).sum(dim=dim, keepdim=True) / safe_n
    std = torch.sqrt(var + eps)
    empty = n == 0
    mean = torch.where(empty, torch.zeros_like(mean), mean)
    std = torch.where(empty, torch.ones_like(std), std)
    return mean, std
