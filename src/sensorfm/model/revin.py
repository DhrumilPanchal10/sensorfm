"""Reversible instance normalization (RevIN, Kim et al. 2022).

Sensor series differ in scale by many orders of magnitude (accelerations of 1e4 next to
temperatures of 20), so each window is z-normalized by its own mean/std before the encoder and
predictions are mapped back afterwards. Two details matter:

- Statistics use only **visible** points (observed and not hidden by the pretraining mask).
  Computing them over the whole window would leak the mean/std of the very values the model is
  asked to predict (forecast horizon or masked patches).
- An optional learnable per-channel affine (we are univariate, so one scale/shift) lets the model
  undo the normalization where absolute level matters.
"""

from __future__ import annotations

import torch
from torch import nn

from sensorfm.data.normalize import masked_mean_std


class RevIN(nn.Module):
    def __init__(self, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def normalize(
        self, x: torch.Tensor, visible: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x``, ``visible`` (B, L) -> (normalized x with hidden points zeroed, loc, scale)."""
        loc, scale = masked_mean_std(x.float(), visible, dim=-1, eps=self.eps)
        xn = (x - loc) / scale
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn * visible.to(xn.dtype), loc, scale

    def denormalize(self, y: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Map normalized predictions back. ``loc``/``scale`` are (B, 1); ``y`` is (B, ...)."""
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps)
        shape = (loc.shape[0],) + (1,) * (y.dim() - 1)
        return y * scale.reshape(shape) + loc.reshape(shape)
