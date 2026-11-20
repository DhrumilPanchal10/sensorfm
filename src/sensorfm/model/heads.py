"""Output heads. All are per-token linear maps, so they work for any number of patches.

- ``PatchReconstructionHead``: token -> P point values (masked patch modeling).
- ``PatchQuantileHead``: token -> P x Q quantiles (probabilistic forecasting). Forecasting is posed
  as predicting *hidden trailing patches*, so the same head serves any horizon that is a multiple
  of P, and pretraining + zero-shot forecasting share one code path. Quantiles are made
  monotone by construction (cumulative softplus increments around the median) so that crossing
  quantiles can't happen at inference.
- ``PooledHead``: masked mean-pool over valid tokens -> MLP, for RUL regression / classification.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import nn


class PatchReconstructionHead(nn.Module):
    def __init__(self, d_model: int, patch_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, patch_len)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # (B, N, P)


class PatchQuantileHead(nn.Module):
    def __init__(self, d_model: int, patch_len: int, quantiles: tuple[float, ...]) -> None:
        super().__init__()
        if any(b <= a for a, b in itertools.pairwise(quantiles)):
            raise ValueError("quantiles must be strictly increasing")
        self.register_buffer("quantiles", torch.tensor(quantiles))
        self.patch_len = patch_len
        self.n_q = len(quantiles)
        self.mid = min(range(self.n_q), key=lambda i: abs(quantiles[i] - 0.5))
        self.proj = nn.Linear(d_model, patch_len * self.n_q)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, P, Q) non-decreasing along Q."""
        raw = self.proj(h).unflatten(-1, (self.patch_len, self.n_q))
        center = raw[..., self.mid : self.mid + 1]
        up = F.softplus(raw[..., self.mid + 1 :]).cumsum(-1)
        down = F.softplus(raw[..., : self.mid]).flip(-1).cumsum(-1).flip(-1)
        return torch.cat([center - down, center, center + up], dim=-1)


def masked_mean_pool(h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    w = valid.to(h.dtype).unsqueeze(-1)
    return (h * w).sum(1) / w.sum(1).clamp_min(1.0)


class PooledHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
