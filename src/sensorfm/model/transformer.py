"""Pre-LN transformer encoder on ``scaled_dot_product_attention``.

Why these choices:
- **Pre-LN** (LayerNorm before attention/MLP) trains stably without careful warmup tuning, which
  matters when every run competes for limited free GPU hours.
- **SDPA** dispatches to FlashAttention / memory-efficient kernels when the hardware supports them
  and to the math kernel otherwise, so the same code is fast on GPU and debuggable on CPU.
- **Key-padding mask**: windows shorter than L are left-padded; fully padded patches must not be
  attended to. Each query always keeps at least its own position so no attention row is empty
  (an all-False row makes SDPA return NaN).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, d // self.n_heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, N, Dh)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), attn_mask))
        return x + self.drop(self.mlp(self.ln2(x)))


class Encoder(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, n_heads: int, mlp_ratio: int, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_valid: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` (B, N, D); ``key_valid`` (B, N) bool — False for fully padded tokens."""
        attn_mask = None
        if key_valid is not None and not bool(key_valid.all()):
            n = x.shape[1]
            eye = torch.eye(n, dtype=torch.bool, device=x.device)
            attn_mask = (key_valid[:, None, None, :] | eye[None, None]).expand(-1, 1, n, n)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.ln_f(x)
