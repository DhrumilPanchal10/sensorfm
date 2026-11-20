"""SensorFM: RevIN -> patch embedding (+mask token, +learned positions) -> encoder -> heads.

Input contract: ``x`` (B, L) raw values, ``obs`` (B, L) bool (False = padding / missing) and
``hide`` (B, N) bool marking patches whose content the model must not see (masked-patch targets
and the forecast horizon). Hidden patches are replaced by a learned mask token *after* embedding,
and RevIN statistics are computed from visible points only — so nothing about hidden values
reaches the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sensorfm.eval.metrics import QUANTILES
from sensorfm.model.heads import PatchQuantileHead, PatchReconstructionHead, masked_mean_pool
from sensorfm.model.patching import num_patches, patchify
from sensorfm.model.revin import RevIN
from sensorfm.model.transformer import Encoder


@dataclass(frozen=True)
class ModelConfig:
    name: str = "tiny"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    mlp_ratio: int = 4
    patch_len: int = 16
    patch_stride: int = 16
    dropout: float = 0.1
    context_length: int = 512
    quantiles: tuple[float, ...] = QUANTILES
    revin_affine: bool = True
    grad_checkpoint: bool = False

    @property
    def n_patches(self) -> int:
        return num_patches(self.context_length, self.patch_len, self.patch_stride)


@dataclass
class ModelOutput:
    hidden: torch.Tensor  # (B, N, D)
    recon: torch.Tensor  # (B, N, P) normalized scale
    quantiles: torch.Tensor  # (B, N, P, Q) normalized scale
    loc: torch.Tensor  # (B, 1)
    scale: torch.Tensor  # (B, 1)
    x_norm: (
        torch.Tensor
    )  # (B, L) normalized input with hidden points zeroed (targets use x, not this)
    token_valid: torch.Tensor  # (B, N) patch has >= 1 observed point


class SensorFM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.patch_stride != cfg.patch_len:
            raise ValueError(
                "overlapping patches are not supported (hide/unpatchify assume S == P)"
            )
        self.cfg = cfg
        self.revin = RevIN(affine=cfg.revin_affine)
        # Embed values + per-point observation flag so the model can tell padding from zeros.
        self.embed = nn.Linear(2 * cfg.patch_len, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.d_model))
        self.encoder = Encoder(
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, cfg.grad_checkpoint
        )
        self.recon_head = PatchReconstructionHead(cfg.d_model, cfg.patch_len)
        self.quantile_head = PatchQuantileHead(cfg.d_model, cfg.patch_len, cfg.quantiles)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor, hide: torch.Tensor | None = None
    ) -> ModelOutput:
        cfg = self.cfg
        b, length = x.shape
        if length != cfg.context_length:
            raise ValueError(f"expected length {cfg.context_length}, got {length}")
        n = cfg.n_patches
        if hide is None:
            hide = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        hide_pts = hide.repeat_interleave(cfg.patch_len, dim=1)
        visible = obs & ~hide_pts
        xn, loc, scale = self.revin.normalize(x, visible)
        p_vals = patchify(xn, cfg.patch_len, cfg.patch_stride)
        p_obs = patchify(visible.to(xn.dtype), cfg.patch_len, cfg.patch_stride)
        tok = self.embed(torch.cat([p_vals, p_obs], dim=-1))
        tok = torch.where(hide.unsqueeze(-1), self.mask_token.to(tok.dtype), tok)
        tok = tok + self.pos
        token_valid = patchify(obs.to(torch.uint8), cfg.patch_len, cfg.patch_stride).amax(-1) > 0
        # Hidden future patches have no observations but must still be attendable as queries; as
        # keys they carry only the mask token + position, which is harmless.
        key_valid = token_valid | hide
        h = self.encoder(tok, key_valid)
        return ModelOutput(
            hidden=h,
            recon=self.recon_head(h),
            quantiles=self.quantile_head(h),
            loc=loc,
            scale=scale,
            x_norm=xn,
            token_valid=token_valid,
        )

    def embed_series(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Pooled (B, D) representation of fully visible windows, for downstream heads."""
        out = self.forward(x, obs)
        return masked_mean_pool(out.hidden, out.token_valid)

    def normalize_target(
        self, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Targets on the same normalized scale the heads predict in (incl. RevIN affine)."""
        t = (x - loc) / scale
        if self.revin.affine:
            t = t * self.revin.weight + self.revin.bias
        return t


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def non_embedding_params(model: SensorFM) -> int:
    """Params excluding the positional table (the 'N' of the 6N FLOPs rule)."""
    return count_params(model) - model.pos.numel()
