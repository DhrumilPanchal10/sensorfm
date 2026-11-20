"""Pretraining objectives, computed from a single forward pass.

For each window we hide (a) the trailing ``H / P`` patches — the forecast horizon — and (b) a
random fraction ``r`` of the remaining *context* patches. The encoder sees neither. Then:

- ``masked`` = MSE between reconstructed and true normalized values on the hidden context patches
  (observed points only).
- ``forecast`` = pinball (quantile) loss on the horizon patches (observed points only).
- ``total = alpha * masked + (1 - alpha) * forecast``. With ``alpha == 1`` nothing is hidden for
  forecasting and with ``alpha == 0`` no context patches are masked, so neither extreme wastes
  compute on a zero-weighted task (ablation E6).

Targets are normalized with the RevIN statistics of the *visible* points (what the model saw), so
the losses are scale-free across series.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sensorfm.model.patching import patchify
from sensorfm.model.sensorfm import ModelOutput, SensorFM


@dataclass(frozen=True)
class ObjectiveConfig:
    mask_ratio: float = 0.4
    horizon: int = 64
    alpha: float = 0.5


def make_hide_masks(
    obs: torch.Tensor,
    patch_len: int,
    cfg: ObjectiveConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (masked_ctx (B, N), horizon (B, N)) bool masks over patches.

    Only patches with at least one observed point are eligible for context masking; at least one
    observed context patch always stays visible.
    """
    b, length = obs.shape
    n = length // patch_len
    if cfg.horizon % patch_len:
        raise ValueError("horizon must be a multiple of patch_len")
    n_h = cfg.horizon // patch_len if cfg.alpha < 1.0 else 0
    if n_h >= n:
        raise ValueError("horizon leaves no context")
    horizon = torch.zeros(b, n, dtype=torch.bool, device=obs.device)
    if n_h:
        horizon[:, n - n_h :] = True
    p_obs = patchify(obs.to(torch.uint8), patch_len, patch_len).amax(-1) > 0
    eligible = p_obs & ~horizon
    masked = torch.zeros_like(horizon)
    if cfg.alpha > 0.0 and cfg.mask_ratio > 0.0:
        scores = torch.rand(b, n, generator=generator, device="cpu").to(obs.device)
        scores = scores.masked_fill(~eligible, 2.0)  # ineligible sort last
        n_elig = eligible.sum(1)
        n_mask = (
            (n_elig.float() * cfg.mask_ratio).floor().long().clamp(max=(n_elig - 1).clamp_min(0))
        )
        ranks = scores.argsort(1).argsort(1)
        masked = ranks < n_mask[:, None]
    return masked, horizon


def pinball_loss(
    pred_q: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Mean pinball loss. ``pred_q`` (..., Q), ``target``/``weight`` (...)."""
    diff = target.unsqueeze(-1) - pred_q
    loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff).mean(-1)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def compute_losses(
    model: SensorFM,
    out: ModelOutput,
    x: torch.Tensor,
    obs: torch.Tensor,
    masked: torch.Tensor,
    horizon: torch.Tensor,
    cfg: ObjectiveConfig,
) -> dict[str, torch.Tensor]:
    p = model.cfg.patch_len
    target = patchify(model.normalize_target(x, out.loc, out.scale), p, p).float()
    p_obs = patchify(obs.to(target.dtype), p, p)
    zero = target.new_zeros(())
    losses = {"masked": zero, "forecast": zero}
    if masked.any():
        w = p_obs * masked.unsqueeze(-1)
        se = (out.recon.float() - target) ** 2
        losses["masked"] = (se * w).sum() / w.sum().clamp_min(1.0)
    if horizon.any():
        w = p_obs * horizon.unsqueeze(-1)
        qs = torch.as_tensor(model.quantile_head.quantiles).to(target.dtype)
        losses["forecast"] = pinball_loss(out.quantiles.float(), target, qs, w)
    losses["total"] = cfg.alpha * losses["masked"] + (1.0 - cfg.alpha) * losses["forecast"]
    return losses


def pretrain_step(
    model: SensorFM,
    x: torch.Tensor,
    obs: torch.Tensor,
    cfg: ObjectiveConfig,
    generator: torch.Generator | None = None,
    forward: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """One pretraining loss evaluation. ``forward`` is the (DDP/FSDP/compiled) wrapper to call;
    ``model`` is the underlying SensorFM used for config, heads and target normalization."""
    masked, horizon = make_hide_masks(obs, model.cfg.patch_len, cfg, generator)
    out = (forward or model)(x, obs, masked | horizon)
    return compute_losses(model, out, x, obs, masked, horizon, cfg)
