"""Forecasting with the quantile head: zero-shot and fine-tuned.

Forecasting is posed exactly as in pretraining: the last ``H / P`` patches of a length-L window
are hidden and the quantile head predicts them. So for a context of ``c`` points and horizon
``H``, the model input is ``c + H`` points (``c + H <= context_length``) with the trailing ``H``
hidden; predictions are de-normalized with RevIN statistics of the visible context only.

Note: with L = 512 and H = 96 the model sees 416 context points, fewer than the 512 given to the
statistical baselines — a disadvantage for SensorFM that the report states.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from sensorfm.model.sensorfm import SensorFM
from sensorfm.pretrain.objectives import ObjectiveConfig, pretrain_step

F32 = npt.NDArray[np.float32]


@torch.no_grad()
def forecast_quantiles(
    model: SensorFM, contexts: F32, horizon: int, device: torch.device, batch: int = 256
) -> F32:
    """``contexts`` (n, c) -> quantiles (n, horizon, Q) on the original scale."""
    p = model.cfg.patch_len
    if horizon % p:
        raise ValueError(f"horizon must be a multiple of {p}")
    n, c = contexts.shape
    total = c + horizon
    if total > model.cfg.context_length or total % p:
        raise ValueError(
            f"context + horizon = {total} must be <= {model.cfg.context_length} and % {p}"
        )
    model.eval()
    n_h = horizon // p
    outs = []
    for i in range(0, n, batch):
        ctx = torch.as_tensor(contexts[i : i + batch], dtype=torch.float32, device=device)
        b = ctx.shape[0]
        x = torch.cat([ctx, torch.zeros(b, horizon, device=device)], dim=1)
        obs = torch.ones_like(x, dtype=torch.bool)
        hide = torch.zeros(b, total // p, dtype=torch.bool, device=device)
        hide[:, -n_h:] = True
        out = model(x, obs, hide)
        q = out.quantiles[:, -n_h:].reshape(b, horizon, -1)  # (b, H, Q) normalized
        outs.append(model.revin.denormalize(q, out.loc, out.scale).cpu())
    return torch.cat(outs).numpy()


def finetune_forecaster(
    model: SensorFM,
    train_series: list[F32],
    horizon: int,
    steps: int,
    seed: int,
    device: torch.device,
    batch: int = 64,
    lr: float = 1e-4,
) -> list[float]:
    """Continue training on in-domain windows with the forecasting objective only (alpha = 0)."""
    rng = np.random.default_rng(seed)
    L = model.cfg.context_length
    usable = [s for s in train_series if len(s) >= L]
    if not usable:
        raise ValueError("no training series long enough")
    obj = ObjectiveConfig(mask_ratio=0.0, horizon=horizon, alpha=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for step in range(steps):
        rows = []
        for _ in range(batch):
            s = usable[int(rng.integers(len(usable)))]
            st = int(rng.integers(0, len(s) - L + 1))
            rows.append(s[st : st + L])
        x = torch.as_tensor(np.stack(rows), dtype=torch.float32, device=device)
        obs = torch.ones_like(x, dtype=torch.bool)
        g = torch.Generator().manual_seed(seed * 1_000_003 + step)
        loss = pretrain_step(model, x, obs, obj, g)["total"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses
