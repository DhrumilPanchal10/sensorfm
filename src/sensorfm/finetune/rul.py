"""C-MAPSS RUL with SensorFM (pretrained) or the same architecture from scratch (PatchTST-style).

Windows -> channel-independent encoder features (+ RevIN level/scale per sensor) -> head -> RUL.
Early stopping on held-out *units*; predictions clipped to [0, 125]. Window length must be a
multiple of the patch length (default 64 = 4 patches; shorter histories are left-padded and
masked).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sensorfm.downstream import cmapss
from sensorfm.eval.metrics import nasa_score, rmse
from sensorfm.finetune.common import (
    ChannelIndependent,
    FitCfg,
    Regressor,
    fit,
    load_encoder,
    predict,
    save_regressor,
)
from sensorfm.model.sensorfm import ModelConfig


def run_rul(
    root: Path,
    subset: str,
    regime: str,
    seed: int,
    model_cfg: ModelConfig,
    ckpt: Path | None,
    window: int = 64,
    few_shot: float = 0.0,
    device: torch.device | None = None,
    fit_cfg: FitCfg | None = None,
    save_to: Path | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    data = cmapss.load_subset(root, subset)
    tr_u, va_u = cmapss.split_units(data.train["unit"], 0.2, seed)
    if few_shot:
        rng = np.random.default_rng(seed)
        tr_u = np.sort(rng.choice(tr_u, max(2, round(few_shot * len(tr_u))), replace=False))
    train = cmapss.train_windows(data, tr_u, window)
    val = cmapss.train_windows(data, va_u, window)
    xte, mte, yte = cmapss.test_windows(data, window)

    torch.manual_seed(seed)
    enc = load_encoder(ckpt, model_cfg, device)
    model = Regressor(ChannelIndependent(enc, len(data.sensors)), regime).to(device)
    # Scale the target to ~unit range for optimization; metrics are on the original scale.
    scale = float(cmapss.RUL_CAP)

    class Scaled(nn.Module):
        def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return nn.functional.mse_loss(pred, y / scale)

    def val_rmse(p: Any, y: Any) -> float:
        return rmse(np.clip(p * scale, 0, cmapss.RUL_CAP), y)

    info = fit(model, Scaled(), train, val, val_rmse, seed, device, fit_cfg)
    pred = np.clip(predict(model, xte, mte, device) * scale, 0, cmapss.RUL_CAP)
    if save_to is not None:
        save_regressor(model, save_to, subset=subset, window=window, target_scale=scale,
                       sensors=list(data.sensors))  # fmt: skip
    return {
        "subset": subset, "regime": regime, "seed": seed, "pretrained": ckpt is not None,
        "few_shot_frac": few_shot or None, "n_train_units": len(tr_u), "window": window,
        "rmse": rmse(pred, yte), "nasa": nasa_score(pred, yte),
        "rmse_capped_target": rmse(pred, np.minimum(yte, cmapss.RUL_CAP)), **info,
        "pred": [round(float(v), 3) for v in pred], "true": [float(v) for v in yte],
    }  # fmt: skip
