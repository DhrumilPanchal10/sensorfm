"""CWRU bearing fault classification with SensorFM (or from-scratch) features.

Windows of ``window`` points (default 512 = ~1.3 shaft revolutions at 12 kHz) from the drive-end
accelerometer -> univariate encoder features -> classifier. Split by **load**: training loads
exclude the test load, and model selection holds out one *training load* (never windows of the
same recording), so no recording contributes to more than one of train / val / test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sensorfm.downstream import bearing
from sensorfm.eval.metrics import macro_f1
from sensorfm.finetune.common import (
    ChannelIndependent,
    FitCfg,
    Regressor,
    fit,
    load_encoder,
    predict,
)
from sensorfm.model.sensorfm import ModelConfig


def run_bearing(
    root: Path,
    test_load: int,
    regime: str,
    seed: int,
    model_cfg: ModelConfig,
    ckpt: Path | None,
    window: int = 512,
    few_shot: float = 0.0,
    device: torch.device | None = None,
    fit_cfg: FitCfg | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    train_loads = tuple(ld for ld in range(4) if ld != test_load)
    val_load = train_loads[seed % len(train_loads)]
    fit_loads = tuple(ld for ld in train_loads if ld != val_load)
    xtr, ytr = bearing.load_windows(root, fit_loads, window)
    xva, yva = bearing.load_windows(root, (val_load,), window)
    xte, yte = bearing.load_windows(root, (test_load,), window)
    if few_shot:
        rng = np.random.default_rng(seed)
        keep = np.concatenate([
            rng.choice(idx, max(1, int(few_shot * len(idx))), replace=False)
            for idx in (np.flatnonzero(ytr == c) for c in np.unique(ytr))
        ])  # fmt: skip
        xtr, ytr = xtr[keep], ytr[keep]

    def as_input(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[:, None, :].astype(np.float32), np.ones((len(x), x.shape[1]), dtype=bool)

    torch.manual_seed(seed)
    enc = load_encoder(ckpt, model_cfg, device)
    n_cls = len(bearing.CLASSES)
    model = Regressor(ChannelIndependent(enc, 1), regime, n_out=n_cls).to(device)

    def neg_f1(p: Any, y: Any) -> float:
        return -macro_f1(np.argmax(p, -1), y, n_cls)

    info = fit(model, nn.CrossEntropyLoss(), (*as_input(xtr), ytr), (*as_input(xva), yva),
               neg_f1, seed, device, fit_cfg)  # fmt: skip
    pred = np.argmax(predict(model, *as_input(xte), device), -1)
    return {
        "test_load": test_load, "val_load": val_load, "regime": regime, "seed": seed,
        "pretrained": ckpt is not None, "few_shot_frac": few_shot or None,
        "n_train_windows": len(xtr), "macro_f1": macro_f1(pred, yte, n_cls),
        "accuracy": float(np.mean(pred == yte)),
        "macro_f1_no_ball": macro_f1_subset(pred, yte, exclude="ball"), **info,
    }  # fmt: skip


def macro_f1_subset(pred: np.ndarray, true: np.ndarray, exclude: str) -> float:
    """Macro-F1 over classes whose name doesn't contain ``exclude`` (7-class secondary metric;
    see the CWRU label check in docs/dataset_card.md)."""
    keep = [i for i, c in enumerate(bearing.CLASSES) if exclude not in c]
    sel = np.isin(true, keep)
    remap = {c: i for i, c in enumerate(keep)}
    t = np.array([remap[v] for v in true[sel]])
    p = np.array([remap.get(int(v), -1) for v in pred[sel]])
    return macro_f1(p, t, len(keep))
