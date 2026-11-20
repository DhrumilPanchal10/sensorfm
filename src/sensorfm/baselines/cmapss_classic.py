"""Classic C-MAPSS RUL baselines: gradient boosting on window features, LSTM, 1D-CNN.

These are the reference points every RUL paper reports; SensorFM must beat them to claim anything.
All use the same windows (W = 30, the common choice), the same per-condition normalization and
the same unit-level validation split as the foundation-model fine-tuning, so differences come from
the model, not the protocol. Predictions are clipped to [0, 125] (the label cap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from sensorfm.downstream.cmapss import RUL_CAP

F32 = npt.NDArray[np.float32]


def window_features(x: F32, mask: npt.NDArray[np.bool_]) -> F32:
    """Per-channel last, mean, std, min, max, least-squares slope over observed cycles."""
    w = x.shape[2]
    m = mask[:, None, :].astype(np.float32)
    cnt = m.sum(-1).clip(min=1)
    mean = (x * m).sum(-1) / cnt
    var = (((x - mean[..., None]) * m) ** 2).sum(-1) / cnt
    big = np.where(m > 0, x, np.inf)
    small = np.where(m > 0, x, -np.inf)
    t = np.arange(w, dtype=np.float32)[None, None, :]
    tm = (t * m).sum(-1) / cnt
    cov = (((t - tm[..., None]) * (x - mean[..., None])) * m).sum(-1)
    tvar = (((t - tm[..., None]) ** 2) * m).sum(-1).clip(min=1e-6)
    feats = [x[:, :, -1], mean, np.sqrt(var), big.min(-1), small.max(-1), cov / tvar]
    return np.concatenate(feats, axis=1).astype(np.float32)


def fit_predict_gbm(
    xtr: F32, mtr: npt.NDArray[np.bool_], ytr: F32,
    xva: F32, mva: npt.NDArray[np.bool_], yva: F32,
    xte: F32, mte: npt.NDArray[np.bool_], seed: int,
) -> tuple[F32, dict[str, float]]:  # fmt: skip
    from sklearn.ensemble import HistGradientBoostingRegressor

    ftr, fva, fte = window_features(xtr, mtr), window_features(xva, mva), window_features(xte, mte)
    best: tuple[float, int] = (np.inf, 0)
    for n_iter in (100, 200, 400):
        m = HistGradientBoostingRegressor(max_iter=n_iter, learning_rate=0.05, max_leaf_nodes=31,
                                          random_state=seed, early_stopping=False)  # fmt: skip
        m.fit(ftr, ytr)
        rmse = float(np.sqrt(np.mean((np.clip(m.predict(fva), 0, RUL_CAP) - yva) ** 2)))
        if rmse < best[0]:
            best = (rmse, n_iter)
    final = HistGradientBoostingRegressor(max_iter=best[1], learning_rate=0.05, max_leaf_nodes=31,
                                          random_state=seed, early_stopping=False)  # fmt: skip
    final.fit(np.concatenate([ftr, fva]), np.concatenate([ytr, yva]))
    pred = np.clip(final.predict(fte), 0, RUL_CAP).astype(np.float32)
    return pred, {"val_rmse": best[0], "max_iter": best[1]}


class LSTMRegressor(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm((x * mask[:, None, :]).transpose(1, 2))  # (B, W, H)
        return self.head(h[:, -1]).squeeze(-1)


class CNNRegressor(nn.Module):
    """1D CNN in the style of Li et al. (2018): stacked temporal convolutions over sensors."""

    def __init__(self, n_in: int, width: int = 10, k: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, width, k, padding="same"), nn.Tanh(),
            nn.Conv1d(width, width, k, padding="same"), nn.Tanh(),
            nn.Conv1d(width, width, k, padding="same"), nn.Tanh(),
            nn.Conv1d(width, 1, 3, padding="same"), nn.Tanh(),
        )  # fmt: skip
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.LazyLinear(100), nn.Tanh(),
                                  nn.Linear(100, 1))  # fmt: skip

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x * mask[:, None, :])).squeeze(-1)


@dataclass
class TrainCfg:
    epochs: int = 60
    patience: int = 8
    batch: int = 256
    lr: float = 1e-3


def fit_predict_torch(
    model: nn.Module,
    xtr: F32, mtr: npt.NDArray[np.bool_], ytr: F32,
    xva: F32, mva: npt.NDArray[np.bool_], yva: F32,
    xte: F32, mte: npt.NDArray[np.bool_],
    seed: int, cfg: TrainCfg | None = None,
) -> tuple[F32, dict[str, float]]:  # fmt: skip
    cfg = cfg or TrainCfg()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    T = torch.from_numpy
    xtr_t, mtr_t, ytr_t = T(xtr), T(mtr.astype(np.float32)), T(ytr)
    model(xtr_t[:2], mtr_t[:2])  # materialize lazy layers before building the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def predict(x: F32, m: npt.NDArray[np.bool_]) -> F32:
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(T(x[i : i + 2048]), T(m[i : i + 2048].astype(np.float32)))
                             for i in range(0, len(x), 2048)])  # fmt: skip
        model.train()
        return np.clip(out.numpy() * RUL_CAP, 0, RUL_CAP).astype(np.float32)

    best_rmse, best_state, bad, epochs_run = np.inf, None, 0, 0
    for _epoch in range(cfg.epochs):
        epochs_run += 1
        perm = torch.randperm(len(xtr_t), generator=g)
        for i in range(0, len(perm), cfg.batch):
            idx = perm[i : i + cfg.batch]
            # Targets scaled to ~[0, 1]: raw 0-125 targets saturate the tanh CNN (a seed collapsed
            # to a constant) and slow the LSTM: an unfairly weak baseline. Same scaling as SensorFM.
            loss = nn.functional.mse_loss(model(xtr_t[idx], mtr_t[idx]), ytr_t[idx] / RUL_CAP)
            opt.zero_grad()
            loss.backward()
            opt.step()
        rmse = float(np.sqrt(np.mean((predict(xva, mva) - yva) ** 2)))
        if rmse < best_rmse - 1e-3:
            best_rmse, bad = rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return predict(xte, mte), {"val_rmse": best_rmse, "epochs": float(epochs_run)}
