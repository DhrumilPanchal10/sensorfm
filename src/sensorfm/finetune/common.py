"""Shared fine-tuning pieces: encoder loading, channel-independent features, training loop.

The encoder is either loaded from a pretraining checkpoint or **randomly initialized with the same
architecture** — the latter is the from-scratch PatchTST-style baseline, so "pretrained vs scratch"
(ablation E1) differs only in the initial weights.

Multivariate downstream inputs (C-MAPSS has 14-17 sensors) are handled channel-independently as
in PatchTST: each sensor's window is encoded on its own by the univariate encoder, and the head
sees the concatenation of all per-channel features. Regimes:

- ``linear_probe``: encoder frozen (eval mode), a single linear layer on the features;
- ``full``: everything trained, encoder at 0.1x the head learning rate (protects pretrained
  features early in training);
- few-shot is ``full`` (or probe) on a 10% subset of training units, chosen by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from sensorfm.model.sensorfm import ModelConfig, SensorFM

F32 = npt.NDArray[np.float32]


def load_encoder(
    ckpt: Path | None, model_cfg: ModelConfig | None, device: torch.device
) -> SensorFM:
    """Pretrained encoder from a checkpoint directory, or random init when ``ckpt`` is None."""
    if ckpt is None:
        if model_cfg is None:
            raise ValueError("model_cfg is required for a randomly initialized encoder")
        return SensorFM(model_cfg).to(device)
    meta = torch.load(ckpt / "meta.pt", map_location="cpu", weights_only=False)
    cfg_dict = dict(meta["model_config"])
    cfg_dict["quantiles"] = tuple(cfg_dict["quantiles"])
    model = SensorFM(ModelConfig(**cfg_dict)).to(device)
    if (ckpt / "state.pt").exists():  # single process / DDP
        state = torch.load(ckpt / "state.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        return model
    if (ckpt / "dcp").is_dir():  # FSDP2: consolidate the sharded checkpoint once, on CPU
        import tempfile

        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

        with tempfile.TemporaryDirectory() as td:
            full = Path(td) / "full.pt"
            dcp_to_torch_save(str(ckpt / "dcp"), str(full))
            state = torch.load(full, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        return model
    raise FileNotFoundError(f"{ckpt} has neither state.pt nor a dcp/ directory")


class ChannelIndependent(nn.Module):
    """(B, C, W) -> (B, C * (D + 2)) by encoding each channel separately."""

    def __init__(self, encoder: SensorFM, n_channels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_channels = n_channels
        self.out_dim = n_channels * (encoder.cfg.d_model + 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, w = x.shape
        flat = x.reshape(b * c, w)
        m = mask[:, None, :].expand(b, c, w).reshape(b * c, w)
        return self.encoder.features(flat, m).reshape(b, -1)


class Regressor(nn.Module):
    def __init__(self, backbone: ChannelIndependent, regime: str, n_out: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.regime = regime
        self.norm = nn.LayerNorm(backbone.out_dim)
        if regime == "linear_probe":
            self.head: nn.Module = nn.Linear(backbone.out_dim, n_out)
            for p in backbone.parameters():
                p.requires_grad_(False)
        else:
            self.head = nn.Sequential(nn.Linear(backbone.out_dim, 128), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(128, n_out))  # fmt: skip

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.regime == "linear_probe":
            self.backbone.eval()
            with torch.no_grad():
                z = self.backbone(x, mask)
        else:
            z = self.backbone(x, mask)
        out = self.head(self.norm(z))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def make_optimizer(model: Regressor, lr: float) -> torch.optim.Optimizer:
    head = [
        p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad
    ]
    enc = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    groups = [{"params": head, "lr": lr}]
    if enc:
        groups.append({"params": enc, "lr": 0.1 * lr})
    return torch.optim.AdamW(groups, weight_decay=0.01)


@dataclass
class FitCfg:
    epochs: int = 40
    patience: int = 6
    batch: int = 128
    lr: float = 1e-3


def fit(
    model: Regressor,
    loss_fn: nn.Module,
    train: tuple[F32, npt.NDArray[np.bool_], npt.NDArray[np.float32] | npt.NDArray[np.int64]],
    val: tuple[F32, npt.NDArray[np.bool_], npt.NDArray[np.float32] | npt.NDArray[np.int64]],
    val_metric: Callable[[Any, Any], float],  # (pred, target) -> float, lower is better
    seed: int,
    device: torch.device,
    cfg: FitCfg | None = None,
) -> dict[str, float]:
    """Mini-batch training with early stopping on a validation metric (restores best weights)."""
    cfg = cfg or FitCfg()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, cfg.lr)
    x, m, y = (torch.as_tensor(a) for a in train)
    best, best_state, bad, epochs = np.inf, None, 0, 0
    for _ in range(cfg.epochs):
        epochs += 1
        model.train()
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(perm), cfg.batch):
            idx = perm[i : i + cfg.batch]
            pred = model(x[idx].to(device), m[idx].to(device))
            loss = loss_fn(pred, y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = float(val_metric(predict(model, val[0], val[1], device), val[2]))
        if score < best - 1e-6:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return {"val_metric": best, "epochs": float(epochs)}


@torch.no_grad()
def predict(
    model: nn.Module, x: F32, m: npt.NDArray[np.bool_], device: torch.device, batch: int = 512
) -> F32:
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i : i + batch]).to(device)
        mb = torch.as_tensor(m[i : i + batch]).to(device)
        outs.append(model(xb, mb).cpu())
    return torch.cat(outs).numpy()
