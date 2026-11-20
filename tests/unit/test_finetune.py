"""Task 4.2 fine-tuning code paths on tiny models (correctness, not quality)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sensorfm.finetune.anomaly import anomaly_scores, rolling_robust_z
from sensorfm.finetune.common import ChannelIndependent, FitCfg, Regressor, load_encoder
from sensorfm.finetune.forecast import finetune_forecaster, forecast_quantiles
from sensorfm.model.sensorfm import ModelConfig, SensorFM

CFG = ModelConfig(context_length=128, dropout=0.0)
DEV = torch.device("cpu")


def test_channel_independent_shapes_and_probe_freezes_encoder() -> None:
    enc = SensorFM(CFG)
    ci = ChannelIndependent(enc, n_channels=3)
    x = torch.randn(4, 3, 64)
    m = torch.ones(4, 64, dtype=torch.bool)
    assert ci(x, m).shape == (4, ci.out_dim)
    probe = Regressor(ci, "linear_probe")
    assert not any(p.requires_grad for p in probe.backbone.parameters())
    assert probe(x, m).shape == (4,)
    full = Regressor(ChannelIndependent(SensorFM(CFG), 3), "full")
    assert all(p.requires_grad for p in full.backbone.parameters())


def test_forecast_quantiles_shapes_scale_and_monotone() -> None:
    torch.manual_seed(0)
    m = SensorFM(CFG)
    ctx = (np.random.default_rng(0).standard_normal((5, 96)) * 50 + 1000).astype(np.float32)
    q = forecast_quantiles(m, ctx, 32, DEV)
    assert q.shape == (5, 32, len(CFG.quantiles))
    assert np.all(np.diff(q, axis=-1) >= -1e-4)
    assert 500 < np.median(q) < 1500  # de-normalized back to the input's scale


def test_finetune_forecaster_reduces_loss_on_periodic_signal() -> None:
    torch.manual_seed(0)
    m = SensorFM(CFG)
    t = np.arange(4000, dtype=np.float32)
    series = [np.sin(2 * np.pi * t / 32).astype(np.float32) * 3 + 10]
    losses = finetune_forecaster(m, series, horizon=32, steps=400, seed=0, device=DEV, lr=1e-3)
    assert np.mean(losses[-20:]) < 0.1 * np.mean(losses[:20])  # plateau, then phase-locks


def test_anomaly_scores_peak_at_injected_anomaly() -> None:
    torch.manual_seed(0)
    m = SensorFM(CFG)
    t = np.arange(6000, dtype=np.float32)
    x = (np.sin(2 * np.pi * t / 32) + 0.05 * np.random.default_rng(1).standard_normal(6000)).astype(
        np.float32
    )
    finetune_forecaster(m, [x[:3000]], horizon=16, steps=150, seed=0, device=DEV, lr=3e-3)
    x[4500:4532] += 3.0  # injected anomaly
    s = anomaly_scores(m, x, train_end=3000, device=DEV, norm_window=500)
    loc = int(np.nanargmax(np.where(np.arange(6000) >= 3000, s, np.nan)))
    assert 4400 < loc < 4650


def test_rolling_robust_z() -> None:
    x = np.r_[np.random.default_rng(0).standard_normal(500), 50.0]
    z = rolling_robust_z(x, 200)
    assert z[-1] > 10 and np.nanmax(np.abs(z[200:-1])) < 6


def test_load_encoder_from_training_checkpoint(tmp_path: Path) -> None:
    from sensorfm.pretrain import checkpoint as ckpt
    from sensorfm.utils.dist import DistInfo

    m = SensorFM(CFG)
    opt = torch.optim.AdamW(m.parameters())
    info = DistInfo(0, 1, 0, DEV)
    ckpt.save(
        tmp_path, 5, m, opt, {"model_config": CFG.__dict__, "fingerprint": {}}, info, False, 2
    )
    loaded = load_encoder(tmp_path / "step_00000005", CFG, DEV)
    for a, b in zip(m.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert FitCfg().epochs > 0
