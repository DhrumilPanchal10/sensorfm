"""Tasks 2.1/2.2: model shapes, no information leakage, objectives, FLOPs."""

from __future__ import annotations

import pytest
import torch

from sensorfm.model.heads import PatchQuantileHead
from sensorfm.model.patching import num_patches, patchify, unpatchify
from sensorfm.model.revin import RevIN
from sensorfm.model.sensorfm import ModelConfig, SensorFM, count_params, non_embedding_params
from sensorfm.pretrain.objectives import ObjectiveConfig, make_hide_masks, pretrain_step
from sensorfm.utils.flops import mfu, peak_flops, train_flops_per_token

TINY = ModelConfig(context_length=128, dropout=0.0)


def _batch(b: int = 4, length: int = 128, pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, length, generator=g).cumsum(-1) * 5 + 100
    obs = torch.ones(b, length, dtype=torch.bool)
    if pad:
        x[:, :pad] = 0
        obs[:, :pad] = False
    return x, obs


def test_patching_roundtrip() -> None:
    x = torch.arange(64.0).reshape(2, 32)
    p = patchify(x, 16, 16)
    assert p.shape == (2, 2, 16)
    assert torch.equal(unpatchify(p, 16), x)
    with pytest.raises(ValueError):
        num_patches(30, 16, 16)


def test_revin_roundtrip_and_visible_only_stats() -> None:
    r = RevIN(affine=True)
    x = torch.randn(3, 64) * 50 + 7
    vis = torch.ones_like(x, dtype=torch.bool)
    vis[:, 32:] = False
    xn, loc, _ = r.normalize(x, vis)
    assert torch.allclose(loc[:, 0], x[:, :32].mean(1), atol=1e-4)
    assert torch.all(xn[:, 32:] == 0)
    back = r.denormalize(
        r.normalize(x, torch.ones_like(vis))[0], *r.normalize(x, torch.ones_like(vis))[1:]
    )
    assert torch.allclose(back, x, atol=1e-3)


def test_forward_shapes() -> None:
    m = SensorFM(TINY)
    x, obs = _batch()
    out = m(x, obs)
    n, p, q = TINY.n_patches, TINY.patch_len, len(TINY.quantiles)
    assert out.hidden.shape == (4, n, TINY.d_model)
    assert out.recon.shape == (4, n, p)
    assert out.quantiles.shape == (4, n, p, q)
    assert m.embed_series(x, obs).shape == (4, TINY.d_model)


def test_no_leak_from_hidden_patches_or_padding() -> None:
    torch.manual_seed(0)
    m = SensorFM(TINY).eval()
    x, obs = _batch(pad=20)
    hide = torch.zeros(4, TINY.n_patches, dtype=torch.bool)
    hide[:, -2:] = True  # forecast horizon
    hide[:, 3] = True  # one masked context patch
    out1 = m(x, obs, hide)
    x2 = x.clone()
    x2[:, -32:] += torch.randn(4, 32) * 1000  # change hidden horizon values
    x2[:, 48:64] -= 777.0  # change hidden context patch
    x2[:, :20] = 12345.0  # change padded values
    out2 = m(x2, obs, hide)
    for a, b in (
        (out1.hidden, out2.hidden),
        (out1.quantiles, out2.quantiles),
        (out1.loc, out2.loc),
    ):
        assert torch.allclose(a, b, atol=1e-5)


def test_quantiles_are_monotone() -> None:
    head = PatchQuantileHead(8, 4, (0.1, 0.5, 0.9))
    q = head(torch.randn(5, 3, 8) * 10)
    assert torch.all(q[..., 1:] >= q[..., :-1])


def test_hide_masks() -> None:
    _, obs = _batch(b=64, length=512, pad=100)
    masked, horizon = make_hide_masks(
        obs, 16, ObjectiveConfig(mask_ratio=0.4, horizon=64, alpha=0.5)
    )
    assert horizon[:, -4:].all() and not horizon[:, :-4].any()
    assert not (masked & horizon).any()
    padded_patches = torch.arange(32) < (100 // 16)  # fully padded patches 0..5
    assert not masked[:, padded_patches].any()
    eligible = 32 - 4 - 6  # minus horizon, minus fully padded (patch 6 is partially observed)
    assert torch.all(masked.sum(1) == int(eligible * 0.4))
    m1, h1 = make_hide_masks(obs, 16, ObjectiveConfig(alpha=1.0))
    assert not h1.any() and m1.any()
    m0, h0 = make_hide_masks(obs, 16, ObjectiveConfig(alpha=0.0))
    assert not m0.any() and h0.any()


def test_pretrain_step_overfits_one_batch() -> None:
    torch.manual_seed(0)
    m = SensorFM(TINY)
    x, obs = _batch(b=8)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    cfg = ObjectiveConfig(mask_ratio=0.25, horizon=32, alpha=0.5)
    first = None
    for i in range(150):
        g = torch.Generator().manual_seed(0)  # fixed masks -> pure overfitting test
        losses = pretrain_step(m, x, obs, cfg, g)
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        if i == 0:
            first = {k: float(v.detach()) for k, v in losses.items()}
    assert first is not None
    assert float(losses["masked"].detach()) < 0.3 * first["masked"]
    assert float(losses["forecast"].detach()) < 0.5 * first["forecast"]


@pytest.mark.parametrize(
    ("name", "d", "layers", "heads"),
    [("s20m", 384, 8, 6), ("m60m", 512, 16, 8), ("l150m", 768, 18, 12)],
)
def test_param_counts_in_expected_range(name: str, d: int, layers: int, heads: int) -> None:
    cfg = ModelConfig(name=name, d_model=d, n_layers=layers, n_heads=heads)
    with torch.device("meta"):
        m = SensorFM(cfg)
    n = count_params(m)
    expected = 12 * d * d * layers
    assert 0.95 * expected < n < 1.1 * expected


def test_flops_and_mfu() -> None:
    m = SensorFM(TINY)
    f = train_flops_per_token(non_embedding_params(m), 2, TINY.n_patches, 64)
    assert f > 6 * non_embedding_params(m)
    assert peak_flops("Tesla T4", "fp16") == 65e12
    assert peak_flops("Tesla T4", "bf16") == 65e12  # falls back to fp16 tensor peak
    assert peak_flops("Tesla P100-PCIE-16GB", "fp32") == 9.3e12
    assert peak_flops("Apple M2", "fp32") is None
    assert mfu(1e6, 1e6, 1e13) == pytest.approx(0.1)


@pytest.mark.parametrize("name", ["tiny", "s20m", "m60m", "l150m"])
def test_model_yaml_configs_build(name: str) -> None:
    """Regression: a shell loop once wrote these files with empty fields."""
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(Path(f"configs/model/{name}.yaml").read_text())
    assert raw["name"] == name and isinstance(raw["d_model"], int)
    with torch.device("meta"):
        SensorFM(ModelConfig(**raw))
