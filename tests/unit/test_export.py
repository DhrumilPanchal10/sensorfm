"""Task 5.1: ONNX export parity and INT8 quantization on tiny models (CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")

from sensorfm.export.onnx_export import ForecastModule, export_onnx
from sensorfm.export.ort_quant import (
    model_size_mb,
    quantize_dynamic_int8,
    quantize_static_int8,
)
from sensorfm.export.verify import compare, ort_session, run_ort
from sensorfm.finetune.common import ChannelIndependent, Regressor
from sensorfm.finetune.forecast import forecast_quantiles
from sensorfm.model.sensorfm import ModelConfig, SensorFM

CFG = ModelConfig(context_length=128, dropout=0.0, d_model=64, n_layers=2)


def _rul_inputs(b: int, pad: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((b, 3, 64)).astype(np.float32)
    m = np.ones((b, 64), dtype=bool)
    if pad:
        m[:, :pad] = False
        x[:, :, :pad] = 0
    return {"x": x, "mask": m}


def test_rul_export_matches_torch_dynamic_batch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = Regressor(ChannelIndependent(SensorFM(CFG), 3), "full").eval()
    path = export_onnx(model, tuple(torch.as_tensor(v) for v in _rul_inputs(2).values()),
                       ["x", "mask"], ["rul"], tmp_path / "rul.onnx")  # fmt: skip
    for b, pad in ((1, 0), (7, 0), (5, 20)):  # other batch sizes + padded (masked) windows
        d = compare(model, path, _rul_inputs(b, pad))
        assert d["rel_mean"] < 1e-4, (b, pad, d)


def test_forecast_export_matches_quantile_forecast(tmp_path: Path) -> None:
    torch.manual_seed(0)
    sfm = SensorFM(CFG).eval()
    fm = ForecastModule(sfm, context=96, horizon=32)
    ctx = (np.random.default_rng(1).standard_normal((4, 96)) * 20 + 300).astype(np.float32)
    path = export_onnx(fm, (torch.as_tensor(ctx),), ["context"], ["quantiles"], tmp_path / "f.onnx")
    got = run_ort(ort_session(path), {"context": ctx})
    ref = forecast_quantiles(sfm, ctx, 32, torch.device("cpu"))
    assert got.shape == (4, 32, len(CFG.quantiles))
    assert np.abs(got - ref).mean() / np.abs(ref).mean() < 1e-4


def test_int8_variants_run_and_shrink(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(context_length=128, dropout=0.0, d_model=128, n_layers=2)
    model = Regressor(ChannelIndependent(SensorFM(cfg), 3), "full").eval()
    fp32 = export_onnx(model, tuple(torch.as_tensor(v) for v in _rul_inputs(2).values()),
                       ["x", "mask"], ["rul"], tmp_path / "fp32.onnx")  # fmt: skip
    dyn = quantize_dynamic_int8(fp32, tmp_path / "dyn.onnx")
    calib = _rul_inputs(64)
    sta = quantize_static_int8(fp32, tmp_path / "static.onnx", calib)
    inp = _rul_inputs(8)
    ref = run_ort(ort_session(fp32), inp)
    for q in (dyn, sta):
        out = run_ort(ort_session(q), inp)
        assert out.shape == ref.shape and np.all(np.isfinite(out))
        assert model_size_mb(q) < 0.6 * model_size_mb(fp32)
