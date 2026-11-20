"""Task 5.3: windowing, drift, end-to-end stream replay with an exported model, FastAPI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient

from sensorfm.export.onnx_export import ForecastModule, export_onnx
from sensorfm.export.verify import ort_session
from sensorfm.finetune.common import ChannelIndependent, Regressor
from sensorfm.finetune.forecast import finetune_forecaster
from sensorfm.model.sensorfm import ModelConfig, SensorFM
from sensorfm.monitoring.drift import DriftMonitor, psi, window_features
from sensorfm.serving.app import create_app
from sensorfm.serving.metrics import ServiceMetrics
from sensorfm.serving.stream_consumer import (
    AnomalyScorer,
    Reading,
    RulScorer,
    StreamConsumer,
    queue_source,
    replay,
)
from sensorfm.serving.windowing import SlidingWindows

CFG = ModelConfig(context_length=128, dropout=0.0)


# --- windowing ---------------------------------------------------------------------------------


def test_windows_are_last_w_readings_in_order_with_hop() -> None:
    sw = SlidingWindows(window=5, hop=2, channels=2)
    emitted = []
    for i in range(12):
        w = sw.push("a", [i, 100 + i])
        if w is not None:
            emitted.append(w)
    assert [w.end_index for w in emitted] == [5, 7, 9, 11]
    for w in emitted:
        e = w.end_index
        np.testing.assert_array_equal(w.values[0], np.arange(e - 5, e))
        np.testing.assert_array_equal(w.values[1], 100 + np.arange(e - 5, e))


def test_windows_are_independent_per_asset() -> None:
    sw = SlidingWindows(window=3, hop=1)
    sw.push("a", [1])
    sw.push("b", [9])
    sw.push("a", [2])
    w = sw.push("a", [3])
    assert w is not None and w.values.tolist() == [[1, 2, 3]]
    assert sw.push("b", [8]) is None
    assert sw.assets() == ["a", "b"]


# --- drift -------------------------------------------------------------------------------------


def test_psi_zero_for_same_distribution_and_large_for_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(5000)
    edges = np.quantile(ref, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    assert psi(ref, rng.standard_normal(5000), edges) < 0.02
    assert psi(ref, rng.standard_normal(5000) + 1.5, edges) > 0.25


def test_drift_monitor_fires_on_operating_condition_change() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(64)

    def regime(level: float, amp: float, n: int) -> list[np.ndarray]:
        return [level + amp * np.sin(t / 3 + rng.uniform(0, 6)) + 0.1 * rng.standard_normal(64)
                for _ in range(n)]  # fmt: skip

    mon = DriftMonitor.from_windows(regime(0.0, 1.0, 300), buffer_size=200)
    for w in regime(0.0, 1.0, 200):
        mon.update(w)
    assert not mon.drifted()
    for w in regime(3.0, 2.0, 200):  # new operating condition: level and amplitude shift
        mon.update(w)
    assert mon.drifted()
    assert window_features(np.ones(64))[1] == 0.0


# --- end-to-end stream replay ------------------------------------------------------------------


@pytest.fixture(scope="module")
def forecast_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    m = SensorFM(CFG)
    t = np.arange(6000, dtype=np.float32)
    clean = (
        np.sin(2 * np.pi * t / 32) + 0.05 * np.random.default_rng(0).standard_normal(6000)
    ).astype(np.float32)
    finetune_forecaster(
        m, [clean[:3000]], horizon=16, steps=300, seed=0, device=torch.device("cpu"), lr=1e-3
    )
    path = tmp_path_factory.mktemp("onnx") / "forecast.onnx"
    fm = ForecastModule(m.eval(), context=112, horizon=16)
    return export_onnx(fm, (torch.zeros(2, 112),), ["context"], ["quantiles"], path)


def test_stream_replay_flags_injected_anomaly(forecast_onnx: Path) -> None:
    t = np.arange(4000, dtype=np.float32)
    x = (np.sin(2 * np.pi * t / 32) + 0.05 * np.random.default_rng(3).standard_normal(4000)).astype(
        np.float32
    )
    x[3000:3016] += 2.5  # injected anomaly
    scorer = AnomalyScorer(ort_session(forecast_onnx), context=112, horizon=16, threshold=4.0)
    drift = DriftMonitor.from_windows([x[i : i + 128] for i in range(0, 2000, 16)])
    consumer = StreamConsumer(scorer, SlidingWindows(window=128, hop=16), ServiceMetrics(), drift)

    async def go() -> None:
        q: asyncio.Queue[Reading | None] = asyncio.Queue(maxsize=256)
        await asyncio.gather(replay(q, "ucr-demo", ([v] for v in x)), consumer.run(queue_source(q)))

    asyncio.run(go())
    assert consumer.scored == (4000 - 128) // 16 + 1
    assert consumer.alerts, "anomaly not flagged"
    assert all(2990 <= a.end_index <= 3150 for a in consumer.alerts), [
        a.end_index for a in consumer.alerts
    ]
    assert set(consumer.drift_scores()) == {"mean", "std", "hf_energy"}


def test_rul_scorer_and_fastapi(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = Regressor(ChannelIndependent(SensorFM(CFG), 3), "full").eval()
    path = export_onnx(model, (torch.zeros(2, 3, 64), torch.ones(2, 64, dtype=torch.bool)),
                       ["x", "mask"], ["rul"], tmp_path / "rul.onnx")  # fmt: skip
    scorer = RulScorer(ort_session(path), threshold=1e9)  # every window alerts
    sw = SlidingWindows(window=64, hop=8, channels=3)
    windows = [w for i in range(100) if (w := sw.push("engine-1", np.full(3, i))) is not None]
    assert scorer.score(windows).shape == (len(windows),)

    client = TestClient(create_app(str(path), "rul", "test-v1"))
    assert client.get("/healthz").json() == {"status": "ok", "task": "rul", "version": "test-v1"}
    body = {"windows": np.random.default_rng(0).standard_normal((4, 3, 64)).tolist()}
    r = client.post("/score", json=body)
    assert r.status_code == 200 and len(r.json()["scores"]) == 4
    assert client.post("/score", json={"windows": [[1.0, 2.0]]}).status_code == 422
    metrics = client.get("/metrics").text
    assert 'sensorfm_windows_scored_total{task="rul"} 4.0' in metrics
    assert "sensorfm_score_latency_seconds_bucket" in metrics
    assert 'version="test-v1"' in metrics
