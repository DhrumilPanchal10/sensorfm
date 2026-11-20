"""FastAPI scoring service.

    SENSORFM_MODEL=exports/rul_int8.onnx SENSORFM_TASK=rul uvicorn sensorfm.serving.app:app

Endpoints: ``POST /score`` (batch of windows), ``GET /healthz``, ``GET /metrics`` (Prometheus:
latency histogram, windows scored, alerts, drift PSI, model version). The ONNX session is created
once at startup; requests are validated with Pydantic so malformed windows are rejected with 422
before reaching the model.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from sensorfm.serving.metrics import ServiceMetrics


class ScoreRequest(BaseModel):
    windows: list[list[list[float]]] = Field(
        ..., description="(B, C, W) for rul; (B, 1, c) for forecast"
    )
    mask: list[list[bool]] | None = None


class ScoreResponse(BaseModel):
    scores: list[Any]
    model_version: str
    latency_ms: float


def create_app(
    model_path: str | None = None, task: str | None = None, version: str | None = None
) -> FastAPI:
    from sensorfm.export.verify import ort_session

    model_path = model_path or os.environ.get("SENSORFM_MODEL", "")
    task = task or os.environ.get("SENSORFM_TASK", "rul")
    version = version or os.environ.get("SENSORFM_VERSION", os.path.basename(model_path))
    metrics = ServiceMetrics()
    metrics.model.info({"version": version, "task": task, "path": model_path})
    sess = ort_session(Path(model_path)) if model_path else None
    app = FastAPI(title="SensorFM scoring service")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok" if sess is not None else "no_model",
            "task": task,
            "version": version,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        if sess is None:
            raise HTTPException(503, "no model loaded")
        x = np.asarray(req.windows, dtype=np.float32)
        if x.ndim != 3 or x.shape[0] == 0:
            raise HTTPException(422, "windows must be a non-empty (B, C, W) array")
        t0 = time.perf_counter()
        if task == "rul":
            mask = (np.asarray(req.mask, dtype=bool) if req.mask is not None
                    else np.ones((x.shape[0], x.shape[2]), dtype=bool))  # fmt: skip
            out = sess.run(None, {"x": x, "mask": mask})[0].reshape(-1).tolist()
        else:
            out = sess.run(None, {"context": x[:, 0, :]})[0].tolist()
        dt = time.perf_counter() - t0
        metrics.latency.observe(dt)
        metrics.windows.labels(task).inc(x.shape[0])
        return ScoreResponse(scores=out, model_version=version, latency_ms=dt * 1000)

    @app.get("/metrics")
    def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.state.metrics = metrics
    return app


app = create_app() if os.environ.get("SENSORFM_MODEL") else FastAPI(title="SensorFM (no model)")
