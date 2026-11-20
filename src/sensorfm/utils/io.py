"""Run logging: always a local JSONL file, plus MLflow when installed and enabled.

The JSONL file (``runs/<id>/metrics.jsonl``) is the source of truth for tests, reports and the
resume check — it works offline, on Kaggle, and needs no server. MLflow (file store by default)
mirrors it for browsing. Appending (not overwriting) means a resumed run continues the same file;
duplicated steps after a crash are resolved by "last write wins" in ``read_metrics``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(
        self, run_dir: Path, params: dict[str, Any], use_mlflow: bool, enabled: bool = True
    ) -> None:
        self.enabled = enabled
        self.path = run_dir / "metrics.jsonl"
        self._mlflow: Any = None
        if not enabled:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
        if use_mlflow:
            try:
                import mlflow

                mlflow.set_experiment("sensorfm")
                mlflow.start_run(run_name=run_dir.name)
                mlflow.log_params(_flatten(params))
                self._mlflow = mlflow
            except ImportError:
                pass

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        with self.path.open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics, step=step)

    def close(self) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v)[:250]
    return out


def read_metrics(path: Path) -> dict[int, dict[str, float]]:
    """step -> metrics, last write wins (handles re-logged steps after resume)."""
    rows: dict[int, dict[str, float]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows.setdefault(int(r["step"]), {}).update(r)
    return rows
