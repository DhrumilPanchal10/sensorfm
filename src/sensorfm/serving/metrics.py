"""Prometheus metrics for the scoring service (exposed at GET /metrics).

A private registry (not the global default) so tests and multiple app instances don't collide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info


@dataclass
class ServiceMetrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        r = self.registry
        buckets = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
        self.latency = Histogram(
            "sensorfm_score_latency_seconds", "Model latency per batch", buckets=buckets, registry=r
        )
        self.windows = Counter(
            "sensorfm_windows_scored_total", "Windows scored", ["task"], registry=r
        )
        self.alerts = Counter("sensorfm_alerts_total", "Alerts emitted", ["task"], registry=r)
        self.drift = Gauge(
            "sensorfm_drift_psi", "Input drift PSI per feature", ["feature"], registry=r
        )
        self.model = Info("sensorfm_model", "Deployed model version", registry=r)
