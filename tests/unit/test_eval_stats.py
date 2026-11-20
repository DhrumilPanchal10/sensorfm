"""Paired bootstrap and slicing."""

from __future__ import annotations

import numpy as np

from sensorfm.eval.bootstrap import paired_bootstrap, rmse_from_sq
from sensorfm.eval.metrics import rmse
from sensorfm.eval.slices import by_key, rul_bucket


def test_paired_bootstrap_detects_real_difference_and_not_noise() -> None:
    rng = np.random.default_rng(0)
    difficulty = rng.exponential(10, 200)  # shared per-unit difficulty dominates
    a = difficulty + rng.normal(0, 0.5, 200) - 1.0  # A is better by 1
    b = difficulty + rng.normal(0, 0.5, 200)
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r.significant and r.hi < 0 and r.p_a_better > 0.99
    c = difficulty + rng.normal(0, 0.5, 200)
    assert not paired_bootstrap(c, b, n_boot=2000).significant


def test_bootstrap_rmse_aggregate_matches_metric() -> None:
    p, t = np.array([1.0, 5.0, 2.0]), np.array([2.0, 1.0, 2.0])
    r = paired_bootstrap((p - t) ** 2, np.zeros(3), aggregate=rmse_from_sq, n_boot=10)
    assert r.diff == rmse(p, t)


def test_slices() -> None:
    assert rul_bucket([3, 30, 130]) == ["0-25", "25-50", "125-inf"]
    s = by_key(["a", "a", "b"], [1, 2, 3], [1, 4, 3], rmse)
    assert s["a"]["n"] == 2 and s["b"]["metric"] == 0.0
