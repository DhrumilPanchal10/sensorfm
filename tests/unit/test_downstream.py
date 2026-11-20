"""Task 1.5: downstream loaders and metrics (synthetic files in the verified real formats)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensorfm.downstream import bearing, cmapss, forecast_bench, ucr_anomaly
from sensorfm.eval import metrics

# --- C-MAPSS ----------------------------------------------------------------------------------


def _write_cmapss(root: Path, subset: str = "FD002", n_units: int = 6) -> None:
    rng = np.random.default_rng(0)
    conds = [(0.0, 0.0, 100.0), (42.0, 0.84, 100.0)]

    def frame(units: range, lengths: dict[int, int]) -> pd.DataFrame:
        rows = []
        for u in units:
            for c in range(1, lengths[u] + 1):
                op = conds[int(rng.integers(2))]
                base = 100.0 * (1 + conds.index(op))  # condition shifts sensor level
                sensors = base + rng.standard_normal(21) + 0.05 * c
                sensors[0] = 518.67  # constant sensor -> must be dropped
                rows.append([u, c, op[0] + 1e-4, op[1], op[2], *sensors])
        return pd.DataFrame(rows)

    tr_len = {u: int(rng.integers(40, 80)) for u in range(1, n_units + 1)}
    te_len = {u: int(rng.integers(10, 40)) for u in range(1, n_units + 1)}
    frame(range(1, n_units + 1), tr_len).to_csv(
        root / f"train_{subset}.txt", sep=" ", header=False, index=False
    )
    frame(range(1, n_units + 1), te_len).to_csv(
        root / f"test_{subset}.txt", sep=" ", header=False, index=False
    )
    np.savetxt(root / f"RUL_{subset}.txt", rng.integers(5, 200, n_units))


def test_cmapss_protocol(tmp_path: Path) -> None:
    _write_cmapss(tmp_path)
    d = cmapss.load_subset(tmp_path, "FD002")
    assert "s1" not in d.sensors and len(d.sensors) == 20
    assert d.train["rul"].max() == 125 or d.train["rul"].max() < 125
    assert d.train["rul"].min() == 0
    # Per-condition normalization removes the condition offset.
    for _, g in d.train.groupby("cond"):
        assert abs(g[d.sensors].to_numpy().mean()) < 0.2
    tr, va = cmapss.split_units(d.train["unit"], 0.34, seed=0)
    assert not set(tr) & set(va) and len(tr) + len(va) == 6
    x, m, y = cmapss.train_windows(d, tr, window=32)
    assert x.shape[1:] == (20, 32) and len(x) == len(y) == len(m)
    assert not m[0, :-1].any() and m[0, -1]  # first cycle: only last slot observed
    xt, _, yt = cmapss.test_windows(d, window=32)
    assert len(xt) == 6 and np.array_equal(
        yt, np.loadtxt(tmp_path / "RUL_FD002.txt").astype(np.float32)
    )
    assert cmapss.test_windows(d, 32, cap_test=True)[2].max() <= 125


def test_cmapss_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cmapss.load_subset(tmp_path, "FD001")


# --- UCR --------------------------------------------------------------------------------------


def test_ucr_parse_and_load(tmp_path: Path) -> None:
    p = tmp_path / "007_UCR_Anomaly_foo_bar_100_250_260.txt"
    p.write_text("\n".join(str(v) for v in np.sin(np.arange(400))))
    s = ucr_anomaly.load_file(p)
    assert (s.id, s.name, s.train_end, s.begin, s.end) == (7, "foo_bar", 100, 250, 260)
    assert len(s.train) == 100
    q = tmp_path / "008_UCR_Anomaly_one_line_10_20_25.txt"  # single-line variant
    q.write_text("  " + "  ".join(str(v) for v in range(40)))
    assert len(ucr_anomaly.load_file(q).values) == 40


def test_ucr_scoring_rule() -> None:
    # L = 11; window = (min(250-11, 250-100), max(260+11, 260+100)) = (150, 360), exclusive.
    assert ucr_anomaly.is_correct(255, 250, 260)
    assert ucr_anomaly.is_correct(151, 250, 260) and not ucr_anomaly.is_correct(150, 250, 260)
    assert ucr_anomaly.is_correct(359, 250, 260) and not ucr_anomaly.is_correct(360, 250, 260)
    # Long anomaly: L dominates the slack.
    assert ucr_anomaly.is_correct(1000 - 499, 1000, 1499)


def test_ucr_predict_location_ignores_train_part() -> None:
    scores = np.zeros(500)
    scores[10] = 99.0  # inside training prefix: must be ignored
    scores[300] = 5.0
    assert ucr_anomaly.predict_location(scores, train_end=100) == 301


# --- CWRU -------------------------------------------------------------------------------------


def test_cwru_file_table_is_complete() -> None:
    assert len(bearing.FILES) == 40
    for load in range(4):
        labels = sorted(c for c, ld in bearing.FILES.values() if ld == load)
        assert labels == sorted(bearing.CLASSES)


def test_cwru_windows_and_load_split(tmp_path: Path) -> None:
    rec = bearing.Recording(105, "inner_007", 0, 1797.0, np.arange(10_000, dtype=np.float32))
    w = bearing.make_windows(rec, 2048)
    assert w.shape == (4, 2048) and w[1, 0] == 2048
    with pytest.raises(ValueError):
        bearing.load_split(tmp_path, (0, 1), (1, 3))


def test_envelope_detects_synthetic_outer_race_fault() -> None:
    """Our diagnosis check works on the synthetic generator's bearing model (cross-check)."""
    from sensorfm.data.synthetic_physics import rotating_signal

    y, params = rotating_signal(np.random.default_rng(0), 120_000, bearing.FS, 29.95, "outer_race")
    peak = bearing.envelope_spectrum_peak(y, bearing.FS, 0.5 * 29.95, 6.5 * 29.95)
    harmonics = params["bpfo"] * np.arange(1, 4)
    assert np.min(np.abs(harmonics - peak) / harmonics) < 0.03


# --- forecasting ------------------------------------------------------------------------------


def test_ett_borders_and_rolling_windows(tmp_path: Path) -> None:
    assert forecast_bench.split_borders("ETTh1") == (8640, 11520, 14400)
    assert forecast_bench.split_borders("ETTm2") == (34560, 46080, 57600)
    n = 14400
    df = pd.DataFrame({"date": range(n), "OT": np.arange(n, dtype=np.float32)})
    df.to_csv(tmp_path / "ETTh1.csv", index=False)
    (s,) = forecast_bench.load_dataset(tmp_path, "ETTh1")
    ctx, tgt = forecast_bench.rolling_windows(s, context=512, horizon=96)
    assert ctx.shape[1] == 512 and tgt.shape[1] == 96
    assert tgt.min() >= s.val_end and tgt.max() < s.test_end  # targets only inside test
    np.testing.assert_array_equal(ctx[:, -1] + 1, tgt[:, 0])  # contiguous


# --- metrics ----------------------------------------------------------------------------------


def test_rul_metrics() -> None:
    assert metrics.rmse([1, 2], [1, 4]) == pytest.approx(np.sqrt(2))
    late = metrics.nasa_score([110], [100])
    early = metrics.nasa_score([90], [100])
    assert late == pytest.approx(np.e - 1) and early == pytest.approx(np.exp(10 / 13) - 1)
    assert late > early  # late predictions are penalized more


def test_forecast_metrics() -> None:
    insample = np.tile(np.arange(24.0), 10)
    assert metrics.seasonal_naive_scale(insample, 24) == 1.0  # periodic -> falls back to 1
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    qs = np.asarray(metrics.QUANTILES)
    from scipy.stats import norm

    perfect = np.broadcast_to(norm.ppf(qs), (5000, len(qs)))
    bad = perfect + 2.0
    assert metrics.crps_quantile(perfect, y) < metrics.crps_quantile(bad, y)
    # For Y ~ N(0,1) and the true quantiles z_q, E[pinball_q] = pdf(z_q) exactly, so the
    # 9-quantile approximation has expectation 2 * mean(pdf(z_q)) (~0.62, above the exact CRPS
    # 1/sqrt(pi) ~ 0.564 because the low-loss tails are not in the grid).
    expected = 2 * norm.pdf(norm.ppf(qs)).mean()
    assert metrics.crps_quantile(perfect, y) == pytest.approx(expected, abs=0.02)
    assert metrics.mase(y + 1, y, insample=np.arange(100.0), season=1) == pytest.approx(1.0)


def test_macro_f1() -> None:
    assert metrics.macro_f1([0, 1, 2], [0, 1, 2], 3) == 1.0
    assert metrics.macro_f1([0, 0, 0], [0, 1, 2], 3) == pytest.approx((0.5 + 0 + 0) / 3)
