"""Physics checks for the synthetic generators (Task 1.2 acceptance)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert, welch

from sensorfm.data import synthetic_physics as sp


def _peak_hz(x: np.ndarray, fs: float, fmin: float = 0.0) -> float:
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(len(x), 4096))
    p[f < fmin] = 0.0
    return float(f[np.argmax(p)])


def test_free_vibration_energy_decays_monotonically() -> None:
    fs, fn, zeta = 1000.0, 20.0, 0.02
    k, c = sp.sdof_params(fn, zeta)
    _, states = sp.simulate_msd(
        np.ones(1), np.array([k]), np.array([c]), fs, np.zeros(4000), x0=np.array([1.0])
    )
    energy = 0.5 * k * states[:, 0] ** 2 + 0.5 * states[:, 1] ** 2
    # Viscous damping: dE/dt = -c v^2 <= 0, and the exact discretization preserves that.
    assert np.all(np.diff(energy) <= 1e-12 * energy[0])
    # Envelope decays as exp(-2 zeta wn t) in energy.
    t_end = len(energy) / fs
    expected = np.exp(-2 * zeta * 2 * np.pi * fn * t_end)
    assert energy[-1] / energy[0] == pytest.approx(expected, rel=0.5)


def test_free_vibration_oscillates_at_damped_frequency() -> None:
    fs, fn, zeta = 2000.0, 37.0, 0.01
    k, c = sp.sdof_params(fn, zeta)
    y, _ = sp.simulate_msd(
        np.ones(1), np.array([k]), np.array([c]), fs, np.zeros(8000), x0=np.array([1.0])
    )
    crossings = np.where(np.diff(np.signbit(y)))[0]
    period = 2 * np.mean(np.diff(crossings)) / fs
    fd = fn * np.sqrt(1 - zeta**2)
    assert 1 / period == pytest.approx(fd, rel=0.01)


@pytest.mark.parametrize("fn", [12.0, 50.0, 180.0])
def test_fft_peak_at_natural_frequency_under_noise(fn: float) -> None:
    fs, zeta = 1000.0, 0.02
    k, c = sp.sdof_params(fn, zeta)
    u = np.random.default_rng(0).standard_normal(2**15)
    y, _ = sp.simulate_msd(np.ones(1), np.array([k]), np.array([c]), fs, u)
    bin_hz = fs / 4096
    assert abs(_peak_hz(y, fs) - fn) <= max(2 * bin_hz, 0.02 * fn)


def test_chain_peaks_match_modes() -> None:
    rng = np.random.default_rng(3)
    fs = 1000.0
    masses, stiffness, damping, modes = sp.random_chain(rng, 3, fs, 0.3)
    u = rng.standard_normal(2**15)
    for observe in ("displacement", "acceleration"):
        y, _ = sp.simulate_msd(masses, stiffness, damping, fs, u, 0, 2, observe)  # type: ignore[arg-type]
        peak = _peak_hz(y, fs)
        assert np.min(np.abs(modes - peak)) <= max(2 * fs / 4096, 0.03 * peak)


def test_acceleration_matches_finite_difference_of_velocity() -> None:
    fs = 5000.0
    k, c = sp.sdof_params(30.0, 0.05)
    u = np.sin(2 * np.pi * 7.0 * np.arange(5000) / fs)
    acc, states = sp.simulate_msd(
        np.ones(1), np.array([k]), np.array([c]), fs, u, observe="acceleration"
    )
    fd = np.gradient(states[:, 1], 1 / fs)
    assert (
        np.corrcoef(acc[10:-10], fd[10:-10])[0, 1] > 0.995
    )  # ZOH input is half a sample offset from central diff


def test_add_noise_hits_requested_snr() -> None:
    rng = np.random.default_rng(1)
    x = np.sin(np.linspace(0, 200 * np.pi, 200_000))
    for snr in (10.0, 25.0, 40.0):
        noisy = sp.add_noise(x, snr, rng)
        measured = 10 * np.log10(np.mean(x**2) / np.mean((noisy - x) ** 2))
        assert measured == pytest.approx(snr, abs=0.2)


def test_rotating_shaft_harmonic_and_outer_race_envelope() -> None:
    rng = np.random.default_rng(5)
    fs, length = 12000.0, 2**16
    y, params = sp.rotating_signal(rng, length, fs, shaft_hz=30.0, defect="outer_race")
    # Raw spectrum below the resonance band is dominated by the shaft fundamental.
    f, p = welch(y, fs=fs, nperseg=8192)
    low = f < 0.1 * fs
    assert abs(f[low][np.argmax(p[low])] - 30.0) < 2 * fs / 8192
    # Envelope spectrum shows BPFO: the textbook bearing-diagnosis signature.
    env = np.abs(hilbert(y - np.convolve(y, np.ones(64) / 64, mode="same")))
    peak = _peak_hz(env, fs, fmin=0.5 * params["bpfo"])
    harmonics = params["bpfo"] * np.arange(1, 4)
    assert np.min(np.abs(harmonics - peak) / harmonics) < 0.03


def test_bearing_frequency_identities() -> None:
    f = sp.bearing_frequencies(30.0, 9, 8.0, 40.0, 0.0)
    assert f["bpfo"] + f["bpfi"] == pytest.approx(9 * 30.0)
    assert f["ftf"] * 9 == pytest.approx(f["bpfo"])


def test_stiffness_degradation_lowers_frequency() -> None:
    rng = np.random.default_rng(2)
    fs, length = 1000.0, 2**16
    y, health = sp.degrading_oscillator(rng, length, fs, 100.0, 0.02, 0.5, 1.0)
    q = length // 4
    f_start, f_end = _peak_hz(y[:q], fs), _peak_hz(y[-q:], fs)
    assert health[0] > 0.99 and health[-1] == pytest.approx(0.5, abs=0.01)
    expected_end = 100.0 * np.sqrt(np.mean(health[-q:]))
    assert f_end < f_start
    assert f_end == pytest.approx(expected_end, rel=0.05)


@pytest.mark.parametrize("kind", sp.KINDS)
def test_every_kind_is_finite_float32_and_right_length(kind: str) -> None:
    for seed in range(5):
        s = sp.sample_series(np.random.default_rng(seed), 1024, kind=kind)
        assert s.values.dtype == np.float32
        assert s.values.shape == (1024,)
        assert np.all(np.isfinite(s.values))
        assert s.values.std() > 0
        assert 10.0 <= s.params["snr_db"] <= 40.0


def test_generate_is_deterministic_and_index_addressable() -> None:
    a = list(sp.generate(8, 512, seed=11))
    b = list(sp.generate(8, 512, seed=11))
    assert all(np.array_equal(x.values, y.values) for x, y in zip(a, b, strict=True))
    assert {s.kind for s in a} == set(sp.KINDS)
    c = list(sp.generate(8, 512, seed=12))
    assert not np.array_equal(a[0].values, c[0].values)
