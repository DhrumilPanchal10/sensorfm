"""Synthetic physics simulators for sensor-like time series.

Why this exists
---------------
1. **Debug/CI data.** Every training path must run in < 2 min on CPU without downloading anything.
   Physically grounded signals are better debug data than random noise: they have spectra, damping
   and trends, so a model can visibly learn something in a few hundred steps.
2. **Pretraining ablation E5.** Real public sensor corpora under-represent vibration and
   degradation. Mixing 10-20% of these signals into pretraining tests whether simulated physics
   helps the industrial downstream tasks.

How it works
------------
Linear mass-spring-damper systems ``M x'' + C x' + K x = f(t)`` are written in state-space form
``s' = A s + B u`` and discretized **exactly** under a zero-order hold on the input:

    [Ad Bd]            ( [A B]      )
    [0  I ]  =  expm   ( [0 0] * dt )

so ``s[k+1] = Ad s[k] + Bd u[k]`` has no integration error for piecewise-constant forcing. This is
faster and more accurate than ``solve_ivp`` for linear systems, and it makes the physics tests
(energy decay, resonance location) tight. Slowly time-varying systems (degradation) are handled by
re-discretizing per chunk, which is accurate as long as parameters change slowly relative to the
chunk length.

Rotating machinery uses the standard bearing-fault model: an impulse train at the defect
characteristic frequency (BPFO/BPFI/BSF) exciting a lightly damped structural resonance, on top of
shaft-speed harmonics. All frequencies are drawn as fractions of the sampling rate, so every
generated series is alias-free regardless of the (randomized) sampling rate stored in metadata.

All functions take an explicit ``np.random.Generator``; nothing touches global RNG state.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from scipy.signal import chirp, fftconvolve

FloatArray = npt.NDArray[np.float64]
Observe = Literal["displacement", "velocity", "acceleration"]
Forcing = Literal["none", "impulse", "white_noise", "sweep"]
DefectKind = Literal["none", "outer_race", "inner_race", "ball"]

KINDS: tuple[str, ...] = ("oscillator", "chain", "rotating", "degradation")


@dataclass(frozen=True)
class SyntheticSeries:
    """One generated univariate series plus the parameters that produced it."""

    values: npt.NDArray[np.float32]
    fs: float
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Linear mass-spring-damper systems
# ---------------------------------------------------------------------------------------------


def chain_matrices(
    masses: FloatArray, stiffness: FloatArray, damping: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Mass, damping and stiffness matrices for a fixed-free chain of ``n`` masses.

    ``stiffness[0]`` connects mass 0 to the wall; ``stiffness[i]`` connects mass ``i-1`` to ``i``.
    Same topology for ``damping``.
    """
    n = len(masses)
    if not (len(stiffness) == len(damping) == n):
        raise ValueError("masses, stiffness and damping must have equal length")

    def assemble(e: FloatArray) -> FloatArray:
        mat = np.zeros((n, n))
        for i in range(n):
            mat[i, i] += e[i]
            if i + 1 < n:
                mat[i, i] += e[i + 1]
                mat[i, i + 1] -= e[i + 1]
                mat[i + 1, i] -= e[i + 1]
        return mat

    return np.diag(masses), assemble(damping), assemble(stiffness)


def modal_frequencies_hz(m: FloatArray, k: FloatArray) -> FloatArray:
    """Undamped natural frequencies (Hz), ascending."""
    eig = np.linalg.eigvals(np.linalg.solve(m, k)).real
    return np.sort(np.sqrt(np.clip(eig, 0.0, None)) / (2.0 * np.pi))


def discretize(
    m: FloatArray, c: FloatArray, k: FloatArray, force_dir: FloatArray, dt: float
) -> tuple[FloatArray, FloatArray]:
    """Exact ZOH discretization of ``M x'' + C x' + K x = force_dir * u(t)``.

    State is ``s = [x, v]``. Returns ``(Ad, Bd)`` with ``Bd`` of shape ``(2n,)``.
    """
    n = m.shape[0]
    minv = np.linalg.inv(m)
    a = np.zeros((2 * n, 2 * n))
    a[:n, n:] = np.eye(n)
    a[n:, :n] = -minv @ k
    a[n:, n:] = -minv @ c
    b = np.zeros(2 * n)
    b[n:] = minv @ force_dir
    aug = np.zeros((2 * n + 1, 2 * n + 1))
    aug[: 2 * n, : 2 * n] = a
    aug[: 2 * n, 2 * n] = b
    phi = expm(aug * dt)
    return phi[: 2 * n, : 2 * n], phi[: 2 * n, 2 * n]


def simulate_lti(ad: FloatArray, bd: FloatArray, u: FloatArray, s0: FloatArray) -> FloatArray:
    """Run ``s[k+1] = Ad s[k] + Bd u[k]``; returns states ``s[0..T-1]``, shape ``(T, 2n)``."""
    out = np.empty((len(u), len(s0)))
    s = s0.astype(np.float64).copy()
    for i in range(len(u)):
        out[i] = s
        s = ad @ s + bd * u[i]
    return out


def make_forcing(
    kind: Forcing, length: int, fs: float, rng: np.random.Generator, f_hi: float | None = None
) -> FloatArray:
    """Input signal ``u`` (piecewise constant per sample)."""
    if kind == "none":
        return np.zeros(length)
    if kind == "impulse":
        u = np.zeros(length)
        n_hits = int(rng.integers(1, 4))
        for idx in rng.integers(0, max(1, length // 2), size=n_hits):
            u[idx] = rng.uniform(0.5, 2.0) * fs  # unit-ish momentum over one sample
        return u
    if kind == "white_noise":
        return rng.standard_normal(length)
    if kind == "sweep":
        t = np.arange(length) / fs
        hi = f_hi if f_hi is not None else 0.4 * fs
        f0, f1 = rng.uniform(0.005, 0.05) * fs, hi
        return np.asarray(chirp(t, f0=f0, t1=t[-1], f1=f1, method="logarithmic"), np.float64)
    raise ValueError(f"unknown forcing {kind!r}")


def _observe(
    states: FloatArray,
    n: int,
    dof: int,
    u: FloatArray,
    a_row: FloatArray,
    b_acc: float,
    what: Observe,
) -> FloatArray:
    if what == "displacement":
        return states[:, dof]
    if what == "velocity":
        return states[:, n + dof]
    return states @ a_row + b_acc * u


def simulate_msd(
    masses: FloatArray,
    stiffness: FloatArray,
    damping: FloatArray,
    fs: float,
    u: FloatArray,
    force_dof: int = 0,
    sensor_dof: int = -1,
    observe: Observe = "displacement",
    x0: FloatArray | None = None,
    v0: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Simulate a chain; returns ``(sensor_signal, states)``."""
    m, c, k = chain_matrices(masses, stiffness, damping)
    n = len(masses)
    fdir = np.zeros(n)
    fdir[force_dof] = 1.0
    ad, bd = discretize(m, c, k, fdir, 1.0 / fs)
    s0 = np.concatenate(
        [
            x0 if x0 is not None else np.zeros(n),
            v0 if v0 is not None else np.zeros(n),
        ]
    )
    states = simulate_lti(ad, bd, u, s0)
    dof = sensor_dof % n
    minv = np.linalg.inv(m)
    a_row = np.concatenate([-(minv @ k)[dof], -(minv @ c)[dof]])
    y = _observe(states, n, dof, u, a_row, float((minv @ fdir)[dof]), observe)
    return y, states


def sdof_params(fn_hz: float, zeta: float, mass: float = 1.0) -> tuple[float, float]:
    """Stiffness and damping giving natural frequency ``fn_hz`` and damping ratio ``zeta``."""
    k = mass * (2.0 * np.pi * fn_hz) ** 2
    return k, 2.0 * zeta * np.sqrt(k * mass)


# ---------------------------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------------------------


def add_noise(x: FloatArray, snr_db: float, rng: np.random.Generator) -> FloatArray:
    """Add white Gaussian noise so that ``10 log10(P_signal / P_noise) == snr_db``.

    Signal power is measured about the mean so offsets do not inflate it.
    """
    p_sig = float(np.mean((x - x.mean()) ** 2))
    if p_sig == 0.0:
        return x.copy()
    sigma = float(np.sqrt(p_sig / 10.0 ** (snr_db / 10.0)))
    noisy: FloatArray = x + sigma * rng.standard_normal(len(x))
    return noisy


# ---------------------------------------------------------------------------------------------
# Generators (one per kind). Frequencies are fractions of fs → alias-free for any fs.
# ---------------------------------------------------------------------------------------------


def gen_oscillator(rng: np.random.Generator, length: int, fs: float) -> SyntheticSeries:
    """Single damped harmonic oscillator under random forcing."""
    fn = float(rng.uniform(0.005, 0.25) * fs)
    zeta = float(np.exp(rng.uniform(np.log(0.005), np.log(0.3))))
    k, c = sdof_params(fn, zeta)
    forcing: Forcing = rng.choice(np.array(["impulse", "white_noise", "sweep", "none"]))
    u = make_forcing(forcing, length, fs, rng)
    x0 = np.array([rng.uniform(-1, 1)]) if forcing in ("none", "impulse") else None
    observe: Observe = rng.choice(np.array(["displacement", "acceleration"]))
    y, _ = simulate_msd(np.ones(1), np.array([k]), np.array([c]), fs, u, observe=observe, x0=x0)
    return SyntheticSeries(
        y.astype(np.float32),
        fs,
        "oscillator",
        {"fn_hz": fn, "zeta": zeta, "forcing": forcing, "observe": observe},
    )


def random_chain(
    rng: np.random.Generator, n_dof: int, fs: float, f_max_frac: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Random chain rescaled so its highest mode sits at ``f_max_frac * fs``."""
    masses = rng.uniform(0.5, 2.0, n_dof)
    stiffness = rng.uniform(0.2, 5.0, n_dof)
    m, _, k = chain_matrices(masses, stiffness, np.zeros(n_dof))
    scale = (f_max_frac * fs / modal_frequencies_hz(m, k)[-1]) ** 2
    stiffness = stiffness * scale
    zeta = rng.uniform(0.005, 0.08, n_dof)
    damping = 2.0 * zeta * np.sqrt(stiffness * masses)
    _, _, k = chain_matrices(masses, stiffness, damping)
    return masses, stiffness, damping, modal_frequencies_hz(m, k)


def gen_chain(rng: np.random.Generator, length: int, fs: float) -> SyntheticSeries:
    """Coupled n-DOF mass-spring-damper chain, forced at one mass, sensed at another."""
    n_dof = int(rng.integers(2, 7))
    masses, stiffness, damping, modes = random_chain(rng, n_dof, fs, rng.uniform(0.1, 0.4))
    forcing: Forcing = rng.choice(np.array(["impulse", "white_noise", "sweep"]))
    u = make_forcing(forcing, length, fs, rng)
    force_dof, sensor_dof = int(rng.integers(n_dof)), int(rng.integers(n_dof))
    observe: Observe = rng.choice(np.array(["displacement", "acceleration"]))
    y, _ = simulate_msd(masses, stiffness, damping, fs, u, force_dof, sensor_dof, observe)
    return SyntheticSeries(
        y.astype(np.float32),
        fs,
        "chain",
        {
            "n_dof": n_dof,
            "modes_hz": modes.tolist(),
            "forcing": forcing,
            "observe": observe,
            "force_dof": force_dof,
            "sensor_dof": sensor_dof,
        },
    )


def bearing_frequencies(
    shaft_hz: float, n_balls: int, ball_d: float, pitch_d: float, contact_deg: float
) -> dict[str, float]:
    """Classic bearing characteristic frequencies (Hz)."""
    r = ball_d / pitch_d * np.cos(np.deg2rad(contact_deg))
    return {
        "bpfo": n_balls / 2.0 * shaft_hz * (1.0 - r),
        "bpfi": n_balls / 2.0 * shaft_hz * (1.0 + r),
        "bsf": pitch_d / (2.0 * ball_d) * shaft_hz * (1.0 - r**2),
        "ftf": shaft_hz / 2.0 * (1.0 - r),
    }


def rotating_signal(
    rng: np.random.Generator,
    length: int,
    fs: float,
    shaft_hz: float,
    defect: DefectKind,
    defect_amp: float | FloatArray = 1.0,
    n_harmonics: int = 5,
) -> tuple[FloatArray, dict[str, Any]]:
    """Shaft harmonics + optional bearing-defect impulse train exciting a resonance.

    ``defect_amp`` may be an array of length ``length`` (used by degradation to grow a fault).
    """
    t = np.arange(length) / fs
    y = np.zeros(length)
    for h in range(1, n_harmonics + 1):
        if h * shaft_hz >= 0.45 * fs:
            break
        y += (
            rng.uniform(0.2, 1.0)
            / h
            * np.sin(2 * np.pi * h * shaft_hz * t + rng.uniform(0, 2 * np.pi))
        )
    geom: dict[str, Any] = {
        "n_balls": int(rng.integers(7, 14)),
        "ball_d": float(rng.uniform(6, 12)),
        "pitch_d": float(rng.uniform(30, 60)),
        "contact_deg": float(rng.uniform(0, 15)),
    }
    freqs = bearing_frequencies(shaft_hz, **geom)
    params: dict[str, Any] = {"shaft_hz": shaft_hz, "defect": defect, **geom, **freqs}
    if defect == "none":
        return y, params
    f_def = {"outer_race": freqs["bpfo"], "inner_race": freqs["bpfi"], "ball": freqs["bsf"]}[defect]
    # Impulse times with ~1% random slip, as observed in real bearings.
    period = fs / f_def
    times = np.cumsum(period * (1.0 + 0.01 * rng.standard_normal(int(length / period) + 2)))
    idx = times[times < length].astype(int)
    imp = np.zeros(length)
    imp[idx] = 1.0
    if defect == "inner_race":  # load-zone amplitude modulation at shaft speed
        imp *= 1.0 + 0.8 * np.cos(2 * np.pi * shaft_hz * t)
    imp *= defect_amp
    f_res = float(rng.uniform(0.2, 0.4) * fs)
    zeta = float(rng.uniform(0.02, 0.08))
    tk = np.arange(int(min(length, 8 * fs / (2 * np.pi * zeta * f_res)))) / fs
    kernel = np.exp(-2 * np.pi * zeta * f_res * tk) * np.sin(2 * np.pi * f_res * tk)
    y += 3.0 * np.asarray(fftconvolve(imp, kernel)[:length])
    params.update({"defect_hz": f_def, "resonance_hz": f_res})
    return y, params


def gen_rotating(rng: np.random.Generator, length: int, fs: float) -> SyntheticSeries:
    """Rotating-machinery vibration: shaft harmonics and (maybe) a bearing defect."""
    shaft_hz = float(rng.uniform(0.002, 0.02) * fs)
    defect: DefectKind = rng.choice(np.array(["none", "outer_race", "inner_race", "ball"]))
    y, params = rotating_signal(rng, length, fs, shaft_hz, defect)
    return SyntheticSeries(y.astype(np.float32), fs, "rotating", params)


def degrading_oscillator(
    rng: np.random.Generator,
    length: int,
    fs: float,
    fn0: float,
    zeta: float,
    stiffness_loss: float,
    exponent: float,
    chunk: int = 128,
) -> tuple[FloatArray, FloatArray]:
    """SDOF under white-noise forcing whose stiffness drops by ``stiffness_loss`` over the run.

    Returns ``(displacement, health)`` with ``health = k(t)/k0`` per sample.
    """
    k0, _ = sdof_params(fn0, zeta)
    u = rng.standard_normal(length)
    y = np.empty(length)
    health = np.empty(length)
    s = np.zeros(2)
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        frac = (start + end) / 2 / length
        h = 1.0 - stiffness_loss * frac**exponent
        k = k0 * h
        c = 2.0 * zeta * np.sqrt(k)
        ad, bd = discretize(np.eye(1), np.array([[c]]), np.array([[k]]), np.ones(1), 1.0 / fs)
        st = simulate_lti(ad, bd, u[start:end], s)
        s = ad @ st[-1] + bd * u[end - 1]
        y[start:end] = st[:, 0]
        health[start:end] = h
    return y, health


def gen_degradation(rng: np.random.Generator, length: int, fs: float) -> SyntheticSeries:
    """Slow degradation: either stiffness loss (frequency drift) or a growing bearing fault."""
    if rng.random() < 0.5:
        fn0 = float(rng.uniform(0.02, 0.2) * fs)
        zeta = float(rng.uniform(0.01, 0.1))
        loss = float(rng.uniform(0.2, 0.6))
        expo = float(rng.uniform(1.0, 4.0))
        y, health = degrading_oscillator(rng, length, fs, fn0, zeta, loss, expo)
        params: dict[str, Any] = {
            "mode": "stiffness",
            "fn0_hz": fn0,
            "zeta": zeta,
            "stiffness_loss": loss,
            "exponent": expo,
        }
    else:
        onset = rng.uniform(0.1, 0.6)
        prog = np.clip((np.arange(length) / length - onset) / (1 - onset), 0, None)
        amp = prog ** rng.uniform(1.0, 3.0)
        y, params = rotating_signal(
            rng, length, fs, float(rng.uniform(0.002, 0.02) * fs), "outer_race", defect_amp=amp
        )
        params.update({"mode": "bearing_growth", "onset_frac": float(onset)})
        health = 1.0 - amp
    params["final_health"] = float(health[-1])
    return SyntheticSeries(y.astype(np.float32), fs, "degradation", params)


_GENERATORS = {
    "oscillator": gen_oscillator,
    "chain": gen_chain,
    "rotating": gen_rotating,
    "degradation": gen_degradation,
}


def sample_series(
    rng: np.random.Generator,
    length: int,
    kind: str | None = None,
    snr_db: float | None = None,
) -> SyntheticSeries:
    """Draw one series: random kind, fs log-uniform in [1 Hz, 25.6 kHz], SNR in [10, 40] dB."""
    kind = kind if kind is not None else str(rng.choice(np.array(KINDS)))
    fs = float(np.exp(rng.uniform(np.log(1.0), np.log(25600.0))))
    s = _GENERATORS[kind](rng, length, fs)
    snr = snr_db if snr_db is not None else float(rng.uniform(10.0, 40.0))
    noisy = add_noise(s.values.astype(np.float64), snr, rng).astype(np.float32)
    return SyntheticSeries(noisy, fs, kind, {**s.params, "snr_db": snr})


def generate(
    n: int, length: int, seed: int, kinds: tuple[str, ...] = KINDS
) -> Iterator[SyntheticSeries]:
    """Deterministic stream of ``n`` series. Series ``i`` depends only on ``(seed, i)``, so the
    stream can be split across workers/ranks without changing content."""
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        yield sample_series(rng, length, kind=kinds[i % len(kinds)])
