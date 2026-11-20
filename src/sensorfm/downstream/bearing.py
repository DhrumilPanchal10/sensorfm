"""CWRU bearing fault classification (Case Western Reserve University Bearing Data Center).

Why CWRU (vs Paderborn): freely downloadable per file without registration, by far the most used
benchmark (so results are comparable), and small (~150 MB for the subset used). Its known weakness —
windows from one recording are near-duplicates, so window-level random splits leak and give ~100%
accuracy — is exactly why we split **by load condition**: train on some motor loads, test on a load
never seen in training (different speed, different spectra). See ``docs/dataset_card.md``.

Subset: 12 kHz drive-end accelerometer (``*_DE_time``), 10 classes = normal + {inner race, ball,
outer race @6:00} x {0.007", 0.014", 0.021"}, each at loads 0-3 hp (file numbers verified against
the Bearing Data Center's 12k drive-end table on 2026-11-20). Normal-baseline files are recorded at
48 kHz (verified via shaft-harmonic location, see ``verify_file``) and are decimated to 12 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import decimate, hilbert, welch

BASE_URL = "https://engineering.case.edu/sites/default/files/{n}.mat"
FS = 12_000.0

# 6205-2RS JEM SKF drive-end bearing defect frequencies, as multiples of shaft speed (CWRU site).
DEFECT_ORDERS = {"inner": 5.4152, "outer": 3.5848, "ball": 2.3568, "cage": 0.39828}

CLASSES = ["normal", "inner_007", "ball_007", "outer_007", "inner_014", "ball_014", "outer_014",
           "inner_021", "ball_021", "outer_021"]  # fmt: skip

# file number -> (class, load_hp)
FILES: dict[int, tuple[str, int]] = {}
for _load, _nums in enumerate([
    (97, 105, 118, 130, 169, 185, 197, 209, 222, 234),
    (98, 106, 119, 131, 170, 186, 198, 210, 223, 235),
    (99, 107, 120, 132, 171, 187, 199, 211, 224, 236),
    (100, 108, 121, 133, 172, 188, 200, 212, 225, 237),
]):  # fmt: skip
    for _cls, _n in zip(CLASSES, _nums, strict=True):
        FILES[_n] = (_cls, _load)

NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


@dataclass
class Recording:
    number: int
    label: str
    load_hp: int
    rpm: float
    signal: npt.NDArray[np.float32]  # drive-end acceleration at 12 kHz


def load_recording(path: str | Path, number: int) -> Recording:
    import scipy.io as sio

    m = sio.loadmat(str(path))
    # Some files carry extra channels from neighbouring recordings (e.g. 99.mat also has X098_*),
    # so prefer the key matching the file number.
    own = f"X{number:03d}"
    de = [k for k in m if k.endswith("_DE_time")]
    if len(de) > 1:
        de = [k for k in de if k.startswith(own)]
    if len(de) != 1:
        raise ValueError(f"{path}: cannot pick a *_DE_time key from {de}")
    x = np.asarray(m[de[0]], dtype=np.float64).ravel()
    rpm_keys = [k for k in m if k.endswith("RPM")]
    if len(rpm_keys) > 1:
        rpm_keys = [k for k in rpm_keys if k.startswith(own)]
    label, load = FILES[number]
    rpm = float(np.asarray(m[rpm_keys[0]]).ravel()[0]) if rpm_keys else float(NOMINAL_RPM[load])
    if label == "normal":
        x = decimate(x, 4, zero_phase=True)  # 48 kHz -> 12 kHz
    return Recording(number, label, load, rpm, x.astype(np.float32))


def envelope_spectrum_peak(
    x: npt.NDArray[np.floating], fs: float, fmin: float, fmax: float
) -> float:
    """Frequency of the largest envelope-spectrum peak in [fmin, fmax] (bearing diagnosis)."""
    xd = np.asarray(x, np.float64) - np.mean(x)
    env = np.abs(np.asarray(hilbert(xd)))
    f, p = welch(env - env.mean(), fs=fs, nperseg=min(len(env), 2**15))
    band = (f >= fmin) & (f <= fmax)
    return float(f[band][np.argmax(p[band])])


def verify_file(rec: Recording, raw_fs_normal: float = 48_000.0) -> dict[str, float | str | bool]:
    """Physics sanity check of a recording's label.

    - load: reported RPM within 2% of the nominal RPM for its load.
    - faults: the envelope-spectrum peak between 0.5x and 6x shaft speed is within 3% of a low
      harmonic (1x-3x) of the labeled defect frequency. Ball faults are notoriously weak in CWRU, so
      their check is reported but not required to pass.
    """
    shaft = rec.rpm / 60.0
    out: dict[str, float | str | bool] = {"number": rec.number, "label": rec.label,
                                          "load_hp": rec.load_hp, "rpm": rec.rpm}  # fmt: skip
    out["rpm_ok"] = abs(rec.rpm - NOMINAL_RPM[rec.load_hp]) / NOMINAL_RPM[rec.load_hp] < 0.02
    if rec.label == "normal":
        # Shaft 1x should be visible at the right place only if the 48 kHz assumption holds.
        f, p = welch(rec.signal.astype(np.float64), fs=FS, nperseg=2**15)
        band = (f > 0.5 * shaft) & (f < 1.5 * shaft)
        out["peak_hz"] = float(f[band][np.argmax(p[band])])
        out["expected_hz"] = shaft
        out["signature_ok"] = abs(float(out["peak_hz"]) - shaft) / shaft < 0.05
        return out
    kind = rec.label.split("_")[0]
    f_def = DEFECT_ORDERS[kind] * shaft
    peak = envelope_spectrum_peak(rec.signal, FS, 0.5 * shaft, 6.5 * shaft)
    harmonics = f_def * np.arange(1, 4)
    out["peak_hz"] = peak
    out["expected_hz"] = f_def
    out["signature_ok"] = bool(np.min(np.abs(harmonics - peak) / harmonics) < 0.03)
    return out


def make_windows(
    rec: Recording, length: int = 2048, stride: int | None = None
) -> npt.NDArray[np.float32]:
    """Non-overlapping (default) windows of one recording, shape (n, length)."""
    stride = stride or length
    n = (len(rec.signal) - length) // stride + 1
    idx = np.arange(length)[None, :] + stride * np.arange(n)[:, None]
    out: npt.NDArray[np.float32] = rec.signal[idx]
    return out


def load_windows(
    root: str | Path, loads: tuple[int, ...], length: int = 2048
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Windows + integer labels of every recording at the given motor loads."""
    xs, ys = [], []
    for n, (label, load) in sorted(FILES.items()):
        if load not in loads:
            continue
        path = Path(root) / f"{n}.mat"
        if not path.exists():
            raise FileNotFoundError(f"missing CWRU file {path}; run scripts/prepare_downstream.py")
        w = make_windows(load_recording(path, n), length)
        xs.append(w)
        ys.append(np.full(len(w), CLASSES.index(label), dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def load_split(
    root: str | Path, train_loads: tuple[int, ...], test_loads: tuple[int, ...], length: int = 2048
) -> dict[str, tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]]:
    """Windows + integer labels for train/test, split by load (no recording in both)."""
    if set(train_loads) & set(test_loads):
        raise ValueError("train and test loads overlap — that leaks recordings")
    return {
        "train": load_windows(root, train_loads, length),
        "test": load_windows(root, test_loads, length),
    }
