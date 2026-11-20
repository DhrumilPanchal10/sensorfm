"""Analytic training FLOPs and model FLOPs utilization (MFU).

FLOPs per token (Kaplan et al. 2020, "Scaling Laws", Table 1; same accounting as PaLM's MFU):

    forward  = 2 * N  +  2 * n_layers * n_ctx * d_model     (matmuls + attention scores/values)
    training = 3 * forward                                   (backward ~ 2x forward)
             = 6 * N  +  6 * n_layers * n_ctx * d_model

where N = non-embedding parameters and n_ctx = tokens per sequence (patches here, not points).
MFU = achieved model FLOPs/s / peak FLOPs/s of the hardware at the precision used. MFU counts only
the useful model math, so activation recomputation or padding waste lowers it (by design).

Peak numbers are **dense** (no structured-sparsity 2x) tensor-core figures from NVIDIA datasheets:
T4 (Turing): 65 TFLOPS fp16 tensor, 8.1 TFLOPS fp32, no bf16.
P100 (Pascal, PCIe 16 GB): 18.7 TFLOPS fp16 (no tensor cores), 9.3 TFLOPS fp32, no bf16.
V100 (SXM2): 125 fp16 tensor, 15.7 fp32. A100 (SXM): 312 bf16/fp16, 19.5 fp32 (156 tf32).
L4: 121 bf16/fp16 dense, 30.3 fp32. H100 SXM: 989 bf16/fp16 dense, 67 fp32.
Free-tier Kaggle hardware is T4 / P100; the rest are for comparison only.
"""

from __future__ import annotations

PEAK_TFLOPS: dict[str, dict[str, float]] = {
    "T4": {"fp16": 65.0, "fp32": 8.1},
    "P100": {"fp16": 18.7, "fp32": 9.3},
    "V100": {"fp16": 125.0, "fp32": 15.7},
    "A100": {"bf16": 312.0, "fp16": 312.0, "fp32": 19.5},
    "L4": {"bf16": 121.0, "fp16": 121.0, "fp32": 30.3},
    "H100": {"bf16": 989.0, "fp16": 989.0, "fp32": 67.0},
}


def train_flops_per_token(
    non_embedding_params: int, n_layers: int, n_ctx: int, d_model: int
) -> float:
    return 6.0 * non_embedding_params + 6.0 * n_layers * n_ctx * d_model


def match_gpu(device_name: str) -> str | None:
    """Map a CUDA device name (e.g. 'Tesla T4') to a PEAK_TFLOPS key."""
    name = device_name.upper()
    for key in PEAK_TFLOPS:
        if key in name:
            return key
    return None


def peak_flops(device_name: str, precision: str) -> float | None:
    key = match_gpu(device_name)
    if key is None:
        return None
    table = PEAK_TFLOPS[key]
    prec = "fp16" if precision == "bf16" and "bf16" not in table else precision
    if prec not in table:
        return None
    return table[prec] * 1e12


def mfu(
    tokens_per_s: float, flops_per_token: float, peak: float | None, n_gpus: int = 1
) -> float | None:
    if not peak:
        return None
    return tokens_per_s * flops_per_token / (peak * n_gpus)
