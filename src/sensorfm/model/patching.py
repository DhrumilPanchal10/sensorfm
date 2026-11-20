"""Patching: turn a length-L series into N = (L - P) / S + 1 tokens of P points each.

Patches (Nie et al., PatchTST 2023) are what make transformers viable on long sensor windows:
attention cost drops from O(L^2) to O(N^2) with N = L/16, and each token carries local shape
information instead of a single noisy point.
"""

from __future__ import annotations

import torch


def num_patches(length: int, patch_len: int, stride: int) -> int:
    if length < patch_len or (length - patch_len) % stride:
        raise ValueError(f"length {length} incompatible with patch {patch_len}/stride {stride}")
    return (length - patch_len) // stride + 1


def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """(B, L) -> (B, N, P).

    Non-overlapping patches (the only kind the model uses) are a pure reshape: no copy, and it
    exports to ONNX/TensorRT with dynamic batch (``unfold`` does not, in the TorchScript exporter).
    """
    n = num_patches(x.shape[-1], patch_len, stride)
    if stride == patch_len:
        return x.reshape(*x.shape[:-1], n, patch_len)
    return x.unfold(-1, patch_len, stride)


def unpatchify(p: torch.Tensor, stride: int) -> torch.Tensor:
    """(B, N, P) -> (B, L) for non-overlapping patches (stride == P)."""
    b, n, plen = p.shape
    if stride != plen:
        raise ValueError("unpatchify only supports stride == patch_len")
    return p.reshape(b, n * plen)
