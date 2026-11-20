"""Deterministic seeding.

Every script seeds python, numpy and torch from one integer so runs are reproducible and the seed
can be logged next to the config. Data generators take an explicit `np.random.Generator` instead of
touching global state, so they stay deterministic even when called from multiple workers.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy (legacy global), and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Independent numpy generator; preferred over global state in library code."""
    return np.random.default_rng(seed)
