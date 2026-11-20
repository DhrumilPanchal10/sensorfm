import numpy as np
import torch

import sensorfm
from sensorfm.utils.seed import make_rng, seed_everything


def test_version() -> None:
    assert sensorfm.__version__


def test_seed_everything_is_deterministic() -> None:
    seed_everything(7)
    a = (np.random.rand(3), torch.rand(3))
    seed_everything(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_make_rng_independent() -> None:
    assert make_rng(1).random() == make_rng(1).random()
