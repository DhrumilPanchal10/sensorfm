"""Task 1.4: loader correctness — coverage without duplication, determinism, resume."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from sensorfm.data.loader import ShardWindowDataset, make_loader
from sensorfm.data.normalize import masked_mean_std
from sensorfm.data.shard import ShardRecord, write_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Series whose values encode their id: value = id * 10_000 + position (exact in float32
    up to ~16.7M, so ids < 1000 and lengths < 10_000 are safe)."""
    root = tmp_path_factory.mktemp("shards")
    rng = np.random.default_rng(0)
    recs = []
    for i in range(240):
        n = int(rng.integers(40, 1500))
        vals = (i * 10_000 + np.arange(n)).astype(np.float32)
        recs.append(ShardRecord(f"d/{i}", "d", "x", "H", vals))
    write_corpus(recs, root, shard_bytes=40_000, val_frac=0.0, seed=0)
    return root


def _ids(samples: list[dict[str, torch.Tensor]]) -> list[int]:
    out = []
    for s in samples:
        v = s["x"][s["mask"]]
        out.append(int(v[0].item()) // 10_000)
    return out


def _epoch(root: Path, **kw: object) -> list[dict[str, torch.Tensor]]:
    ds = ShardWindowDataset(root, context_length=128, infinite=False, shuffle_buffer=16, **kw)  # type: ignore[arg-type]
    return list(ds)


def test_windows_are_contiguous_and_mask_marks_padding(corpus: Path) -> None:
    for s in _epoch(corpus):
        x, m = s["x"], s["mask"]
        assert x.shape == (128,) and m.dtype == torch.bool
        v = x[m]
        assert torch.all(torch.diff(v) == 1)  # contiguous slice of one series
        if not m.all():  # left padding only
            assert not m[0] and m[-1]
            assert torch.all(x[~m] == 0)


@pytest.mark.parametrize("world", [1, 2, 3, 5])
def test_every_series_once_per_epoch_across_ranks(corpus: Path, world: int) -> None:
    single = Counter(_ids(_epoch(corpus)))
    combined: Counter[int] = Counter()
    for r in range(world):
        combined.update(_ids(_epoch(corpus, rank=r, world_size=world)))
    assert set(combined) == set(range(240))
    assert combined == single  # same windows-per-series, nothing duplicated or dropped


def test_many_consumers_more_than_shards_still_partition(corpus: Path) -> None:
    n_shards = len(ShardWindowDataset(corpus).shards)
    world = n_shards + 3  # forces series-level ownership
    per_rank = [set(_ids(_epoch(corpus, rank=r, world_size=world))) for r in range(world)]
    for a in range(world):
        for b in range(a + 1, world):
            assert not per_rank[a] & per_rank[b]
    assert set().union(*per_rank) == set(range(240))


def test_workers_partition_with_dataloader(corpus: Path) -> None:
    ds = ShardWindowDataset(corpus, context_length=128, infinite=False, shuffle_buffer=8)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2)
    assert Counter(_ids(list(dl))) == Counter(_ids(_epoch(corpus)))


def test_deterministic_given_seed(corpus: Path) -> None:
    a = _epoch(corpus, seed=3)
    b = _epoch(corpus, seed=3)
    c = _epoch(corpus, seed=4)
    assert all(torch.equal(x["x"], y["x"]) for x, y in zip(a, b, strict=True))
    assert not all(torch.equal(x["x"], y["x"]) for x, y in zip(a, c, strict=False))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resume_reproduces_remaining_stream(corpus: Path, num_workers: int) -> None:
    bs, k = 4, 7

    def batches(resume: int) -> list[torch.Tensor]:
        ds = ShardWindowDataset(corpus, context_length=128, seed=1, shuffle_buffer=32,
                                resume_batches=resume)  # fmt: skip
        dl = make_loader(ds, batch_size=bs, num_workers=num_workers)
        out = []
        for i, b in enumerate(dl):
            out.append(b["x"])
            if i + 1 >= 12:
                break
        return out

    full = batches(0)
    resumed = batches(k)
    for a, b in zip(full[k:], resumed, strict=False):
        assert torch.equal(a, b)


def test_infinite_stream_rolls_over_epochs(corpus: Path) -> None:
    ds = ShardWindowDataset(corpus, context_length=128, infinite=True, shuffle_buffer=4)
    n_epoch = len(_epoch(corpus))
    it = iter(ds)
    for _ in range(n_epoch * 2 + 5):
        next(it)


def test_masked_mean_std() -> None:
    x = torch.tensor([[0.0, 0.0, 1.0, 3.0], [5.0, 5.0, 5.0, 5.0]])
    m = torch.tensor([[False, False, True, True], [False, False, False, False]])
    mean, std = masked_mean_std(x, m, eps=0.0)
    assert mean[0, 0] == 2.0 and std[0, 0] == 1.0
    assert mean[1, 0] == 0.0 and std[1, 0] == 1.0
