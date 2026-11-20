"""Streaming random-window loader over memmap shards.

Design goals and how they are met:

**No duplication across ranks/workers.** Every *consumer* (one per (DDP rank, DataLoader worker)
pair; ``G = world_size * num_workers`` in total) walks the same epoch-permuted shard order. If there
are at least ``G`` shards, consumer ``g`` owns whole shards ``g, g+G, ...`` (best locality). With
fewer shards (debug corpora, small val sets), ownership falls back to a running series counter
modulo ``G``. Either way each series is visited by exactly one consumer per epoch (tested).

**Random windows.** A series of length ``n`` contributes ``max(1, round(n / L))`` windows per epoch
at uniformly random offsets, so long series are sampled proportionally to their length without
ever materializing all windows. Series shorter than ``L`` are left-padded; the mask marks real data.

**Shuffling** happens at three levels: shard order (per epoch), series order within a shard, and a
reservoir-style shuffle buffer across series and shards.

**Determinism & resume.** All randomness derives from ``(seed, epoch, consumer)``, so a consumer's
stream is a pure function of those. Resuming after ``k`` global batches only needs ``k``: with
``W`` workers the DataLoader returns batches round-robin, so worker ``w`` produced
``ceil((k - w) / W)`` of them; it regenerates its stream and skips that many samples. Worker roles
are rotated by ``k mod W`` because a fresh DataLoader restarts round-robin at worker 0. Exact as
long as ``num_workers``, ``batch_size`` and world size are unchanged (checked by the checkpoint).


**Normalization** is *not* done here: the model applies RevIN per window (and needs the raw scale to
de-normalize forecasts), so the loader returns raw values + mask.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from sensorfm.data.shard import ShardReader, load_manifest

Sample = dict[str, torch.Tensor]


def _dist_rank_world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class ShardWindowDataset(IterableDataset[Sample]):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        context_length: int = 512,
        seed: int = 0,
        shuffle_buffer: int = 1024,
        infinite: bool = True,
        rank: int | None = None,
        world_size: int | None = None,
        batch_size: int = 1,
        resume_batches: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.manifest = load_manifest(self.root)
        self.shards: list[dict[str, Any]] = self.manifest["splits"][split]["shards"]
        if not self.shards:
            raise ValueError(f"no shards in split {split!r} of {self.root}")
        self.L = context_length
        self.seed = seed
        self.shuffle_buffer = max(1, shuffle_buffer)
        self.infinite = infinite
        self._rank, self._world = rank, world_size
        self.batch_size = batch_size
        self.resume_batches = resume_batches

    # -- consumer identity ---------------------------------------------------------------------

    def _consumer(self) -> tuple[int, int, int, int]:
        """(global consumer id, total consumers, worker id, num workers)."""
        r, w = _dist_rank_world()
        rank = self._rank if self._rank is not None else r
        world = self._world if self._world is not None else w
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        # A fresh DataLoader always starts round-robin at worker 0, but after k batches the next
        # batch belongs to worker k % nw. Rotating roles makes resumed worker 0 play that role.
        role = (wid + self.resume_batches) % nw
        return rank * nw + role, world * nw, role, nw

    # -- stream --------------------------------------------------------------------------------

    def _windows(self, g: int, total: int) -> Iterator[Sample]:
        """Unshuffled-buffer stream of windows for consumer ``g`` (before the shuffle buffer)."""
        split_dir = self.root / self.split
        readers: dict[int, ShardReader] = {}
        epoch = 0
        while True:
            rng = np.random.default_rng([self.seed, epoch, g])
            order = np.random.default_rng([self.seed, epoch]).permutation(len(self.shards))
            whole_shards = len(self.shards) >= total
            counter = 0
            produced = False
            for pos, sid in enumerate(order):
                if whole_shards and pos % total != g:
                    continue
                rd = readers.get(int(sid))
                if rd is None:
                    rd = readers[int(sid)] = ShardReader(split_dir, self.shards[int(sid)])
                series_order = np.random.default_rng([self.seed, epoch, int(sid)]).permutation(
                    len(rd)
                )
                for i in series_order:
                    if not whole_shards:
                        owner = counter % total
                        counter += 1
                        if owner != g:
                            continue
                    n = int(rd.lengths[i])
                    for _ in range(max(1, round(n / self.L))):
                        produced = True
                        yield self._make_window(rd, int(i), n, rng)
            if not self.infinite or not produced:
                return
            epoch += 1

    def _make_window(self, rd: ShardReader, i: int, n: int, rng: np.random.Generator) -> Sample:
        L = self.L
        x = np.zeros(L, dtype=np.float32)
        mask = np.zeros(L, dtype=bool)
        if n >= L:
            start = int(rng.integers(0, n - L + 1))
            x[:] = rd.window(i, start, L)
            mask[:] = True
        else:
            x[L - n :] = rd.series(i)
            mask[L - n :] = True
        return {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask)}

    def __iter__(self) -> Iterator[Sample]:
        g, total, wid, nw = self._consumer()
        rng = np.random.default_rng([self.seed, 10_000_019, g])
        skip = 0
        if self.resume_batches:
            my_batches = max(0, -(-(self.resume_batches - wid) // nw))
            skip = my_batches * self.batch_size
        buf: list[Sample] = []
        emitted = 0
        for s in self._windows(g, total):
            if len(buf) < self.shuffle_buffer:
                buf.append(s)
                continue
            j = int(rng.integers(len(buf)))
            out, buf[j] = buf[j], s
            emitted += 1
            if emitted > skip:
                yield out
        rng.shuffle(buf)
        for out in buf:
            emitted += 1
            if emitted > skip:
                yield out


def make_loader(
    dataset: ShardWindowDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = 2,
) -> torch.utils.data.DataLoader[Sample]:
    """DataLoader with settings that matter for throughput (Task 3.2 sweeps these).

    A private generator is passed so creating an iterator never draws from the *global* torch RNG:
    otherwise resuming (which restores the global RNG, then creates the iterator) would shift every
    later dropout mask and break exact resumption. All our randomness is seeded in the dataset.
    """
    dataset.batch_size = batch_size
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(dataset.seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
