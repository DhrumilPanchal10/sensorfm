# Progress

- 2026-11-20 phase1.task1: scaffold, Makefile, CI, Hydra configs, free-tier ADR 000, targets.
- 2026-11-20 phase1.task2: synthetic physics generators (SDOF, n-DOF chain, rotating/bearing, degradation) with exact ZOH discretization; 16 physics tests.
- 2026-11-20 phase1.task3 (code): LOTSA verified (925 GB, arrow stream), selection config + ADR 001, cleaning, window-hash decontamination, memmap shards + manifest, debug shards. Real build pending Kaggle + Checkpoint 1.
- 2026-11-20 phase1.task4: streaming window loader (rank/worker partition, 3-level shuffle, exact resume incl. multi-worker), masked stats, loader bench (78k samples/s local, 1 proc).
- 2026-11-20 phase1.task5: downstream loaders verified on real files; UCR scoring rule from archive slides; CWRU label check (ball 0/12 signature); dedup vectorized + low-info-window fix (16 false hits -> 0, 50/50 planted caught); dataset card.
