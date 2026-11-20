# Progress

- 2026-11-20 phase1.task1: scaffold, Makefile, CI, Hydra configs, free-tier ADR 000, targets.
- 2026-11-20 phase1.task2: synthetic physics generators (SDOF, n-DOF chain, rotating/bearing, degradation) with exact ZOH discretization; 16 physics tests.
- 2026-11-20 phase1.task3 (code): LOTSA verified (925 GB, arrow stream), selection config + ADR 001, cleaning, window-hash decontamination, memmap shards + manifest, debug shards. Real build pending Kaggle + Checkpoint 1.
- 2026-11-20 phase1.task4: streaming window loader (rank/worker partition, 3-level shuffle, exact resume incl. multi-worker), masked stats, loader bench (78k samples/s local, 1 proc).
- 2026-11-20 phase1.task5: downstream loaders verified on real files; UCR scoring rule from archive slides; CWRU label check (ball 0/12 signature); dedup vectorized + low-info-window fix (16 false hits -> 0, 50/50 planted caught); dataset card.
- 2026-11-20 phase2.task1-3 (CPU): model + objectives + train loop; SIGKILL-resume bit-exact (lost 5 steps, 0.94 s to first step); CPU debug val MSE 0.750->0.632 in 300 steps. s20m GPU run pending Kaggle.
- 2026-11-20 phase3.task3 (CPU part): DDP & FSDP2 match single-process loss to 1.2e-7; exact resume under both (FSDP2 via sharded DCP); ADR 002. GPU scaling study pending Kaggle T4x2 + Checkpoint 2.
