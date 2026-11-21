# Progress

- 2026-11-20 phase1.task1: scaffold, Makefile, CI, Hydra configs, free-tier ADR 000, targets.
- 2026-11-20 phase1.task2: synthetic physics generators (SDOF, n-DOF chain, rotating/bearing, degradation) with exact ZOH discretization; 16 physics tests.
- 2026-11-20 phase1.task3 (code): LOTSA verified (925 GB, arrow stream), selection config + ADR 001, cleaning, window-hash decontamination, memmap shards + manifest, debug shards. Real build pending Kaggle + Checkpoint 1.
- 2026-11-20 phase1.task4: streaming window loader (rank/worker partition, 3-level shuffle, exact resume incl. multi-worker), masked stats, loader bench (78k samples/s local, 1 proc).
- 2026-11-20 phase1.task5: downstream loaders verified on real files; UCR scoring rule from archive slides; CWRU label check (ball 0/12 signature); dedup vectorized + low-info-window fix (16 false hits -> 0, 50/50 planted caught); dataset card.
- 2026-11-20 phase2.task1-3 (CPU): model + objectives + train loop; SIGKILL-resume bit-exact (lost 5 steps, 0.94 s to first step); CPU debug val MSE 0.750->0.632 in 300 steps. s20m GPU run pending Kaggle.
- 2026-11-20 phase3.task3 (CPU part): DDP & FSDP2 match single-process loss to 1.2e-7; exact resume under both (FSDP2 via sharded DCP); ADR 002. GPU scaling study pending Kaggle T4x2 + Checkpoint 2.
- 2026-11-20 phase3.task1-2 (tooling only): bench ladder (7 rungs, parity vs seed noise, torch.profiler traces), act-ckpt, generated efficiency report, kaggle/run_ladder.sh + pretrain.sh + budget estimate. Measurements pending Kaggle.
- 2026-11-20 phase4-5 (code): baselines (naive/ETS/GBM/LSTM/CNN), fine-tune regimes (RUL/bearing/forecast/anomaly), ONNX export + ORT INT8 + TRT builder + latency bench, streaming service + drift. GPU/pretrained results pending Kaggle.
- 2026-11-20 04:2x: eval tooling complete (UCR runner, analyze_rul bootstrap, report sections); fixed baseline seeding bug; 100/100 tests. CPU baselines (ETT naive/ETS; C-MAPSS GBM/LSTM/CNN full + 10% few-shot) running.
- 2026-11-20 09:06: C-MAPSS baseline rerun (post target-scaling fix) stalled after 3/36 runs (machine likely slept); not committed. Rerun: make sure Mac stays awake (caffeinate), then the two eval_cmapss_baselines.py commands in PROGRESS notes.
- 2026-11-20 10:45: C-MAPSS baselines (fixed target scaling) done: 36 full + 36 few-shot runs, 0 collapsed; paired bootstrap: GBM/LSTM/CNN indistinguishable except CNN worse on FD004. Bootstrap vectorized (10 min -> 0.3 s).
- 2026-11-20: Checkpoint 1 APPROVED (defaults). Real LOTSA pipeline dry-run on 5 subsets OK (10.2M obs, 96 s; train on real shards OK). Repo listing cached. Kaggle data-build notebook added.
