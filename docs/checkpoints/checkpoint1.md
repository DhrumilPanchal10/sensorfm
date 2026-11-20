# 🛑 Human Checkpoint 1 — corpus, decontamination, downstream protocols

Prepared 2026-11-20. Nothing has been run on Kaggle and nothing has cost money.

## What to review (≈15 min)

1. **Corpus selection** — [decisions/001-corpus.md](../decisions/001-corpus.md),
   `configs/data/lotsa_selection.yaml`. 46 LOTSA subsets across energy / industrial / telemetry /
   transport / climate, capped to ≈100–150M observations (~0.5 GB float32).
2. **Decontamination** — `src/sensorfm/data/dedup.py`, dataset card §1. Tested on real data:
   0 false matches on 1,012 real LOTSA series (after fixing a degenerate-window bug that produced
   16), 50/50 planted downstream copies detected.
3. **Downstream protocols** — [dataset_card.md](../dataset_card.md) §2. All four sources
   verified live, loaders checked on the real files.

## Decisions needed

| # | Question | Default if you just say "approved" |
|---|---|---|
| 1 | Domain inclusion/exclusion (telemetry in? CMIP6 out?) | As proposed |
| 2 | First corpus size ≈100–150M obs, grow after Task 3.1 | Yes |
| 3 | Per-subset licenses are **unverified**; drop any that can't be confirmed? | Drop unconfirmed before the real build |
| 4 | C-MAPSS test targets uncapped (report capped as secondary)? | Uncapped primary |
| 5 | CWRU: ball faults show no envelope signature (0/12). Keep 10 classes, or also report a 7-class variant without ball faults? | Keep 10, report caveat + 7-class secondary |
| 6 | Forecasting has only the ETT family (4 sets). Add a 5th non-LOTSA set? | Proceed with ETT×4; revisit in Phase 4 |
| 7 | Bearing split: train loads {0,1,2} → test load {3}, plus leave-one-load-out average? | Both |

## What you need to do on Kaggle (free, CPU session is enough for data)

See [kaggle/README.md](../../kaggle/README.md). In short: push the repo to a GitHub repo
(private is fine), open a Kaggle notebook with Internet ON, run `kaggle/build_data.sh`, and save
the output as a private Kaggle Dataset. CPU sessions don't use the weekly GPU quota.
