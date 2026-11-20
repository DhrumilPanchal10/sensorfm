# Claim → evidence map

Each claim may only be made once its evidence file exists. Status is maintained by
`scripts/build_report.py` in [reports/status.md](reports/status.md); this table says *where* to look.

| Claim | Evidence | Available now? |
|---|---|---|
| Pretraining scale (params, observations) | `data/shards/main/manifest.json` (Kaggle build), `results/model/params.json`, run `params.json` | params yes; corpus after Checkpoint 1 |
| Throughput gain and MFU | `results/profiles/ladder.json`, traces, `docs/reports/training_efficiency.md` | after Kaggle run 1 |
| Scaling efficiency (DDP / FSDP2) | `results/scaling/*.json` | CPU correctness yes; GPU scaling after run 2 |
| Crash-tested resume | `tests/integration/test_resume.py`, `tests/integration/test_dist.py`, `results/resume_test.json` | yes (CPU) |
| Downstream comparisons and ablations | `results/eval/*.json`, `docs/reports/` | baselines in progress; SensorFM after Kaggle |
| INT8 latency / quality | `results/bench/*.json`, `docs/reports/deployment.md` | export + INT8 machinery tested; numbers after fine-tuned models exist |
| Streaming service with drift detection | `tests/unit/test_serving.py` (end-to-end replay flags injected anomaly; PSI fires on regime shift) | yes |

Suggested wording only after the evidence exists — e.g. "pretrained a 14M-parameter sensor
foundation model on N observations; X× training throughput over a naive fp32 baseline (MFU Y%) …"
with N, X, Y read from the files above.
