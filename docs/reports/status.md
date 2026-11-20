# Results status (generated)

| Item | Source | Status |
|---|---|---|
| Model sizes | `results/model/params.json` | done |
| Forecasting — baselines (ETT, H=96, context 512) | `results/eval/forecast_baselines.json` | done |
| C-MAPSS classic baselines (W=30, 3 seeds) | `results/eval/cmapss_baselines.json` | done |
| C-MAPSS classic baselines, few-shot (10% of training units) | `results/eval/cmapss_baselines_fewshot10.json` | done |
| RUL paired comparisons (bootstrap over engines, seed-averaged predictions) | `results/eval/rul_analysis.json` | done |
| Scaling (Kaggle T4) | `results/scaling` | pending |
| Crash-resume test (CPU) | `results/resume_test.json` | done |
| Distributed correctness (CPU, gloo, 2 processes) | `results/scaling/parity_cpu_gloo.json` | done |
| Training efficiency ladder | `results/profiles/ladder.json` | pending |
