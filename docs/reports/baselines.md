# Results (generated)

### Model sizes

| Config | Parameters | Non-embedding | Train FLOPs / token |
|---|---|---|---|
| tiny | 114,722 | 112,674 | 7.01e+05 |
| s20m | 14,283,426 | 14,271,138 | 8.62e+07 |
| m60m | 50,555,042 | 50,538,658 | 3.05e+08 |
| l150m | 127,756,962 | 127,732,386 | 7.69e+08 |

<sub>Source: `results/model/params.json`</sub>

### Forecasting (ETT, H=96)

*Pending — `results/eval/forecast_baselines.json` not produced yet.*

### C-MAPSS classic baselines (W=30, 3 seeds)

*Pending — `results/eval/cmapss_baselines.json` not produced yet.*

### C-MAPSS classic baselines, few-shot (10% of training units)

*Pending — `results/eval/cmapss_baselines_fewshot10.json` not produced yet.*

### Scaling (Kaggle T4)

*Pending — `results/scaling` not produced yet.*

### Crash-resume test (CPU)

- killed after step 25, resumed from 20 (5 steps lost); first new step 1.40 s after restart; all logged losses identical to the uninterrupted run.

<sub>Source: `results/resume_test.json`</sub>

### Distributed correctness (CPU, gloo, 2 processes)

- max |loss difference| vs single process: DDP 2.4e-07, FSDP2 2.4e-07 over 10 steps

<sub>Source: `results/scaling/parity_cpu_gloo.json`</sub>
