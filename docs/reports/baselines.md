# Results (generated)

### Forecasting baselines (ETT, H=96, context 512)

*Pending — `results/eval/forecast_baselines.json` not produced yet.*

### C-MAPSS classic baselines (W=30, 3 seeds)

*Pending — `results/eval/cmapss_baselines.json` not produced yet.*

### Scaling (Kaggle T4)

*Pending — `results/scaling` not produced yet.*

### Crash-resume test (CPU)

- killed after step 25, resumed from 20 (5 steps lost); first new step 0.95 s after restart; all logged losses identical to the uninterrupted run.

<sub>Source: `results/resume_test.json`</sub>

### Distributed correctness (CPU, gloo, 2 processes)

- max |loss difference| vs single process: DDP 2.4e-07, FSDP2 2.4e-07 over 10 steps

<sub>Source: `results/scaling/parity_cpu_gloo.json`</sub>
