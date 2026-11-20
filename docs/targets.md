# Targets (NOT results)

These are goals set before any experiment. Measured numbers live only in `results/` and generated
reports.

| Metric | Target |
|---|---|
| Throughput vs naive fp32 eager | ≥ 3× (MFU reported) |
| Scaling efficiency at 2 GPUs (Kaggle T4 x2; plan originally said 4) | ≥ 85% |
| Downstream vs from-scratch PatchTST | Better on RUL and anomaly tasks, especially few-shot |
| vs Chronos-2 zero-shot | Competitive MASE at a fraction of its parameters |
| INT8 deployment | ≥ 3× lower latency, ≤ 2% metric degradation |
