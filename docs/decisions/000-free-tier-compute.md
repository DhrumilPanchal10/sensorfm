# ADR 000 — Free-tier compute only

**Status:** accepted (2026-11-20)

## Context

The owner requires that the project costs $0. `plan.md` assumes a rented ≥24 GB GPU and a rented
4-GPU node. Neither is allowed. The local dev machine is an 8 GB Apple Silicon Mac with very little
free disk (~6 GB), so it cannot hold the pretraining corpus or run a real training job.

## Decision

| Plan assumption | Free-tier replacement |
|---|---|
| Single ≥24 GB GPU for dev | Kaggle Notebook GPU (T4 16 GB or P100 16 GB). Weekly GPU quota applies — verify the current quota in Kaggle settings before each phase. |
| Rented 4-GPU node for scaling | Kaggle "GPU T4 x2" accelerator. Scaling study is **1 vs 2 GPUs**, not 1/2/4. |
| bf16 autocast | **T4 (sm_75) and P100 (sm_60) have no bf16 tensor cores.** Precision is a config switch: `fp16` + `GradScaler` on T4, `bf16` where supported. MFU uses the fp16 tensor-core peak for T4. |
| Corpus on local disk | Corpus is built and stored on Kaggle (`/kaggle/working`, then saved as a private Kaggle Dataset). Local machine only ever sees synthetic/debug data. |
| W&B | MLflow with a file store (`file:./mlruns`), copied out of Kaggle as an output artifact. W&B free tier is optional. |
| `GPU_PRICE_PER_HOUR` cost estimate | Still printed before every GPU job, but with price 0 the estimate is reported as **GPU-hours against the weekly Kaggle quota** instead. |
| Session length | Kaggle sessions are time-limited, so checkpoint/resume (Task 2.3) is not optional — every long run must be resumable across sessions. |
| TensorRT INT8 | Built on Kaggle T4 (TensorRT installable via pip). CPU INT8 via ONNX Runtime runs locally. |
| LLM usage | None is needed for this project. If one is ever needed (e.g. drafting text), use a free/open Qwen model; no paid API. |

## Consequences

- Scaling efficiency target is restated for 2 GPUs (see `docs/targets.md`).
- l150m may not fit the weekly quota at a useful token budget; Task 3.5 decides from measured
  throughput and records the choice.
- Free-tier quotas and hardware change; re-check them at each 🛑 checkpoint and update this ADR.
