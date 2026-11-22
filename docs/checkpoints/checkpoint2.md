# 🛑 Human Checkpoint 2 — GPU run plan (free tier: Kaggle)

Prepared 2026-11-20. **Cost: $0** (Kaggle free GPU quota). The plan's "rent a 4-GPU node" is
replaced by Kaggle "GPU T4 x2" (ADR 000). Nothing below has been run. All estimates are
assumptions until the ladder (run 1) measures real throughput — `scripts/estimate_budget.py`
re-prints them from measured numbers before each later run.

Verify your remaining weekly GPU quota in Kaggle (Settings → Accelerator) before starting.

## Prerequisite (CPU session, no GPU quota)

0. `kaggle/build_data.sh` → private dataset `sensorfm-data` (shards + downstream). Needs
   Checkpoint 1 approval of the corpus first.

## Planned GPU runs, in order

| # | Script | Hardware | Est. session-hours | Produces |
|---|---|---|---|---|
| 1 | `02_ladder_scaling.ipynb` (ladder part) | 1× T4 | 0.5 | `results/profiles/ladder.json`, traces (Tasks 3.1/3.2) |
| 2 | `02_ladder_scaling.ipynb` (scaling part) | T4 ×2 | 0.6 | `results/scaling/*.json` (Task 3.3) |
| 3 | `03_pretrain.ipynb` (s20m) | T4 ×2 DDP | ≤ 8 | s20m checkpoint (Task 2.3 GPU acceptance, 3.5) |
| 4 | `04_downstream.ipynb` (s20m) | 1× T4 | ~4 | RUL / forecast / zero-shot results (Phase 4) |
| 5 | m60m pretrain | T4 ×2 DDP | ≤ 10 (2 sessions) | Task 3.5 size point |
| 6 | l150m pretrain | T4 ×2 DDP | budget-dependent | Task 3.5 size point (may be skipped) |
| 7 | Deploy: TensorRT FP16/INT8 + GPU latency | 1× T4 | 0.5 | `results/bench/*_trt.json` (Task 5.2) |

STEPS for run 3 is chosen after run 1: `python scripts/estimate_budget.py --model s20m --steps N`
with the measured ladder throughput, sized so the run fits the approved hours.

## Things that can only be verified on the GPU (watch for these)

- fp16 + FSDP2 gradient scaling (the scaling bench disables the GradScaler for FSDP; the training
  loop keeps it on — confirm it runs, or switch FSDP runs to fp32 reduce + no scaler).
- `torch.compile` on T4 (Turing) with SDPA: compile time and whether it helps (the ladder measures).
- TensorRT 10 API (`check_api()` fails loudly on mismatch).
- The `bool(key_valid.all())` host sync in the encoder: expected to show up in the profile / cause a
  graph break under compile; fix only if the trace shows it matters (rule 8).

## Decision needed

Approve runs 1–4 (≈ 13 session-hours ≈ half a typical weekly quota), and say whether to attempt
runs 5–6 the following week.
