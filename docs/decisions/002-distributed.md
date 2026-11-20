# ADR 002 — Distributed training: DDP and FSDP2, checkpoints, correctness evidence

**Status:** accepted for CPU correctness (2026-11-20); GPU scaling numbers pending Kaggle T4 ×2.

## What each strategy does (the 60-second explanation)

- **DDP:** every GPU holds the full model, gradients and AdamW state (≈ 16 bytes/param in fp32:
  4 weights + 4 grads + 8 Adam moments). Each rank runs forward/backward on its own slice of the
  global batch; gradients are all-reduced in buckets *during* backward so communication overlaps
  compute. Afterwards every rank applies the same update, so replicas stay identical.
- **FSDP2** (`torch.distributed.fsdp.fully_shard`, per-parameter DTensor sharding): each rank
  stores only `1/N` of every parameter, gradient and optimizer state. Before a block's forward,
  its parameters are **all-gathered**; after its backward, gradients are **reduce-scattered** so
  each rank keeps only its shard's gradient, and the optimizer updates only that shard. Each
  transformer block is its own unit, so at most ~one block is fully materialized at a time.
  Memory for model state drops ~N×; communication volume rises (≈1.5× DDP: all-gather in forward,
  again in backward, plus reduce-scatter).
- At our sizes (≤128M params → ≤2 GB of fp32 model+Adam state) everything fits on a 16 GB T4, so
  **DDP is expected to be sufficient and faster**; FSDP's value here is demonstrating memory
  scaling, which the scaling study must measure rather than assume.

## Implementation notes

- Gradient accumulation skips sync on non-final micro-steps (`no_sync` / `set_requires_gradient_sync`).
- FSDP2 mixed precision: params/compute in fp16 (T4) or bf16, gradient reduction in fp32.
- **Explicit device mesh** is passed to `fully_shard`. Without it, torch auto-detects the
  accelerator; on macOS it picks MPS and crashes (`torch.mps.is_initialized` missing) even in a
  CPU/gloo run — found while building the CPU test.
- **Checkpoints** are directories, written to `.tmp` and atomically renamed after a barrier.
  FSDP2 uses sharded `torch.distributed.checkpoint` (each rank writes its own shards);
  `torch.save` of an FSDP2 state dict would silently keep only rank 0's shards. Every rank saves
  its own RNG state (dropout streams differ per rank).
- The wall-clock stop decision is all-reduced so all ranks stop at the same step, and it is only
  evaluated on logging steps because the reduction forces a device sync.

## Evidence (CPU, gloo, 2 processes; `tests/integration/test_dist.py`)

- Loss per step for DDP and FSDP2 vs single process on the same global batch: max |Δ| = 1.2e-7
  over 10 steps (`results/scaling/parity_cpu_gloo.json`).
- Stop at step 20 → resume → steps 21–30 match an uninterrupted run (rel 1e-6), for both DDP and
  FSDP2 (sharded DCP checkpoint).
- macOS gloo needs `GLOO_SOCKET_IFNAME=lo0` and `--master_addr=127.0.0.1`, otherwise rendezvous
  hangs.
