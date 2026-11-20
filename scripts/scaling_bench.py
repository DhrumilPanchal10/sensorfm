"""Scaling study point (Task 3.3): one (model, strategy, world size) measurement.

    python scripts/scaling_bench.py --model m60m --strategy none                     # 1 GPU
    torchrun --nproc_per_node=2 scripts/scaling_bench.py --model m60m --strategy ddp  # 2 GPUs
    torchrun --nproc_per_node=2 scripts/scaling_bench.py --model m60m --strategy fsdp

Fixed per-GPU micro-batch (weak scaling). Measures steady-state global tokens/s (CUDA-synced,
after warm-up), per-GPU peak memory and MFU. Real shards via the rank-partitioned loader.
Writes ``results/scaling/<model>_<strategy>_<world>.json`` from rank 0; the report computes
scaling efficiency = tokens/s(n) / (n * tokens/s(1)).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import typer
import yaml

from sensorfm.data.loader import ShardWindowDataset, make_loader
from sensorfm.model.sensorfm import ModelConfig, SensorFM, non_embedding_params
from sensorfm.pretrain.objectives import ObjectiveConfig, pretrain_step
from sensorfm.pretrain.schedule import make_optimizer
from sensorfm.utils.dist import all_reduce_mean, cleanup, init_distributed, wrap_model
from sensorfm.utils.flops import mfu, peak_flops, train_flops_per_token
from sensorfm.utils.hardware import hardware_info
from sensorfm.utils.seed import seed_everything


def main(
    model: str = "m60m",
    strategy: str = "none",
    shards: Path = Path("data/shards/debug"),
    micro_batch: int = 32,
    precision: str = "fp16",
    steps: int = 60,
    warmup: int = 15,
    workers: int = 2,
    out_dir: Path = Path("results/scaling"),
) -> None:
    info = init_distributed()
    seed_everything(0)
    raw = yaml.safe_load(Path(f"configs/model/{model}.yaml").read_text())
    mcfg = ModelConfig(**raw, context_length=512)
    core = SensorFM(mcfg).to(info.device)
    m = wrap_model(core, strategy, info, precision)
    opt = make_optimizer(m, 3e-4, 0.1, (0.9, 0.95), fused=False)
    scaler = torch.amp.GradScaler(
        info.device.type, enabled=precision == "fp16" and strategy != "fsdp"
    )
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    ds = ShardWindowDataset(
        shards, "train", 512, seed=0, rank=info.rank, world_size=info.world_size
    )
    it = iter(make_loader(ds, micro_batch, workers, pin_memory=info.device.type == "cuda"))
    obj = ObjectiveConfig()
    if info.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(info.device)

    def step(i: int) -> None:
        b = next(it)
        x, obs = (
            b["x"].to(info.device, non_blocking=True),
            b["mask"].to(info.device, non_blocking=True),
        )
        g = torch.Generator().manual_seed(i * 97 + info.rank)
        ctx = (
            torch.autocast(info.device.type, dtype=dtype)
            if dtype
            else torch.autocast(info.device.type, enabled=False)
        )
        with ctx:
            loss = pretrain_step(core, x, obs, obj, g, m)["total"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    for i in range(warmup):
        step(i)
    if info.device.type == "cuda":
        torch.cuda.synchronize(info.device)
    t0 = time.perf_counter()
    for i in range(warmup, steps):
        step(i)
    if info.device.type == "cuda":
        torch.cuda.synchronize(info.device)
    dt = time.perf_counter() - t0
    tok_s_global = (steps - warmup) * micro_batch * mcfg.n_patches * info.world_size / dt
    mem = torch.cuda.max_memory_allocated(info.device) / 1e9 if info.device.type == "cuda" else 0.0
    mem_max = float(all_reduce_mean(torch.tensor(mem, device=info.device)))
    hw = hardware_info()
    peak = peak_flops(hw["gpus"][0], precision) if hw.get("gpus") else None
    f_tok = train_flops_per_token(
        non_embedding_params(core), mcfg.n_layers, mcfg.n_patches, mcfg.d_model
    )
    rec = {"model": model, "strategy": strategy, "world_size": info.world_size,
           "micro_batch_per_gpu": micro_batch, "precision": precision,
           "tokens_per_s": tok_s_global, "peak_mem_gb_per_gpu": mem_max,
           "mfu": mfu(tok_s_global, f_tok, peak, info.world_size),
           "params_non_embedding": non_embedding_params(core), "hardware": hw}  # fmt: skip
    if info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{model}_{strategy}_{info.world_size}.json").write_text(
            json.dumps(rec, indent=2)
        )
        print(json.dumps({k: v for k, v in rec.items() if k != "hardware"}), flush=True)
    cleanup()


if __name__ == "__main__":
    typer.run(main)
