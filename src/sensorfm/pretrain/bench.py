"""Throughput benchmark + optimization ladder (Tasks 3.1 / 3.2).

    python -m sensorfm.pretrain.bench --model s20m --shards $SHARD_DIR \
        --ladder default --steps 300 --out results/profiles/ladder.json [--profile]

Each *rung* is one configuration measured in a fresh process state (new model, same seed):

    0 naive        fp32, eager, num_workers=0, no pinned memory      (reference)
    1 loader       + num_workers, pin_memory, prefetch
    2 amp          + autocast fp16 (T4/P100) or bf16 (Ampere+) with GradScaler for fp16
    3 compile      + torch.compile (compile time recorded separately)
    4 fused_adamw  + AdamW(fused=True)
    5 big_batch    + 2x micro-batch (uses the memory saved by AMP)
    6 act_ckpt     + activation checkpointing (only kept if memory-bound)

Measured per rung: steady-state tokens/s and samples/s (after warm-up steps, CUDA-synchronized),
data-wait fraction, peak memory, MFU, compile time, and the mean training loss over the last
quarter of steps on a fixed seed — the parity check that optimizations did not change learning
(compared to rung 0 against the seed-to-seed noise measured by ``--noise-seeds``).

``--profile`` additionally records a torch.profiler trace (Chrome JSON) and the key-averages
table of a few steps into ``results/profiles/<rung>/``.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import typer
import yaml

from sensorfm.data.loader import ShardWindowDataset, make_loader
from sensorfm.model.sensorfm import ModelConfig, SensorFM, non_embedding_params
from sensorfm.pretrain.objectives import ObjectiveConfig, pretrain_step
from sensorfm.pretrain.schedule import make_optimizer
from sensorfm.utils.flops import mfu, peak_flops, train_flops_per_token
from sensorfm.utils.hardware import hardware_info
from sensorfm.utils.seed import seed_everything

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class Rung:
    name: str
    precision: str = "fp32"
    compile: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    fused_adamw: bool = False
    micro_batch: int = 64
    grad_checkpoint: bool = False


def default_ladder(micro_batch: int, amp: str, workers: int) -> list[Rung]:
    r0 = Rung("0_naive", micro_batch=micro_batch)
    r1 = replace(r0, name="1_loader", num_workers=workers, pin_memory=True)
    r2 = replace(r1, name="2_amp", precision=amp)
    r3 = replace(r2, name="3_compile", compile=True)
    r4 = replace(r3, name="4_fused_adamw", fused_adamw=True)
    r5 = replace(r4, name="5_big_batch", micro_batch=2 * micro_batch)
    r6 = replace(r5, name="6_act_ckpt", grad_checkpoint=True)
    return [r0, r1, r2, r3, r4, r5, r6]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_rung(
    rung: Rung,
    mcfg: ModelConfig,
    shards: Path,
    steps: int,
    warmup: int,
    seed: int,
    device: torch.device,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    mcfg = replace(mcfg, grad_checkpoint=rung.grad_checkpoint)
    core = SensorFM(mcfg).to(device)
    fwd: torch.nn.Module = core
    opt = make_optimizer(core, 3e-4, 0.1, (0.9, 0.95), rung.fused_adamw and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=rung.precision == "fp16")
    obj = ObjectiveConfig()
    ds = ShardWindowDataset(shards, "train", mcfg.context_length, seed=seed, shuffle_buffer=1024)
    it = iter(make_loader(ds, rung.micro_batch, rung.num_workers, pin_memory=rung.pin_memory))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(rung.precision)
    amp = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype is not None
        else contextlib.nullcontext()
    )

    compile_s = 0.0
    if rung.compile:
        fwd = torch.compile(core)  # type: ignore[assignment]

    def step_fn(i: int, batch: dict[str, torch.Tensor] | None = None) -> float:
        if batch is None:
            batch = next(it)
        x = batch["x"].to(device, non_blocking=True)
        obs = batch["mask"].to(device, non_blocking=True)
        g = torch.Generator().manual_seed(seed * 100_003 + i)
        with amp:
            losses = pretrain_step(core, x, obs, obj, g, fwd)
        scaler.scale(losses["total"]).backward()  # type: ignore[no-untyped-call]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return float(losses["total"].detach()) if i >= steps - steps // 4 else float("nan")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    step_fn(0)  # first step includes compilation for compiled rungs
    _sync(device)
    first_step_s = time.perf_counter() - t0
    if rung.compile:
        compile_s = first_step_s
    for i in range(1, warmup):
        step_fn(i)
    _sync(device)

    data_wait = 0.0
    losses: list[float] = []
    t0 = time.perf_counter()
    for i in range(warmup, steps):
        td = time.perf_counter()
        batch = next(it)  # time blocked waiting on the loader (CPU-side; GPU runs async)
        data_wait += time.perf_counter() - td
        losses.append(step_fn(i, batch))
    _sync(device)
    elapsed = time.perf_counter() - t0
    measured = steps - warmup
    samples_s = measured * rung.micro_batch / elapsed
    tok_s = samples_s * mcfg.n_patches

    flops_tok = train_flops_per_token(
        non_embedding_params(core), mcfg.n_layers, mcfg.n_patches, mcfg.d_model
    )
    hw_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    peak = peak_flops(hw_name, rung.precision) if device.type == "cuda" else None
    tail = [v for v in losses if v == v]
    result: dict[str, Any] = {
        "rung": asdict(rung),
        "tokens_per_s": tok_s,
        "samples_per_s": samples_s,
        "step_ms": 1000 * elapsed / measured,
        "data_wait_frac": data_wait / elapsed,
        "mfu": mfu(tok_s, flops_tok, peak),
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else None,
        "compile_s": compile_s,
        "final_loss_mean": sum(tail) / max(1, len(tail)),
        "steps": steps,
        "warmup": warmup,
    }

    if profile_dir is not None:
        from torch.profiler import ProfilerActivity, profile, schedule

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
        out = profile_dir / rung.name
        out.mkdir(parents=True, exist_ok=True)
        with profile(activities=acts, schedule=schedule(wait=1, warmup=2, active=5),
                     record_shapes=True, profile_memory=True) as prof:  # fmt: skip
            for i in range(8):
                step_fn(steps + i)
                prof.step()
        prof.export_chrome_trace(str(out / "trace.json"))
        sort = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        (out / "key_averages.txt").write_text(prof.key_averages().table(sort_by=sort, row_limit=40))
        result["profile_dir"] = str(out)
    return result


@app.command()
def main(
    model: str = "tiny",
    shards: Path = Path("data/shards/debug"),
    ladder: str = "default",
    rungs: str = "",
    steps: int = 200,
    warmup: int = 20,
    micro_batch: int = 64,
    workers: int = 2,
    amp: str = "",
    seed: int = 0,
    noise_seeds: int = 0,
    profile: bool = False,
    out: Path = Path("results/profiles/ladder.json"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = yaml.safe_load(Path(f"configs/model/{model}.yaml").read_text())
    mcfg = ModelConfig(**raw, context_length=512)
    if not amp:
        bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        amp = "bf16" if bf16_ok else ("fp16" if device.type == "cuda" else "bf16")
    all_rungs = default_ladder(micro_batch, amp, workers) if ladder == "default" else []
    if rungs:
        keep = set(rungs.split(","))
        all_rungs = [r for r in all_rungs if r.name.split("_", 1)[0] in keep or r.name in keep]
    results = []
    for r in all_rungs:
        print(f"== {r.name}", flush=True)
        try:
            res = run_rung(r, mcfg, shards, steps, warmup, seed, device,
                           out.parent if profile else None)  # fmt: skip
        except torch.cuda.OutOfMemoryError as e:  # a rung may not fit; record, don't crash
            res = {"rung": asdict(r), "error": f"OOM: {e}"[:300]}
            torch.cuda.empty_cache()
        print(json.dumps({k: v for k, v in res.items() if k != "rung"}), flush=True)
        results.append(res)
    noise = []
    for s in range(1, noise_seeds + 1):  # reference rung, other seeds -> loss noise floor
        noise.append(
            run_rung(all_rungs[0], mcfg, shards, steps, warmup, seed + s, device)["final_loss_mean"]
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model,
        "params_non_embedding": non_embedding_params(SensorFM(mcfg)),
        "amp": amp,
        "results": results,
        "reference_loss_other_seeds": noise,
        "hardware": hardware_info(),
    }
    out.write_text(json.dumps(record, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
