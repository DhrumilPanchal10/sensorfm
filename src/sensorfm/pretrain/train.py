"""Pretraining loop (single process, DDP or FSDP2 via torchrun).

    python -m sensorfm.pretrain.train                                  # CPU debug (tiny, synthetic)
    python -m sensorfm.pretrain.train model=s20m data=pretrain_main train=single_gpu
    torchrun --nproc_per_node=2 -m sensorfm.pretrain.train train=ddp  # Kaggle T4 x2
    python -m sensorfm.pretrain.train run_id=<id> resume=true          # continue a run

Per optimizer step: ``grad_accum`` micro-batches, each under autocast (fp16 needs a GradScaler;
bf16/fp32 don't); gradients are synchronized only on the last micro-batch; then unscale -> clip
(1.0) -> step -> LR schedule. Every ``log_every`` steps it logs loss terms, lr, grad norm,
tokens/s, samples/s, data-wait fraction, peak memory and MFU. Pretraining masks are drawn from a
generator seeded by (seed, rank, step, micro-step) and the loader resumes by batch count, so an
interrupted-and-resumed run reproduces the uninterrupted one exactly (tested on CPU).
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from sensorfm.data.loader import ShardWindowDataset, make_loader
from sensorfm.data.shard import load_manifest
from sensorfm.model.sensorfm import ModelConfig, SensorFM, count_params, non_embedding_params
from sensorfm.pretrain import checkpoint as ckpt
from sensorfm.pretrain.objectives import ObjectiveConfig, pretrain_step
from sensorfm.pretrain.schedule import lr_lambda, make_optimizer
from sensorfm.utils.dist import (
    DistInfo,
    all_reduce_mean,
    cleanup,
    init_distributed,
    maybe_no_sync,
    wrap_model,
)
from sensorfm.utils.flops import mfu, peak_flops, train_flops_per_token
from sensorfm.utils.hardware import hardware_info
from sensorfm.utils.io import RunLogger
from sensorfm.utils.seed import seed_everything
from sensorfm.utils.timing import StepTimer

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _mask_generator(seed: int, rank: int, step: int, micro: int) -> torch.Generator:
    return torch.Generator().manual_seed(
        ((seed * 1_000_003 + rank) * 1_000_003 + step) * 64 + micro
    )


def _autocast(device: torch.device, precision: str) -> contextlib.AbstractContextManager[Any]:
    if precision == "fp32":
        return contextlib.nullcontext()
    if device.type == "cpu" and precision == "fp16":
        raise ValueError("fp16 autocast is not supported on CPU; use fp32 or bf16")
    return torch.autocast(device_type=device.type, dtype=DTYPES[precision])


@torch.no_grad()
def evaluate(
    core: SensorFM, fwd: torch.nn.Module, cfg: DictConfig, obj: ObjectiveConfig, info: DistInfo
) -> dict[str, float]:
    """Held-out (by series) masked-reconstruction MSE and forecast CRPS (2 x mean pinball)."""
    fwd.eval()
    ds = ShardWindowDataset(
        cfg.data.shard_dir,
        "val",
        cfg.data.context_length,
        seed=12345,
        shuffle_buffer=1,
        infinite=False,
    )
    dl = make_loader(ds, cfg.train.micro_batch, num_workers=0)
    tot = {"masked": 0.0, "forecast": 0.0}
    n = 0
    for i, batch in enumerate(dl):
        if i >= cfg.train.val_batches:
            break
        x, obs = batch["x"].to(info.device), batch["mask"].to(info.device)
        with _autocast(info.device, cfg.train.precision):
            losses = pretrain_step(core, x, obs, obj, _mask_generator(999, 0, i, 0), fwd)
        for k in tot:
            tot[k] += float(losses[k])
        n += 1
    fwd.train()
    if n == 0:
        return {}
    out = {"val/masked_mse": tot["masked"] / n, "val/forecast_crps": 2 * tot["forecast"] / n}
    return {k: float(all_reduce_mean(torch.tensor(v, device=info.device))) for k, v in out.items()}


def run(cfg: DictConfig) -> dict[str, Any]:
    t_start = time.time()
    info = init_distributed(prefer_cuda=cfg.train.device != "cpu")
    seed_everything(cfg.seed)  # identical init on every rank
    out_dir = Path(cfg.output_dir)
    tc = cfg.train

    mcfg = ModelConfig(**OmegaConf.to_container(cfg.model), context_length=cfg.data.context_length)  # type: ignore[arg-type]
    core = SensorFM(mcfg).to(info.device)
    obj = ObjectiveConfig(**OmegaConf.to_container(cfg.objective))  # type: ignore[arg-type]
    model = wrap_model(core, tc.strategy, info, tc.precision)
    fwd: torch.nn.Module = torch.compile(model) if tc.compile else model  # type: ignore[assignment]
    torch.manual_seed(cfg.seed * 1000 + info.rank + 1)  # per-rank dropout streams

    opt = make_optimizer(model, tc.lr, tc.weight_decay, tuple(tc.betas), tc.fused_adamw)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(tc.warmup_steps, tc.max_steps, tc.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler(info.device.type, enabled=tc.precision == "fp16")

    fingerprint = {
        "world_size": info.world_size,
        "micro_batch": tc.micro_batch,
        "grad_accum": tc.grad_accum,
        "num_workers": tc.num_workers,
        "seed": cfg.seed,
        "context_length": cfg.data.context_length,
    }
    step = 0
    resumed_from = None
    if cfg.resume:
        last = ckpt.latest(out_dir / "checkpoints")
        if last is None:
            raise FileNotFoundError(f"resume=true but no checkpoint in {out_dir / 'checkpoints'}")
        meta = torch.load(last / "meta.pt", map_location="cpu", weights_only=False)
        ckpt.check_fingerprint(meta["fingerprint"], fingerprint)
        ckpt.load(last, model, opt, info, sharded=tc.strategy == "fsdp" and info.distributed)
        sched.load_state_dict(meta["scheduler"])
        scaler.load_state_dict(meta["scaler"])
        step = int(meta["step"])
        resumed_from = str(last)

    manifest = load_manifest(Path(cfg.data.shard_dir))
    n_tok = mcfg.n_patches
    flops_tok = train_flops_per_token(
        non_embedding_params(core), mcfg.n_layers, n_tok, mcfg.d_model
    )
    hw = hardware_info()
    peak = peak_flops(hw["gpus"][0], tc.precision) if hw.get("gpus") else None
    logger = RunLogger(
        out_dir,
        {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "params": count_params(core),
            "non_embedding_params": non_embedding_params(core),
            "train_flops_per_token": flops_tok,
            "data_manifest_sha256": manifest["manifest_sha256"],
            "hardware": hw,
            "resumed_from": resumed_from,
            "fingerprint": fingerprint,
        },
        use_mlflow=cfg.use_mlflow,
        enabled=info.is_main,
    )

    ds = ShardWindowDataset(
        cfg.data.shard_dir,
        "train",
        cfg.data.context_length,
        seed=cfg.seed,
        shuffle_buffer=cfg.data.shuffle_buffer,
        infinite=True,
        rank=info.rank,
        world_size=info.world_size,
        resume_batches=step * tc.grad_accum,
    )
    it = iter(make_loader(ds, tc.micro_batch, tc.num_workers, pin_memory=tc.pin_memory))

    timer = StepTimer(info.device)
    acc: dict[str, torch.Tensor] = {}
    acc_n = 0
    t_log = time.perf_counter()
    first_step_time: float | None = None
    stop_reason = "max_steps"
    if info.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(info.device)
    model.train()
    while step < tc.max_steps:
        for micro in range(tc.grad_accum):
            timer.mark_data_start()
            batch = next(it)
            timer.mark_data_end()
            x = batch["x"].to(info.device, non_blocking=True)
            obs = batch["mask"].to(info.device, non_blocking=True)
            gen = _mask_generator(cfg.seed, info.rank, step, micro)
            with (
                maybe_no_sync(model, micro == tc.grad_accum - 1),
                _autocast(info.device, tc.precision),
            ):
                losses = pretrain_step(core, x, obs, obj, gen, fwd)
                scaler.scale(losses["total"] / tc.grad_accum).backward()
            for k, v in losses.items():
                acc[k] = acc.get(k, torch.zeros((), device=info.device)) + v.detach().float()
            acc_n += 1
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        log_now = step % tc.log_every == 0 or step == tc.max_steps
        timer.mark_step_end(sync=log_now)
        if first_step_time is None:
            first_step_time = time.time() - t_start

        if log_now:
            dt = time.perf_counter() - t_log
            steps_in_window = tc.log_every if step % tc.log_every == 0 else step % tc.log_every
            samples = steps_in_window * tc.grad_accum * tc.micro_batch * info.world_size
            tok_s = samples * n_tok / dt
            m = {f"loss/{k}": float(all_reduce_mean(v / acc_n)) for k, v in acc.items()}
            m.update(
                {
                    "lr": float(sched.get_last_lr()[0]),
                    "grad_norm": float(grad_norm),
                    "samples_per_s": samples / dt,
                    "tokens_per_s": tok_s,
                    **timer.pop(),
                }
            )
            if scaler.is_enabled():
                m["amp/scale"] = float(scaler.get_scale())
            if info.device.type == "cuda":
                m["mem/peak_gb"] = torch.cuda.max_memory_allocated(info.device) / 1e9
            u = mfu(tok_s, flops_tok, peak, info.world_size)
            if u is not None:
                m["mfu"] = u
            if tc.val_every and (step % tc.val_every == 0 or step == tc.max_steps):
                m.update(evaluate(core, fwd, cfg, obj, info))
            logger.log(step, m)
            if info.is_main:
                print(
                    json.dumps({"step": step, **{k: round(v, 5) for k, v in m.items()}}), flush=True
                )
            acc, acc_n = {}, 0
            t_log = time.perf_counter()

        over_time = False
        if cfg.max_wall_minutes and log_now:  # checked on log steps only: it forces a sync
            local = (time.time() - t_start) > 60 * cfg.max_wall_minutes
            # Same decision on every rank (clocks differ slightly), otherwise ranks deadlock.
            over_time = bool(all_reduce_mean(torch.tensor(float(local), device=info.device)) > 0)
        early_stop = bool(tc.get("stop_at_step", 0)) and step >= tc.stop_at_step
        over_time = over_time or early_stop
        if step % tc.ckpt_every == 0 or step == tc.max_steps or over_time:
            ckpt.save(
                out_dir / "checkpoints",
                step,
                model,
                opt,
                {
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "fingerprint": fingerprint,
                    "model_config": mcfg.__dict__,
                },
                info,
                sharded=tc.strategy == "fsdp" and info.distributed,
                keep_last=tc.keep_last,
            )
        if over_time:
            stop_reason = "stop_at_step" if early_stop else "wall_clock"
            break

    summary = {
        "final_step": step,
        "stop_reason": stop_reason,
        "resumed_from": resumed_from,
        "seconds": time.time() - t_start,
        "seconds_to_first_step": first_step_time,
    }
    if info.is_main:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.close()
    cleanup()
    return summary


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
