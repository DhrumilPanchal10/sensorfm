"""Print the GPU-hour / cost estimate for a planned pretraining run (required before launch).

    python scripts/estimate_budget.py --model s20m --steps 20000 --micro-batch 128 --gpus 1

Throughput comes from the measured ladder (``results/profiles/ladder.json``, best rung for this
model) when available; otherwise from an explicit ``--assume-tokens-per-s`` and the estimate is
labeled as an assumption.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sensorfm.utils.budget import estimate


def main(
    model: str = "s20m",
    steps: int = 10000,
    micro_batch: int = 64,
    grad_accum: int = 1,
    gpus: int = 1,
    tokens_per_window: int = 32,
    ladder: Path = Path("results/profiles/ladder.json"),
    assume_tokens_per_s: float = 0.0,
    scaling_efficiency: float = 0.85,
) -> None:
    source = "assumed"
    tok_s = assume_tokens_per_s
    if ladder.exists():
        lad = json.loads(ladder.read_text())
        if lad.get("model") == model:
            ok = [r for r in lad["results"] if "tokens_per_s" in r]
            if ok:
                tok_s = max(r["tokens_per_s"] for r in ok)
                source = f"measured ({ladder}, best rung)"
    if tok_s <= 0:
        raise typer.BadParameter(
            "no measured throughput for this model; pass --assume-tokens-per-s"
        )
    eff_tok_s = tok_s * gpus * (scaling_efficiency if gpus > 1 else 1.0)
    tokens = steps * grad_accum * micro_batch * gpus * tokens_per_window
    hours = tokens / eff_tok_s / 3600 * 1.1  # +10% for eval/checkpoint overhead
    estimate(hours, gpus, f"pretrain {model}: {tokens / 1e9:.2f}B tokens, throughput {source}")


if __name__ == "__main__":
    typer.run(main)
