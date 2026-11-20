"""SensorFM RUL fine-tuning on C-MAPSS: regimes x subsets x seeds -> results/eval/rul_<tag>.json.

    # pretrained
    python scripts/eval_rul.py --ckpt runs/s20m_main/checkpoints/step_00020000 --tag s20m_pre
    # same architecture from scratch (PatchTST-style baseline, ablation E1)
    python scripts/eval_rul.py --model s20m --tag s20m_scratch

Regimes: ``linear_probe``, ``full``, ``few_shot`` (= full on 10% of training units).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

from sensorfm.downstream import cmapss
from sensorfm.finetune.common import FitCfg
from sensorfm.finetune.rul import run_rul
from sensorfm.model.sensorfm import ModelConfig
from sensorfm.utils.hardware import hardware_info


def main(
    tag: str = typer.Option(...),
    ckpt: Path | None = None,
    model: str = "s20m",
    root: Path = Path("data/downstream/cmapss"),
    subsets: str = ",".join(cmapss.SUBSETS),
    regimes: str = "linear_probe,full,few_shot",
    seeds: str = "0,1,2",
    window: int = 64,
    epochs: int = 40,
    out_dir: Path = Path("results/eval"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = yaml.safe_load(Path(f"configs/model/{model}.yaml").read_text())
    mcfg = ModelConfig(**raw, context_length=512)
    runs: list[dict[str, Any]] = []
    for fd in subsets.split(","):
        for regime in regimes.split(","):
            for seed in (int(s) for s in seeds.split(",")):
                r = run_rul(root, fd, "full" if regime == "few_shot" else regime, seed, mcfg, ckpt,
                            window, 0.1 if regime == "few_shot" else 0.0, device,
                            FitCfg(epochs=epochs))  # fmt: skip
                r["regime"] = regime
                runs.append(r)
                print(
                    json.dumps({k: v for k, v in r.items() if k not in ("pred", "true")}),
                    flush=True,
                )
    summary = {}
    for fd in subsets.split(","):
        for regime in regimes.split(","):
            rs = [r for r in runs if r["subset"] == fd and r["regime"] == regime]
            summary[f"{fd}/{regime}"] = {k: {"mean": float(np.mean([r[k] for r in rs])),
                                             "std": float(np.std([r[k] for r in rs]))}
                                         for k in ("rmse", "nasa")}  # fmt: skip
    out = out_dir / f"rul_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "ckpt": str(ckpt) if ckpt else None, "model": model,
                               "window": window, "runs": runs, "summary": summary,
                               "hardware": hardware_info()}, indent=2))  # fmt: skip
    print(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
