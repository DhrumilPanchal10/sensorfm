"""CWRU bearing classification: leave-one-load-out x regimes x seeds -> results/eval/bearing_<tag>.json.

    python scripts/eval_bearing.py --ckpt <checkpoint dir> --tag s20m_pre
    python scripts/eval_bearing.py --model s20m --tag s20m_scratch

Each test load (0-3) is held out in turn; model selection uses another training load, so no
recording appears in more than one of train / val / test. Reports 10-class macro-F1 and the
7-class variant without ball faults (see the CWRU label check in docs/dataset_card.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

from sensorfm.finetune.bearing import run_bearing
from sensorfm.finetune.common import FitCfg
from sensorfm.model.sensorfm import ModelConfig
from sensorfm.utils.hardware import hardware_info


def main(
    tag: str = typer.Option(...),
    ckpt: Path | None = None,
    model: str = "s20m",
    root: Path = Path("data/downstream/cwru"),
    regimes: str = "linear_probe,full,few_shot",
    loads: str = "0,1,2,3",
    seeds: str = "0,1,2",
    epochs: int = 30,
    out_dir: Path = Path("results/eval"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = yaml.safe_load(Path(f"configs/model/{model}.yaml").read_text())
    mcfg = ModelConfig(**raw, context_length=512)
    runs: list[dict[str, Any]] = []
    for regime in regimes.split(","):
        for load in (int(x) for x in loads.split(",")):
            for seed in (int(s) for s in seeds.split(",")):
                r = run_bearing(root, load, "full" if regime == "few_shot" else regime, seed, mcfg,
                                ckpt, few_shot=0.1 if regime == "few_shot" else 0.0,
                                device=device, fit_cfg=FitCfg(epochs=epochs))  # fmt: skip
                r["regime"] = regime
                runs.append(r)
                print(json.dumps(r), flush=True)
    summary = {}
    for regime in regimes.split(","):
        rs = [r for r in runs if r["regime"] == regime]
        summary[regime] = {k: {"mean": float(np.mean([r[k] for r in rs])),
                               "std": float(np.std([r[k] for r in rs]))}
                           for k in ("macro_f1", "macro_f1_no_ball", "accuracy")}  # fmt: skip
    out = out_dir / f"bearing_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "ckpt": str(ckpt) if ckpt else None, "model": model,
                               "runs": runs, "summary": summary, "hardware": hardware_info()},
                              indent=2))  # fmt: skip
    print(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
