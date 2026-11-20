"""UCR anomaly archive with SensorFM forecast-error scores -> results/eval/ucr_<tag>.json.

    python scripts/eval_ucr.py --ckpt <checkpoint dir> --tag s20m_zeroshot
    python scripts/eval_ucr.py --ckpt <checkpoint dir> --tag s20m_prefix --finetune-steps 200

Each series is independent (archive rule: no information shared across series). With
``--finetune-steps`` a *fresh copy* of the pretrained model is adapted to that series' anomaly-free
training prefix before scoring. Accuracy uses the archive's own rule (see downstream/ucr_anomaly).
Per-series results are stored so the paired bootstrap can compare variants on the same series.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer

from sensorfm.downstream import ucr_anomaly
from sensorfm.finetune.anomaly import anomaly_scores, finetune_on_prefix
from sensorfm.finetune.common import load_encoder
from sensorfm.utils.hardware import hardware_info


def main(
    ckpt: Path = typer.Option(...),
    tag: str = typer.Option(...),
    root: Path = Path("data/downstream/ucr"),
    finetune_steps: int = 0,
    limit: int = 0,
    seed: int = 0,
    out_dir: Path = Path("results/eval"),
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_encoder(ckpt, None, device)
    series = ucr_anomaly.load_archive(root)
    if limit:
        series = series[:limit]
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for s in series:
        model = copy.deepcopy(base)
        if finetune_steps:
            torch.manual_seed(seed)
            finetune_on_prefix(model, s.values, s.train_end, finetune_steps, seed, device)
        scores = anomaly_scores(model, s.values, s.train_end, device)
        pred = ucr_anomaly.predict_location(scores, s.train_end)
        rows.append({"id": s.id, "name": s.name, "pred": pred, "begin": s.begin, "end": s.end,
                     "correct": ucr_anomaly.is_correct(pred, s.begin, s.end)})  # fmt: skip
        print(json.dumps(rows[-1]), flush=True)
    acc = float(np.mean([r["correct"] for r in rows]))
    out = out_dir / f"ucr_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "ckpt": str(ckpt), "finetune_steps": finetune_steps,
                               "n_series": len(rows), "accuracy": acc, "series": rows,
                               "seconds": round(time.time() - t0, 1), "hardware": hardware_info()},
                              indent=2))  # fmt: skip
    print(f"accuracy {acc:.3f} on {len(rows)} series -> {out}")


if __name__ == "__main__":
    typer.run(main)
