"""RUL error analysis (Task 4.5) and paired comparisons (Task 4.3) from saved per-unit predictions.

    python scripts/analyze_rul.py results/eval/cmapss_baselines.json results/eval/rul_s20m_pre.json \
        --compare s20m_pre/full:gbm --out results/eval/rul_analysis.json

For every (file, method-or-regime, subset) the per-engine prediction is averaged over seeds; then
- RMSE by true-RUL bucket (late life = small RUL is where errors are costly);
- paired bootstrap of RMSE between two models on the same engines (``--compare A:B``; names are
  ``method`` for baseline files and ``<tag>/<regime>`` for SensorFM files).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import typer

from sensorfm.eval.bootstrap import paired_bootstrap, rmse_from_sq
from sensorfm.eval.metrics import rmse
from sensorfm.eval.slices import by_key, rul_bucket


def _models(files: list[Path]) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(model name, subset) -> (seed-mean prediction per unit, true RUL per unit)."""
    preds: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    trues: dict[tuple[str, str], np.ndarray] = {}
    for f in files:
        d = json.loads(f.read_text())
        for r in d["runs"]:
            if "pred" not in r:
                raise ValueError(
                    f"{f} has no per-unit predictions (re-run with the current scripts)"
                )
            name = r["method"] if "method" in r else f"{d['tag']}/{r['regime']}"
            key = (name, r["subset"])
            preds[key].append(np.asarray(r["pred"]))
            trues[key] = np.asarray(r["true"])
    return {k: (np.mean(v, axis=0), trues[k]) for k, v in preds.items()}


def main(
    files: list[Path],
    compare: list[str] = typer.Option([], help="A:B pairs to bootstrap (A minus B)"),
    out: Path = Path("results/eval/rul_analysis.json"),
    n_boot: int = 10_000,
) -> None:
    models = _models(files)
    res: dict[str, Any] = {"buckets": {}, "comparisons": [], "sources": [str(f) for f in files]}
    for (name, fd), (p, t) in sorted(models.items()):
        res["buckets"][f"{fd}/{name}"] = by_key(rul_bucket(t), p, t, rmse)
    for pair in compare:
        a, b = pair.split(":")
        for fd in sorted({fd for (_, fd) in models}):
            if (a, fd) not in models or (b, fd) not in models:
                continue
            (pa, t), (pb, tb) = models[(a, fd)], models[(b, fd)]
            if not np.array_equal(t, tb):
                raise ValueError(f"{a} and {b} were scored on different targets for {fd}")
            r = paired_bootstrap(
                (pa - t) ** 2, (pb - t) ** 2, aggregate=rmse_from_sq, n_boot=n_boot
            )
            res["comparisons"].append({
                "subset": fd, "a": a, "b": b, "rmse_a": rmse(pa, t), "rmse_b": rmse(pb, t),
                "diff": r.diff, "ci95": [r.lo, r.hi], "significant": r.significant,
                "p_a_better": r.p_a_better, "n_units": r.n_units,
            })  # fmt: skip
            print(json.dumps(res["comparisons"][-1]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
