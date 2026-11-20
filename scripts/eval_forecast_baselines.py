"""Forecasting baselines on the ETT benchmark -> results/eval/forecast_baselines.json.

    python scripts/eval_forecast_baselines.py --methods seasonal_naive,ets --horizon 96

Protocol (docs/dataset_card.md): ETTh1/h2/m1/m2, each of 7 columns univariate, context 512,
rolling origins in the test span with stride = horizon. Per series: MASE (scale = in-sample
seasonal-naive MAE on the *training* split), CRPS (2 x mean pinball over q=0.1..0.9) and WQL.
Aggregates: mean and geometric mean of MASE over series, mean WQL.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import typer

from sensorfm.baselines.naive import seasonal_naive_quantiles
from sensorfm.baselines.stats_models import ets_quantiles
from sensorfm.downstream.forecast_bench import DATASETS, load_dataset, rolling_windows
from sensorfm.eval.metrics import QUANTILES, crps_quantile, mase, wql
from sensorfm.utils.hardware import hardware_info


def main(
    root: Path = Path("data/downstream/ett"),
    methods: str = "seasonal_naive,ets",
    horizon: int = 96,
    context: int = 512,
    datasets: str = ",".join(DATASETS),
    out: Path = Path("results/eval/forecast_baselines.json"),
) -> None:
    results: dict[str, Any] = {"horizon": horizon, "context": context, "quantiles": QUANTILES,
                               "methods": {}, "hardware": hardware_info()}  # fmt: skip
    for method in methods.split(","):
        t0 = time.time()
        per_series = []
        fallbacks = 0
        n_fc = 0
        for ds in datasets.split(","):
            for s in load_dataset(root, ds):
                ctx, tgt = rolling_windows(s, context, horizon)
                preds = []
                for c in ctx:
                    if method == "seasonal_naive":
                        q = seasonal_naive_quantiles(c, horizon, s.season, QUANTILES)
                    elif method == "ets":
                        q, ok = ets_quantiles(c, horizon, s.season, QUANTILES)
                        fallbacks += int(not ok)
                    else:
                        raise typer.BadParameter(f"unknown method {method}")
                    preds.append(q)
                    n_fc += 1
                qp = np.stack(preds)  # (n, H, Q)
                median = qp[..., QUANTILES.index(0.5)]
                insample = s.values[: s.train_end]
                per_series.append({
                    "dataset": ds, "column": s.column, "n_windows": len(ctx),
                    "mase": mase(median, tgt, insample, s.season),
                    "crps": crps_quantile(qp, tgt), "wql": wql(qp, tgt),
                })  # fmt: skip
            print(f"{method} {ds} done ({time.time() - t0:.0f}s)", flush=True)
        mases = np.array([r["mase"] for r in per_series])
        by_ds = {
            ds: {
                "mase_mean": float(np.mean([r["mase"] for r in per_series if r["dataset"] == ds])),
                "wql_mean": float(np.mean([r["wql"] for r in per_series if r["dataset"] == ds])),
            }
            for ds in datasets.split(",")
        }
        results["methods"][method] = {
            "mase_mean": float(mases.mean()),
            "mase_gmean": float(np.exp(np.log(mases).mean())),
            "wql_mean": float(np.mean([r["wql"] for r in per_series])),
            "by_dataset": by_ds,
            "fallbacks": fallbacks,
            "n_forecasts": n_fc,
            "seconds": round(time.time() - t0, 1),
            "per_series": per_series,
        }
        print(
            json.dumps({k: v for k, v in results["methods"][method].items() if k != "per_series"})
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
