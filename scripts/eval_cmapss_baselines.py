"""Classic C-MAPSS baselines (GBM, LSTM, CNN) over FD001-FD004 x seeds -> results/eval/.

    python scripts/eval_cmapss_baselines.py --seeds 0,1,2

Per (subset, seed): 20% of training *units* held out for model selection / early stopping
(seed-dependent split), test on the official last-cycle windows. Metrics: RMSE and NASA score
against the official (uncapped) test RUL, plus RMSE against the capped RUL (common in papers).
``--few-shot`` trains on only 10% of the training units (plan Task 4.2 regime).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import typer

from sensorfm.baselines.cmapss_classic import (
    CNNRegressor,
    LSTMRegressor,
    fit_predict_gbm,
    fit_predict_torch,
)
from sensorfm.downstream import cmapss
from sensorfm.eval.metrics import nasa_score, rmse
from sensorfm.utils.hardware import hardware_info


def main(
    root: Path = Path("data/downstream/cmapss"),
    subsets: str = ",".join(cmapss.SUBSETS),
    methods: str = "gbm,lstm,cnn",
    seeds: str = "0,1,2",
    window: int = 30,
    few_shot: float = 0.0,
    out: Path = Path("results/eval/cmapss_baselines.json"),
) -> None:
    res: dict[str, Any] = {"window": window, "few_shot_frac": few_shot or None,
                           "hardware": hardware_info(), "runs": []}  # fmt: skip
    for fd in subsets.split(","):
        data = cmapss.load_subset(root, fd)
        xte, mte, yte = cmapss.test_windows(data, window)
        yte_cap = np.minimum(yte, cmapss.RUL_CAP)
        for seed in [int(s) for s in seeds.split(",")]:
            tr_u, va_u = cmapss.split_units(data.train["unit"], 0.2, seed)
            if few_shot:
                rng = np.random.default_rng(seed)
                tr_u = np.sort(rng.choice(tr_u, max(2, round(few_shot * len(tr_u))), replace=False))
            xtr, mtr, ytr = cmapss.train_windows(data, tr_u, window)
            xva, mva, yva = cmapss.train_windows(data, va_u, window)
            for method in methods.split(","):
                t0 = time.time()
                if method == "gbm":
                    pred, info = fit_predict_gbm(xtr, mtr, ytr, xva, mva, yva, xte, mte, seed)
                else:
                    c = xtr.shape[1]
                    model = LSTMRegressor(c) if method == "lstm" else CNNRegressor(c)
                    pred, info = fit_predict_torch(
                        model, xtr, mtr, ytr, xva, mva, yva, xte, mte, seed
                    )
                run = {"subset": fd, "seed": seed, "method": method, "n_train_units": len(tr_u),
                       "rmse": rmse(pred, yte), "nasa": nasa_score(pred, yte),
                       "rmse_capped_target": rmse(pred, yte_cap), "seconds": round(time.time() - t0, 1),
                       **info}  # fmt: skip
                res["runs"].append(run)
                print(json.dumps(run), flush=True)
    summary: dict[str, Any] = {}
    for fd in subsets.split(","):
        for method in methods.split(","):
            rs = [r for r in res["runs"] if r["subset"] == fd and r["method"] == method]
            summary[f"{fd}/{method}"] = {
                k: {
                    "mean": float(np.mean([r[k] for r in rs])),
                    "std": float(np.std([r[k] for r in rs])),
                }
                for k in ("rmse", "nasa", "rmse_capped_target")
            }
    res["summary"] = summary
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for k, v in summary.items():
        print(
            f"{k:16s} RMSE {v['rmse']['mean']:.2f}±{v['rmse']['std']:.2f}  NASA {v['nasa']['mean']:.0f}"
        )


if __name__ == "__main__":
    typer.run(main)
