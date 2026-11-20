"""Compute-budget printout required before every GPU job (plan §0 rule 5).

The plan prices jobs as hours x GPU_PRICE_PER_HOUR x n_gpus. On the free tier the price is 0, so
the meaningful number is GPU-hours against the weekly Kaggle quota. Kaggle counts quota per
*session* hour (a T4 x2 session uses one session-hour of quota, not two) — verify in the Kaggle
UI, which shows remaining hours, before launching.
"""

from __future__ import annotations

import os


def estimate(hours: float, n_gpus: int, label: str) -> str:
    price = float(os.environ.get("GPU_PRICE_PER_HOUR", "0") or 0)
    cost = hours * price * n_gpus
    lines = [
        f"[budget] {label}",
        f"[budget]   estimated wall time : {hours:.2f} h on {n_gpus} GPU(s)",
        f"[budget]   GPU-hours           : {hours * n_gpus:.2f}",
        f"[budget]   Kaggle session-hours: {hours:.2f} (check remaining weekly quota first)",
        f"[budget]   estimated cost      : ${cost:.2f}"
        + ("  (free tier)" if price == 0 else "  (GPU_PRICE_PER_HOUR set: NOT free!)"),
    ]
    msg = "\n".join(lines)
    print(msg, flush=True)
    return msg
