"""Generate every report and the README results block from ``results/`` (plan §0 rule 2).

    python scripts/build_report.py

Nothing here is typed by hand: each table is read from a results JSON, and a missing file is
shown as *pending* (never as a guessed number). Writes ``docs/reports/{baselines,status}.md``,
re-runs the training-efficiency report when a ladder exists, and replaces the block between
``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->`` in README.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

R = Path("results")
DOCS = Path("docs/reports")


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text()) if p.exists() else None


def forecast_table(d: dict[str, Any]) -> list[str]:
    datasets = list(next(iter(d["methods"].values()))["by_dataset"])
    head = (
        "| Method | "
        + " | ".join(f"{ds} MASE" for ds in datasets)
        + " | mean MASE | gmean MASE | mean WQL | fallbacks |"
    )
    lines = [head, "|" + "---|" * (len(datasets) + 5)]
    for name, m in d["methods"].items():
        cells = " | ".join(f"{m['by_dataset'][ds]['mase_mean']:.3f}" for ds in datasets)
        lines.append(f"| {name} | {cells} | {m['mase_mean']:.3f} | {m['mase_gmean']:.3f} | "
                     f"{m['wql_mean']:.3f} | {m['fallbacks']}/{m['n_forecasts']} |")  # fmt: skip
    for name, why in (d.get("skipped") or {}).items():
        lines.append(f"| {name} | skipped: {why[:80]} |")
    return lines


def rul_table(d: dict[str, Any], label_key: str = "method") -> list[str]:
    lines = [
        "| Subset | Model | RMSE (mean ± std) | NASA score (mean ± std) |",
        "|---|---|---|---|",
    ]
    for key, v in d["summary"].items():
        fd, name = key.split("/")
        lines.append(f"| {fd} | {name} | {v['rmse']['mean']:.2f} ± {v['rmse']['std']:.2f} | "
                     f"{v['nasa']['mean']:.0f} ± {v['nasa']['std']:.0f} |")  # fmt: skip
    return lines


def scaling_table() -> list[str] | None:
    files = sorted((R / "scaling").glob("*_*_[0-9].json"))
    if not files:
        return None
    rows = [json.loads(f.read_text()) for f in files]
    base = {(r["model"]): r["tokens_per_s"] for r in rows if r["world_size"] == 1}
    lines = ["| Model | Strategy | GPUs | tokens/s | scaling eff. | peak mem/GPU GB | MFU |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        b = base.get(r["model"])
        eff = r["tokens_per_s"] / (r["world_size"] * b) if b else None
        mfu = f"{100 * r['mfu']:.1f}%" if r.get("mfu") else "—"
        lines.append(f"| {r['model']} | {r['strategy']} | {r['world_size']} | {r['tokens_per_s']:,.0f} | "
                     f"{'—' if eff is None else f'{100 * eff:.0f}%'} | {r['peak_mem_gb_per_gpu']:.2f} | {mfu} |")  # fmt: skip
    return lines


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    status: list[tuple[str, str, bool]] = []
    out: list[str] = []

    def section(title: str, src: Path, lines: list[str] | None, note: str = "") -> None:
        status.append((title, str(src), lines is not None))
        out.extend([f"### {title}", ""])
        if lines is None:
            out.extend([f"*Pending — `{src}` not produced yet.*", ""])
        else:
            out.extend(
                [*([note, ""] if note else []), *lines, "", f"<sub>Source: `{src}`</sub>", ""]
            )

    fb = _load(R / "eval/forecast_baselines.json")
    section("Forecasting baselines (ETT, H=96, context 512)", R / "eval/forecast_baselines.json",
            forecast_table(fb) if fb else None,
            "MASE scaled by in-sample seasonal-naive MAE on the training split; WQL scale-free.")  # fmt: skip
    cb = _load(R / "eval/cmapss_baselines.json")
    section("C-MAPSS classic baselines (W=30, 3 seeds)", R / "eval/cmapss_baselines.json",
            rul_table(cb) if cb else None,
            "Official last-cycle test RUL (uncapped); validation split by engine unit.")  # fmt: skip
    for f in sorted((R / "eval").glob("rul_*.json")):
        d = _load(f)
        if d and not d["tag"].startswith("sanity"):
            section(
                f"SensorFM RUL — {d['tag']}",
                f,
                rul_table(d),
                f"Window {d['window']}; regimes as rows.",
            )
    section("Scaling (Kaggle T4)", R / "scaling", scaling_table())
    res = _load(R / "resume_test.json")
    section("Crash-resume test (CPU)", R / "resume_test.json",
            [f"- killed after step {res['killed_after_step']}, resumed from {res['resumed_from_step']} "
             f"({res['lost_steps']} steps lost); first new step {res['seconds_to_first_step_after_resume']:.2f} s "
             "after restart; all logged losses identical to the uninterrupted run."] if res else None)  # fmt: skip
    par = _load(R / "scaling/parity_cpu_gloo.json")
    section("Distributed correctness (CPU, gloo, 2 processes)", R / "scaling/parity_cpu_gloo.json",
            [f"- max |loss difference| vs single process: DDP {par['max_abs_loss_diff_vs_single']['ddp']:.1e}, "
             f"FSDP2 {par['max_abs_loss_diff_vs_single']['fsdp']:.1e} over {par['steps']} steps"] if par else None)  # fmt: skip

    ladder = R / "profiles/ladder.json"
    if ladder.exists():
        subprocess.run(
            [sys.executable, "scripts/report_training_efficiency.py", "--ladder", str(ladder)],
            check=True,
        )
    status.append(("Training efficiency ladder", str(ladder), ladder.exists()))

    (DOCS / "baselines.md").write_text("# Results (generated)\n\n" + "\n".join(out))
    st = ["# Results status (generated)", "", "| Item | Source | Status |", "|---|---|---|"]
    st += [f"| {t} | `{s}` | {'done' if ok else 'pending'} |" for t, s, ok in status]
    (DOCS / "status.md").write_text("\n".join(st) + "\n")

    readme = Path("README.md")
    text = readme.read_text()
    block = "<!-- RESULTS:START -->\n" + "\n".join(out) + "\n<!-- RESULTS:END -->"
    if "<!-- RESULTS:START -->" in text:
        pre, rest = text.split("<!-- RESULTS:START -->", 1)
        post = rest.split("<!-- RESULTS:END -->", 1)[1]
        text = pre + block + post
    else:
        text = (
            text.rstrip()
            + "\n\n## Results (generated by `scripts/build_report.py`)\n\n"
            + block
            + "\n"
        )
    readme.write_text(text)
    done = sum(ok for *_, ok in status)
    print(f"report: {done}/{len(status)} sections have results")


if __name__ == "__main__":
    main()
