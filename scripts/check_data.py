"""Fail fast if the attached corpus / downstream data is not what the GPU runs expect.

Run at the start of the first GPU job (cheap: seconds), so a broken data build costs minutes of
GPU quota, not hours. Also copies the manifest and a summary into ``results/data/`` so the corpus
facts travel with the job's (small) outputs.

    python scripts/check_data.py --shards "$SHARD_DIR" --downstream "$DOWNSTREAM_ROOT"
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
import yaml


def main(
    shards: Path = typer.Option(...),
    downstream: Path = typer.Option(...),
    selection: Path = Path("configs/data/lotsa_selection.yaml"),
    min_train_obs: int = 30_000_000,
    min_dataset_frac: float = 0.7,
    out_dir: Path = Path("results/data"),
) -> None:
    problems: list[str] = []
    builds = []
    for d, _, fs in os.walk("/kaggle/input", followlinks=True):
        if "job.json" in fs:
            j = json.loads((Path(d) / "job.json").read_text())
            if j.get("job") == "build-data":
                builds.append(j)
    for j in builds:
        if j.get("returncode") != 0:
            problems.append(f"build-data exited {j.get('returncode')}")
    m = json.loads((shards / "manifest.json").read_text())
    n_sel = len(yaml.safe_load(selection.read_text())["subsets"])
    tr, va = m["splits"]["train"], m["splits"].get("val", {})
    n_ds = len(tr["obs_per_dataset"])
    if tr["n_obs"] < min_train_obs:
        problems.append(f"train n_obs {tr['n_obs']:,} < {min_train_obs:,}")
    if n_ds < min_dataset_frac * n_sel:
        missing = sorted(
            {s["name"] for s in yaml.safe_load(selection.read_text())["subsets"]}
            - set(tr["obs_per_dataset"])
        )
        problems.append(f"only {n_ds}/{n_sel} subsets in train; missing {missing}")
    if not va.get("n_series"):
        problems.append("empty val split")
    for sub in ("cmapss", "ett", "cwru", "ucr", "reference"):
        if not (downstream / sub).is_dir():
            problems.append(f"missing downstream/{sub}")
    ref = downstream / "reference"
    ref_counts = {d.name: len(list(d.glob("*.npy"))) for d in sorted(ref.glob("*")) if d.is_dir()}
    if not ref_counts or min(ref_counts.values()) == 0:
        problems.append(f"decontamination reference empty or partial: {ref_counts}")
    summary = {
        "build_job": builds[0] if builds else None,
        "manifest_sha256": m["manifest_sha256"],
        "train": {k: tr[k] for k in ("n_series", "n_obs", "obs_per_domain")},
        "val": {k: va.get(k) for k in ("n_series", "n_obs")},
        "subsets_in_train": n_ds, "subsets_selected": n_sel,
        "decontaminated_series": m.get("build", {}).get("decontaminated_series"),
        "reference_series": ref_counts,
        "problems": problems,
    }  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=1))
    (out_dir / "corpus_check.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if problems:
        raise SystemExit("data check FAILED: " + "; ".join(problems))
    print("data check OK")


if __name__ == "__main__":
    typer.run(main)
