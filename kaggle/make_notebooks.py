"""Generate the importable Kaggle notebooks in kaggle/notebooks/ (run after editing this file).

    python kaggle/make_notebooks.py

01_build_data      CPU, Internet ON                  -> private dataset `sensorfm-data`
02_ladder_scaling  GPU T4 x2: throughput ladder (1 GPU) + 1-vs-2 GPU DDP/FSDP2 scaling study
03_pretrain        GPU T4 x2, DDP: resumable across sessions (attach previous output to resume)
04_downstream      GPU T4 x1: fine-tune / evaluate a checkpoint + scratch twin + zero-shot baselines
                   + deployment study (ONNX / INT8 / TensorRT)

Every GPU notebook prints its budget estimate before launching anything (cost is $0; the
estimate is in session-hours of the weekly Kaggle quota).
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks"
REPO = "https://$GH_TOKEN@github.com/DhrumilPanchal10/sensorfm.git"

SECRETS = """import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ['GH_TOKEN'] = s.get_secret('GH_TOKEN')
try:
    os.environ['HF_TOKEN'] = s.get_secret('HF_TOKEN')
except Exception:
    print('no HF_TOKEN secret (optional)')"""

CLONE = f"""%%bash
set -e
cd /kaggle/working
rm -rf sensorfm-src && git clone -q {REPO} sensorfm-src
cd sensorfm-src && git log --oneline -1"""

CLEANUP = """# Keep the saved output small: results/ and runs/ were copied to /kaggle/working already.
import shutil; shutil.rmtree('/kaggle/working/sensorfm-src', ignore_errors=True)
!du -sh /kaggle/working/* 2>/dev/null"""


def md(t: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t: str) -> dict[str, object]:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def write(name: str, cells: list[dict[str, object]]) -> None:
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},  # fmt: skip
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote", OUT / f"{name}.ipynb")


SETUP_HINT = (
    "Settings: **Internet: On**. Secrets (Add-ons → Secrets): `GH_TOKEN`. "
    "Input: **Add Input → Your Datasets → `sensorfm-data`**."
)

write("01_build_data", [
    md("# SensorFM — build data (Kaggle, CPU, free)\n\nSettings: **Accelerator: None**, **Internet: On**. "
       "Secrets: `GH_TOKEN` (read-only GitHub token), optionally `HF_TOKEN`.\n\nAfter the run: *Save Version*, "
       "then **Output → New Dataset → `sensorfm-data` (Private)**."),
    code(SECRETS),
    code(CLONE + "\nWORK=/kaggle/working TMP_CACHE=/tmp/hf bash kaggle/build_data.sh 2>&1 | grep -v Warning"),
    code("import json\nm = json.load(open('/kaggle/working/shards/main/manifest.json'))\n"
         "for split in ('train', 'val'):\n    sp = m['splits'][split]\n"
         "    print(split, sp['n_series'], 'series', f\"{sp['n_obs']:,}\", 'obs', sp['obs_per_domain'])\n"
         "print('decontaminated:', m['build']['decontaminated_series'])\n"
         "print('manifest sha256:', m['manifest_sha256'])"),
    code(CLEANUP),
])  # fmt: skip

write("02_ladder_scaling", [
    md("# SensorFM — throughput ladder + scaling study (Tasks 3.1–3.3)\n\n**Accelerator: GPU T4 x2**. "
       + SETUP_HINT + "\n\nBudget ≈ 1.1 session-hours. Results land in `/kaggle/working/results`."),
    code(SECRETS),
    code(CLONE),
    code("%%bash\nset -e\ncd /kaggle/working/sensorfm-src\n"
         "CUDA_VISIBLE_DEVICES=0 MODEL=s20m bash kaggle/run_ladder.sh 2>&1 | grep -v Warning"),
    code("%%bash\nset -e\ncd /kaggle/working/sensorfm-src\nbash kaggle/run_scaling.sh 2>&1 | grep -v Warning\n"
         "source kaggle/gpu_setup.sh >/dev/null && python scripts/build_report.py && cat docs/reports/status.md\n"
         "cp -r docs/reports /kaggle/working/reports"),
    code(CLEANUP),
])  # fmt: skip

write("03_pretrain", [
    md("# SensorFM — pretraining (resumable, DDP on T4 x2)\n\n**Accelerator: GPU T4 x2**. " + SETUP_HINT
       + "\n\n**To resume** in a later session: also *Add Input → Your Notebooks → this notebook's previous "
       "version*, set `RESUME = True`. The run stops cleanly with a checkpoint before Kaggle's 12 h limit."),
    code("MODEL = 's20m'      # s20m | m60m | l150m\nSTEPS = 20000       # set from the ladder: scripts/estimate_budget.py\n"
         "MB = 64             # per-GPU micro-batch\nRUN_ID = f'{MODEL}_main'\nRESUME = False"),
    code(SECRETS),
    code(CLONE),
    code("import glob, shutil, os\nif RESUME:\n    prev = glob.glob(f'/kaggle/input/*/runs/{RUN_ID}')\n"
         "    assert prev, 'attach the previous version of this notebook as input to resume'\n"
         "    shutil.copytree(prev[0], f'/kaggle/working/sensorfm-src/runs/{RUN_ID}', dirs_exist_ok=True)\n"
         "    print('resuming from', prev[0], sorted(os.listdir(prev[0] + '/checkpoints')))"),
    code("%%bash -s \"$MODEL\" \"$STEPS\" \"$MB\" \"$RUN_ID\" \"$RESUME\"\nset -e\ncd /kaggle/working/sensorfm-src\n"
         "MODEL=$1 STEPS=$2 MB=$3 RUN_ID=$4 RESUME=$(echo $5 | tr A-Z a-z) NPROC=2 bash kaggle/pretrain.sh 2>&1 | grep -v Warning | tail -40"),
    code("!tail -3 /kaggle/working/runs/*/metrics.jsonl; ls /kaggle/working/runs/*/checkpoints"),
    code(CLEANUP),
])  # fmt: skip

write("04_downstream", [
    md("# SensorFM — downstream evaluation (Phase 4)\n\n**Accelerator: GPU T4 x1** (P100 also fine). " + SETUP_HINT
       + " Also add the **03_pretrain** notebook output as input (for the checkpoint).\n\nBudget ≈ 4.5 session-hours (incl. deployment)."),
    code("MODEL = 's20m'\nTAG = MODEL"),
    code(SECRETS),
    code(CLONE),
    code("import glob\ncks = sorted(glob.glob(f'/kaggle/input/*/runs/{MODEL}_main/checkpoints/step_*'))\n"
         "assert cks, 'attach the 03_pretrain output as input'\nCKPT = cks[-1]\nprint('using', CKPT)"),
    code("%%bash -s \"$CKPT\" \"$MODEL\" \"$TAG\"\nset -e\ncd /kaggle/working/sensorfm-src\n"
         "cp -r /kaggle/input/*/results/eval/cmapss_baselines*.json results/eval/ 2>/dev/null || true\n"
         "CKPT=$1 MODEL=$2 TAG=$3 bash kaggle/run_downstream.sh 2>&1 | grep -v Warning | tail -60"),
    md("## Deployment (Tasks 5.1/5.2): ONNX + INT8 (CPU) and TensorRT FP16/INT8 (T4), ≈ 0.5 h"),
    code("%%bash -s \"$CKPT\" \"$TAG\"\nset -e\ncd /kaggle/working/sensorfm-src\n"
         "CKPT=$1 TAG=$2 bash kaggle/run_deploy.sh 2>&1 | grep -v Warning | tail -30"),
    code(CLEANUP),
])  # fmt: skip
