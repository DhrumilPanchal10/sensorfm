"""Drive the Kaggle runs from this machine with the ``kaggle`` CLI (free tier, $0).

    python scripts/kaggle_ctl.py push-code                    # git archive HEAD -> sensorfm-code
    python scripts/kaggle_ctl.py push build-data              # CPU, Internet on
    python scripts/kaggle_ctl.py push ladder-scaling --after build-data
    python scripts/kaggle_ctl.py push pretrain --after build-data -e STEPS=20000
    python scripts/kaggle_ctl.py push pretrain --name pretrain-2 --after build-data \\
        --after pretrain -e STEPS=20000 -e RESUME=true      # next 12 h session
    python scripts/kaggle_ctl.py push downstream --after build-data --after pretrain-2
    python scripts/kaggle_ctl.py status build-data
    python scripts/kaggle_ctl.py pull build-data --import-results

Every run is tied to one commit: ``push-code`` refuses a dirty tree, and the kernel refuses a
stale code snapshot. Kernels are private. Later jobs mount earlier jobs' outputs (``--after``),
so shards and checkpoints never travel through this machine. Downloads skip shards, downstream
data and checkpoints by default.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
USER = "dhrumilvinaypanchal"
CODE = f"{USER}/sensorfm-code"
KAGGLE = shutil.which("kaggle") or str(Path.home() / ".local" / "bin" / "kaggle")
SKIP_BIG = r"^(?!.*(shards/|downstream/|checkpoints/|mlruns/|exports/.*\.engine)).*$"

# name -> (gpu, kernel timeout s, shell command run from the repo root on Kaggle)
JOBS: dict[str, tuple[bool, int, str]] = {
    "build-data": (False, 6 * 3600,
                   "WORK=/kaggle/working TMP_CACHE=/tmp/hf bash kaggle/build_data.sh"),
    "check-data": (False, 3600,  # CPU: verify the build-data output without GPU quota
                   "pip install -q typer pyyaml && python scripts/check_data.py"
                   " --shards $(find -L /kaggle/input -maxdepth 6 -type d -path '*/shards/main' | head -1)"
                   " --downstream $(find -L /kaggle/input -maxdepth 6 -type d -path '*/shards/main' | head -1)/../../downstream"
                   " && mkdir -p /kaggle/working/results && cp -r results/data /kaggle/working/results/"),
    "ladder-scaling": (True, 3 * 3600,
                       "CUDA_VISIBLE_DEVICES=0 MODEL=${MODEL:-s20m} bash kaggle/run_ladder.sh"
                       " && bash kaggle/run_scaling.sh"),
    "pretrain": (True, 12 * 3600,
                 "MODEL=${MODEL:-s20m} RUN_ID=${MODEL:-s20m}_main NPROC=2 bash kaggle/pretrain.sh"),
    "downstream": (True, 9 * 3600,
                   "MODEL=${MODEL:-s20m} bash kaggle/run_downstream.sh"
                   " && TAG=${MODEL:-s20m} bash kaggle/run_deploy.sh"),
}  # fmt: skip

app = typer.Typer(add_completion=False)


def _kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def _slug(name: str) -> str:
    return "sensorfm-" + name


@app.command("push-code")
def push_code() -> None:
    """Upload ``git archive HEAD`` as a new version of the private code dataset."""
    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty: commit first so the run maps to a hash")
    commit = _git("rev-parse", "HEAD")
    exists = _kaggle("datasets", "status", CODE, check=False).returncode == 0
    with tempfile.TemporaryDirectory() as d:
        stage = Path(d)
        _git("archive", "--format=tar.gz", "-o", str(stage / "sensorfm.tar.gz"), "HEAD")
        (stage / "COMMIT").write_text(commit + "\n")
        meta = {"id": CODE, "title": "sensorfm code snapshot", "licenses": [{"name": "other"}]}
        (stage / "dataset-metadata.json").write_text(json.dumps(meta))
        if exists:
            _kaggle("datasets", "version", "-p", str(stage), "-m", commit[:12], "-q")
        else:  # private unless --public is passed, which we never do
            _kaggle("datasets", "create", "-p", str(stage), "-q")
    for _ in range(60):
        if "ready" in _kaggle("datasets", "status", CODE, check=False).stdout:
            print(f"{CODE} at {commit[:12]}")
            return
        time.sleep(10)
    raise SystemExit(f"{CODE} not ready after 10 min")


@app.command()
def push(
    job: str,
    name: str | None = typer.Option(None, help="kernel name (default: the job name)"),
    after: list[str] = typer.Option([], help="earlier kernel name(s) whose output to mount"),
    env: list[str] = typer.Option([], "--env", "-e", help="KEY=VALUE for the job's command"),
) -> None:
    if job not in JOBS:
        raise SystemExit(f"unknown job {job!r}; one of {sorted(JOBS)}")
    gpu, timeout_s, cmd = JOBS[job]
    kname = name or job
    head = _git("rev-parse", "HEAD")
    envd = dict(e.split("=", 1) for e in env)
    header = (f"EXPECTED_COMMIT = {head!r}\nJOB = {kname!r}\nCMD = {cmd!r}\n"
              f"ENV = {json.dumps(envd)}\n\n")  # fmt: skip
    with tempfile.TemporaryDirectory() as d:
        stage = Path(d)
        (stage / "job.py").write_text(header + (ROOT / "kaggle" / "bootstrap.py").read_text())
        meta = {
            "id": f"{USER}/{_slug(kname)}", "title": _slug(kname), "code_file": "job.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true" if gpu else "false", "enable_internet": "true",
            "dataset_sources": [CODE], "competition_sources": [], "model_sources": [],
            "kernel_sources": [f"{USER}/{_slug(a)}" for a in after],
        }  # fmt: skip
        (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        args = ["kernels", "push", "-p", str(stage), "-t", str(timeout_s)]
        if gpu:
            args += ["--accelerator", "NvidiaTeslaT4"]  # "GPU T4 x2"
        out = _kaggle(*args).stdout.strip()
        print(out)
        if "error" in out.lower():  # the CLI exits 0 on e.g. "Maximum weekly GPU quota reached"
            raise SystemExit(f"push of {kname} rejected by Kaggle")
    print(f"pushed {kname} ({'GPU T4 x2' if gpu else 'CPU'}) at {head[:12]}: {cmd} {envd}")


@app.command()
def status(name: str) -> None:
    print(_kaggle("kernels", "status", f"{USER}/{_slug(name)}").stdout.strip())


@app.command()
def pull(name: str, import_results: bool = False, everything: bool = False) -> None:
    """Download outputs + log to runs/kaggle/NAME (no shards/checkpoints unless --everything)."""
    dst = ROOT / "runs" / "kaggle" / name
    dst.mkdir(parents=True, exist_ok=True)
    pattern = [] if everything else ["--file-pattern", SKIP_BIG]
    _kaggle("kernels", "output", f"{USER}/{_slug(name)}", "-p", str(dst), "-o", *pattern)
    print(f"outputs in {dst}")
    if import_results and (dst / "results").exists():
        for f in sorted(p for p in (dst / "results").rglob("*") if p.is_file()):
            to = ROOT / "results" / f.relative_to(dst / "results")
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, to)
            print("imported", to.relative_to(ROOT))


if __name__ == "__main__":
    app()
