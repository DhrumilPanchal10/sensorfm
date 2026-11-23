# --- bootstrap (prepended to every job by scripts/kaggle_ctl.py push; runs on Kaggle only) ---
"""Turn the attached ``sensorfm-code`` dataset (``git archive`` of one commit) into a repo at
/tmp/sf and run the job's shell command there. No GitHub token is needed. Outputs are whatever
the ``kaggle/*.sh`` scripts copy into /kaggle/working (the only directory Kaggle keeps), plus
``job.json`` (commit, command, exit code, wall time). ``EXPECTED_COMMIT``, ``JOB``, ``CMD`` and
``ENV`` are injected above this header by the push command.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

SRC = pathlib.Path("/tmp/sf")
T0 = time.time()


def _files(root: str = "/kaggle/input") -> list[pathlib.Path]:
    return [pathlib.Path(d) / f for d, _, fs in os.walk(root, followlinks=True) for f in fs]


def _install_code() -> str:
    files = _files()
    tars = [p for p in files if p.name == "sensorfm.tar.gz"]
    roots = [p.parent for p in files
             if p.name == "pyproject.toml" and (p.parent / "src" / "sensorfm").is_dir()]  # fmt: skip
    SRC.mkdir(parents=True, exist_ok=True)
    if len(tars) == 1:  # Kaggle may or may not auto-extract archives in datasets
        with tarfile.open(tars[0]) as tf:
            tf.extractall(SRC, filter="data")
    elif len(roots) == 1:
        shutil.copytree(roots[0], SRC, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"sensorfm-code not found under /kaggle/input: {tars} {roots}")
    commits = {p.read_text().strip() for p in files if p.name == "COMMIT"}
    return commits.pop() if len(commits) == 1 else f"unknown {sorted(commits)}"


commit = _install_code()
if commit != EXPECTED_COMMIT:  # noqa: F821  (injected)
    raise RuntimeError(f"code snapshot {commit[:12]} != pushed {EXPECTED_COMMIT[:12]}: "  # noqa: F821
                       "Kaggle mounted a stale sensorfm-code version; re-push the job.")  # fmt: skip
print(f"job {JOB} at {commit[:12]}\n$ {CMD}", flush=True)  # noqa: F821
env = {**os.environ, **ENV, "SENSORFM_COMMIT": commit, "PYTHONUNBUFFERED": "1"}  # noqa: F821
rc = subprocess.run(["bash", "-c", CMD], cwd=SRC, env=env).returncode  # noqa: F821
pathlib.Path("/kaggle/working/job.json").write_text(json.dumps(
    {"job": JOB, "commit": commit, "cmd": CMD, "env": ENV, "returncode": rc,  # noqa: F821
     "wall_s": round(time.time() - T0)}, indent=1))  # fmt: skip
print(f"exit {rc} after {time.time() - T0:.0f} s", flush=True)
sys.exit(rc)
