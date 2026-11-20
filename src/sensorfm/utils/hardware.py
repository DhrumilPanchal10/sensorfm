"""Hardware/environment fingerprint logged with every run and benchmark result.

Numbers without the hardware they were measured on are not comparable, so every
``results/*.json`` embeds this dict.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

import torch


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda": torch.cuda.is_available(),
        "git": git_hash(),
        "kaggle": "KAGGLE_KERNEL_RUN_TYPE" in os.environ,
    }
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(n)]
        props = torch.cuda.get_device_properties(0)
        info["gpu_mem_gb"] = round(props.total_memory / 1e9, 1)
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["cuda_version"] = torch.version.cuda
    return info
