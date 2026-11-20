"""Exact parameter counts and training FLOPs/token per model config -> results/model/params.json."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from sensorfm.model.sensorfm import ModelConfig, SensorFM, count_params, non_embedding_params
from sensorfm.utils.flops import train_flops_per_token


def main() -> None:
    rows = {}
    for name in ("tiny", "s20m", "m60m", "l150m"):
        raw = yaml.safe_load(Path(f"configs/model/{name}.yaml").read_text())
        cfg = ModelConfig(**raw, context_length=512)
        with torch.device("meta"):
            m = SensorFM(cfg)
        ne = non_embedding_params(m)
        rows[name] = {
            "params": count_params(m),
            "non_embedding_params": ne,
            "tokens_per_window": cfg.n_patches,
            "train_flops_per_token": train_flops_per_token(
                ne, cfg.n_layers, cfg.n_patches, cfg.d_model
            ),
        }
        print(name, rows[name])
    Path("results/model/params.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
