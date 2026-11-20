"""Export fine-tuned SensorFM models to ONNX (fixed or dynamic batch).

Two deployable graphs:

- **RUL / classification**: ``Regressor`` (channel-independent encoder + head), inputs
  ``x (B, C, W) float32`` and ``mask (B, W) bool``.
- **Forecast / anomaly scoring**: ``ForecastModule`` wraps the encoder + quantile head with the
  horizon baked in; input ``context (B, c) float32``, output ``quantiles (B, H, Q)`` on the
  original scale — so the serving code needs no PyTorch.

The encoder is switched to ``export_mode`` (always-built attention mask, no data-dependent
branch). Everything else in the forward is plain tensor math, which keeps the graph portable to
ONNX Runtime and TensorRT.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sensorfm.model.sensorfm import SensorFM


class ForecastModule(nn.Module):
    def __init__(self, model: SensorFM, context: int, horizon: int) -> None:
        super().__init__()
        p = model.cfg.patch_len
        if horizon % p or (context + horizon) % p or context + horizon > model.cfg.context_length:
            raise ValueError("context/horizon incompatible with the model's patching")
        self.model = model
        self.context = context
        self.horizon = horizon
        self.n_h = horizon // p

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        b = context.shape[0]
        x = torch.cat(
            [context, torch.zeros(b, self.horizon, dtype=context.dtype, device=context.device)], 1
        )
        obs = torch.ones_like(x, dtype=torch.bool)
        n = (self.context + self.horizon) // self.model.cfg.patch_len
        hide = torch.arange(n, device=context.device) >= (n - self.n_h)
        hide = hide.unsqueeze(0).expand(b, n)
        out = self.model(x, obs, hide)
        q = out.quantiles[:, -self.n_h :].reshape(b, self.horizon, -1)
        return self.model.revin.denormalize(q, out.loc, out.scale)


def _set_export_mode(module: nn.Module, on: bool) -> None:
    for m in module.modules():
        if hasattr(m, "export_mode"):
            setattr(m, "export_mode", on)  # noqa: B010


def export_onnx(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    path: Path,
    dynamic_batch: bool = True,
    opset: int = 17,
) -> Path:
    module.eval()
    _set_export_mode(module, True)
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {n: {0: "batch"} for n in [*input_names, *output_names]} if dynamic_batch else None
    )
    try:
        with torch.no_grad():
            torch.onnx.export(
                module, example_inputs, str(path), input_names=input_names,
                output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset,
                dynamo=False,
            )  # fmt: skip
    finally:
        _set_export_mode(module, False)
    return path
