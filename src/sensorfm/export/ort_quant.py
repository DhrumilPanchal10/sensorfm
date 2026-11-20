"""ONNX Runtime INT8 quantization for the CPU deployment path.

- **Dynamic** INT8: weights quantized offline, activations quantized on the fly per batch. No
  calibration data needed; usually most of the speed-up for linear-heavy transformers.
- **Static** INT8 (QDQ format): activations quantized with ranges fixed from a calibration set
  (512 representative windows by default). Faster and more predictable, but sensitive to
  calibration coverage.
- Sensitive ops can be kept in FP32 via ``nodes_to_exclude`` (e.g. RevIN arithmetic and the final
  head), which the deployment report documents per variant (plan Task 5.1).

Models are pre-processed with ORT's ``quant_pre_process`` (shape inference + graph optimization),
as the ORT docs recommend before static quantization.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def quantize_dynamic_int8(src: Path, dst: Path, exclude: list[str] | None = None) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                     nodes_to_exclude=exclude or [])  # fmt: skip
    return dst


class ArrayCalibrationReader:
    """Feeds calibration batches (dicts of input name -> array) to the static quantizer."""

    def __init__(self, inputs: dict[str, npt.NDArray[Any]], batch: int = 32) -> None:
        n = len(next(iter(inputs.values())))
        self._it: Iterator[dict[str, npt.NDArray[Any]]] = iter(
            [{k: v[i : i + batch] for k, v in inputs.items()} for i in range(0, n, batch)]
        )

    def get_next(self) -> dict[str, npt.NDArray[Any]] | None:
        return next(self._it, None)

    def rewind(self) -> None:  # required by some ORT versions; one pass is enough here
        pass


def quantize_static_int8(
    src: Path,
    dst: Path,
    calib_inputs: dict[str, npt.NDArray[Any]],
    exclude: list[str] | None = None,
    per_channel: bool = True,
) -> Path:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre))
    quantize_static(
        str(pre), str(dst), ArrayCalibrationReader(calib_inputs),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude or [],
    )  # fmt: skip
    pre.unlink(missing_ok=True)
    return dst


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def onnx_node_names(path: Path, op_types: tuple[str, ...]) -> list[str]:
    """Names of nodes of the given op types (to build ``nodes_to_exclude`` lists)."""
    import onnx

    g = onnx.load(str(path)).graph
    return [n.name for n in g.node if n.op_type in op_types]


def _unused(_: np.ndarray) -> None:  # keep numpy import used for type alias clarity
    return None
