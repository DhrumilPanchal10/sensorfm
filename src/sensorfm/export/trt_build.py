"""TensorRT FP16 / INT8 engines for the GPU path (Kaggle T4: Turing has FP16 + INT8 tensor cores).

Written against the TensorRT 10 Python API (``tensorrt`` wheel; explicit-batch networks are the
only kind, ``build_serialized_network``, optimization profiles for dynamic batch). **Verify the
installed version's API at runtime** (plan §0 rule 4) — ``check_api()`` fails loudly if the
expected entry points are missing instead of building something silently different.

INT8 uses an entropy calibrator fed with representative windows (``calibrate.py``); layers that
lose too much accuracy can be pinned to FP16 with ``fp16_layers`` (matched by layer-name
substring), which the deployment report documents per variant. Not importable without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def check_api() -> Any:
    import tensorrt as trt

    needed = ["Builder", "OnnxParser", "BuilderFlag", "Runtime", "IInt8EntropyCalibrator2"]
    missing = [n for n in needed if not hasattr(trt, n)]
    if missing:
        raise RuntimeError(f"TensorRT {trt.__version__} lacks {missing}; update trt_build.py")
    return trt


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    input_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    calibrator: Any = None,
    fp16_layers: tuple[str, ...] = (),
    workspace_gb: float = 2.0,
) -> Path:
    """``input_shapes``: name -> (min, opt, max) shapes for the dynamic-batch profile."""
    trt = check_api()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in input_shapes.items():
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)
    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        if calibrator is None:
            raise ValueError("INT8 needs a calibrator")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)
        if fp16_layers:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if any(s in layer.name for s in fp16_layers):
                    layer.precision = trt.float16
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path
