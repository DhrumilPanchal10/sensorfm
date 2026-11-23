"""Run a serialized TensorRT engine (TensorRT 10 tensor-address API) with torch CUDA buffers.

TensorRT 10 removed the binding-index API; I/O is by tensor name: ``set_input_shape`` +
``set_tensor_address`` per tensor, then ``execute_async_v3(stream)``. Buffers are torch CUDA
tensors so no pycuda / cuda-python dependency is needed. ``check()`` fails loudly if the installed
TensorRT lacks these entry points (plan §0 rule 4: verify APIs at runtime). GPU-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sensorfm.export.trt_build import check_api

_CTX_API = ("set_input_shape", "set_tensor_address", "execute_async_v3", "get_tensor_shape")


class TrtRunner:
    def __init__(self, engine_path: Path) -> None:
        import torch

        trt = check_api()
        self._trt, self._torch = trt, torch
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path}")
        self.ctx = self.engine.create_execution_context()
        missing = [a for a in _CTX_API if not hasattr(self.ctx, a)]
        if missing:
            raise RuntimeError(f"TensorRT {trt.__version__} context lacks {missing}")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = self.engine.get_tensor_mode
        self.inputs = [n for n in names if mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.Stream()

    def to_device(self, feeds: dict[str, npt.NDArray[Any]]) -> dict[str, Any]:
        return {k: self._torch.as_tensor(v).cuda().contiguous() for k, v in feeds.items()}

    def run_device(self, dev: dict[str, Any]) -> Any:
        """Inputs already on the GPU (what latency timing uses); returns the first output."""
        torch, trt = self._torch, self._trt
        for n in self.inputs:
            self.ctx.set_input_shape(n, tuple(dev[n].shape))
            self.ctx.set_tensor_address(n, int(dev[n].data_ptr()))
        outs = {}
        for n in self.outputs:
            np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(n)))
            dtype = torch.from_numpy(np.empty(0, np_dtype)).dtype
            outs[n] = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=dtype, device="cuda")
            self.ctx.set_tensor_address(n, int(outs[n].data_ptr()))
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return outs[self.outputs[0]]

    def run(self, feeds: dict[str, npt.NDArray[Any]]) -> npt.NDArray[np.float32]:
        out: npt.NDArray[np.float32] = self.run_device(self.to_device(feeds)).float().cpu().numpy()
        return out
