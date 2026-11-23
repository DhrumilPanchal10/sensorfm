"""Deployment study (Tasks 5.1 + 5.2): fine-tuned RUL model -> ONNX / INT8 / TensorRT variants,
each with its C-MAPSS test metric, the delta vs the PyTorch FP32 reference, and latency.

    # fine-tune once (FD001, full regime) from a pretraining checkpoint, then export + benchmark
    python scripts/deploy.py --tag s20m --ckpt runs/s20m_main/checkpoints/step_00020000
    # reuse an already fine-tuned model; add TensorRT variants on a GPU
    python scripts/deploy.py --tag s20m --regressor exports/s20m/rul.pt --gpu

Variants (CPU): torch_fp32 (reference), ort_fp32, ort_int8_dynamic, ort_int8_static (512
calibration windows from *training* units), ort_int8_static_keep (static, with the embedding,
attention-score matmuls and head left in FP32). Static INT8 quantizes MatMul/Gemm only. GPU (``--gpu``): torch_cuda_fp32, trt_fp16, trt_int8,
trt_int8_keep (same ops pinned to FP16). Output: ``results/bench/deploy_<tag>_{cpu,gpu}.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

from sensorfm.downstream import cmapss
from sensorfm.eval.metrics import nasa_score, rmse
from sensorfm.export.bench import time_fn, time_session
from sensorfm.export.calibrate import sample_calibration
from sensorfm.export.onnx_export import export_onnx
from sensorfm.export.ort_quant import model_size_mb, quantize_dynamic_int8, quantize_static_int8
from sensorfm.export.verify import ort_session
from sensorfm.finetune.common import FitCfg, load_regressor
from sensorfm.finetune.rul import run_rul
from sensorfm.model.sensorfm import ModelConfig
from sensorfm.utils.hardware import hardware_info

# Layers kept in higher precision in the *_keep variants, matched as substrings of ONNX node /
# TensorRT layer names (the exporter derives them from module paths): the patch embedding (sees
# RevIN output directly), the activation x activation attention-score matmuls, and the head.
# RevIN arithmetic itself carries no module scope in the export (it is a method call, not a
# forward), which is why static INT8 quantizes only MatMul/Gemm in the first place.
KEEP = ("/embed/", "/attn/MatMul", "/head/")
BATCHES = (1, 32)


def keep_nodes(onnx_path: Path) -> list[str]:
    import onnx

    return [n.name for n in onnx.load(str(onnx_path)).graph.node if any(k in n.name for k in KEEP)]


def score(raw: np.ndarray, y: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, float]]:
    pred = np.clip(raw.reshape(-1) * scale, 0, cmapss.RUL_CAP)
    return pred, {"rmse": rmse(pred, y), "nasa": nasa_score(pred, y)}


def batch_inputs(x: np.ndarray, m: np.ndarray, b: int) -> dict[str, np.ndarray]:
    idx = np.arange(b) % len(x)  # real test windows, tiled if the batch is larger than the set
    return {"x": np.ascontiguousarray(x[idx]), "mask": np.ascontiguousarray(m[idx])}


def main(
    tag: str = typer.Option(...),
    ckpt: Path | None = None,
    model: str = "s20m",
    regressor: Path | None = typer.Option(None, help="skip fine-tuning, use this rul.pt"),
    subset: str = "FD001",
    root: Path = Path("data/downstream/cmapss"),
    seed: int = 0,
    epochs: int = 40,
    window: int = 64,
    n_calib: int = 512,
    iters: int = 200,
    threads: str = "1,0",
    gpu: bool = False,
    export_dir: Path = Path("exports"),
    out_dir: Path = Path("results/bench"),
) -> None:
    edir = export_dir / tag
    finetune: dict[str, Any] | None = None
    if regressor is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mcfg = ModelConfig(**yaml.safe_load(Path(f"configs/model/{model}.yaml").read_text()),
                           context_length=512)  # fmt: skip
        regressor = edir / "rul.pt"
        r = run_rul(root, subset, "full", seed, mcfg, ckpt, window, 0.0, dev,
                    FitCfg(epochs=epochs), save_to=regressor)  # fmt: skip
        finetune = {k: v for k, v in r.items() if k not in ("pred", "true")}
        print("fine-tuned:", json.dumps(finetune), flush=True)
    net, meta = load_regressor(regressor)
    scale = float(meta["target_scale"])
    data = cmapss.load_subset(root, meta["subset"])
    xte, mte, yte = cmapss.test_windows(data, int(meta["window"]))
    tr_u, _ = cmapss.split_units(data.train["unit"], 0.2, seed)
    xtr, mtr, _ = cmapss.train_windows(data, tr_u, int(meta["window"]))
    calib = sample_calibration({"x": xtr, "mask": mtr}, n_calib, seed)
    test = {"x": xte, "mask": mte}

    with torch.no_grad():
        ref_raw = net(torch.as_tensor(xte), torch.as_tensor(mte)).numpy()
    ref_pred, ref_metric = score(ref_raw, yte, scale)

    def row(name: str, raw: np.ndarray, size_mb: float, lat: dict[str, Any], **extra: Any) -> dict:
        pred, met = score(raw, yte, scale)
        r = {"variant": name, "size_mb": round(size_mb, 3), **met,
             "d_rmse": met["rmse"] - ref_metric["rmse"], "d_nasa": met["nasa"] - ref_metric["nasa"],
             "max_abs_pred_diff_cycles": float(np.abs(pred - ref_pred).max()),
             "latency": lat, **extra}  # fmt: skip
        print(json.dumps({k: v for k, v in r.items() if k != "latency"}), flush=True)
        return r

    torch_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / 1e6
    th_list = [int(t) for t in threads.split(",")]
    cpu_rows = []

    def torch_lat(module: Any, device: str) -> dict[str, Any]:
        def fn(inp: dict[str, torch.Tensor]) -> Any:
            with torch.no_grad():
                out = module(inp["x"], inp["mask"])
            if device == "cuda":
                torch.cuda.synchronize()
            return out

        return {f"b{b}": time_fn(fn, {k: torch.as_tensor(v).to(device)
                                      for k, v in batch_inputs(xte, mte, b).items()}, b, iters)
                for b in BATCHES}  # fmt: skip

    lat: dict[str, Any] = {}
    n_default = torch.get_num_threads()
    for th in th_list:
        torch.set_num_threads(th or n_default)
        lat[f"threads_{th or 'all'}"] = torch_lat(net, "cpu")
    torch.set_num_threads(n_default)
    cpu_rows.append(row("torch_fp32", ref_raw, torch_mb, lat))

    fp32 = export_onnx(net, (torch.as_tensor(xte[:2]), torch.as_tensor(mte[:2])), ["x", "mask"],
                       ["rul"], edir / "rul_fp32.onnx")  # fmt: skip
    keep = keep_nodes(fp32)
    variants = {
        "ort_fp32": fp32,
        "ort_int8_dynamic": quantize_dynamic_int8(fp32, edir / "rul_int8_dyn.onnx"),
        "ort_int8_static": quantize_static_int8(fp32, edir / "rul_int8_static.onnx", calib),
        "ort_int8_static_keep": quantize_static_int8(
            fp32, edir / "rul_int8_static_keep.onnx", calib, exclude=keep
        ),
    }
    for name, path in variants.items():
        raw = ort_session(path).run(None, test)[0]
        lat = {}
        for th in th_list:
            sess = ort_session(path, threads=th)
            lat[f"threads_{th or 'all'}"] = {
                f"b{b}": time_session(sess, batch_inputs(xte, mte, b), b, iters) for b in BATCHES
            }
        extra = {"fp32_nodes": len(keep)} if name.endswith("_keep") else {}
        cpu_rows.append(row(name, raw, model_size_mb(path), lat, path=str(path), **extra))

    common = {"tag": tag, "subset": meta["subset"], "window": meta["window"], "seed": seed,
              "ckpt": str(ckpt) if ckpt else None, "regressor": str(regressor),
              "finetune": finetune, "n_test_units": len(yte), "n_calib": len(calib["x"]),
              "kept_ops": list(KEEP), "reference": "torch_fp32"}  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"deploy_{tag}_cpu.json"
    out.write_text(json.dumps({**common, "device": "cpu", "variants": cpu_rows,
                               "hardware": hardware_info()}, indent=2))  # fmt: skip
    print(f"wrote {out}")

    if not gpu:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu given but CUDA is not available")
    from sensorfm.export.calibrate import make_trt_calibrator
    from sensorfm.export.trt_build import build_engine
    from sensorfm.export.trt_run import TrtRunner

    c, w = xte.shape[1], xte.shape[2]
    shapes = {"x": ((1, c, w), (32, c, w), (256, c, w)), "mask": ((1, w), (32, w), (256, w))}
    net_cuda = net.cuda()
    with torch.no_grad():
        raw = net_cuda(torch.as_tensor(xte).cuda(), torch.as_tensor(mte).cuda()).cpu().numpy()
    gpu_rows = [row("torch_cuda_fp32", raw, torch_mb, {"device": torch_lat(net_cuda, "cuda")})]
    engines = {
        "trt_fp16": build_engine(fp32, edir / "rul_fp16.engine", "fp16", shapes),
        "trt_int8": build_engine(fp32, edir / "rul_int8.engine", "int8", shapes,
                                 make_trt_calibrator(calib, 32, edir / "int8.cache")),
        "trt_int8_keep": build_engine(fp32, edir / "rul_int8_keep.engine", "int8", shapes,
                                      make_trt_calibrator(calib, 32, edir / "int8.cache"),
                                      fp16_layers=KEEP),
    }  # fmt: skip
    for name, path in engines.items():
        runner = TrtRunner(path)
        raw = runner.run(test)
        lat = {"device": {f"b{b}": time_fn(runner.run_device,
                                           runner.to_device(batch_inputs(xte, mte, b)), b, iters)
                          for b in BATCHES}}  # fmt: skip
        gpu_rows.append(row(name, raw, path.stat().st_size / 1e6, lat, path=str(path)))
    out = out_dir / f"deploy_{tag}_gpu.json"
    out.write_text(json.dumps({**common, "device": "cuda", "variants": gpu_rows,
                               "timing": "inputs resident on the GPU; includes sync",
                               "hardware": hardware_info()}, indent=2))  # fmt: skip
    print(f"wrote {out}")


if __name__ == "__main__":
    typer.run(main)
