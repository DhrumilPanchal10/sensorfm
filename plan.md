# SensorFM — Implementation Plan (for Claude Code)

> **Project:** Pretrain a compact (~20M / ~60M / ~150M parameter) patch-based transformer foundation model for sensor time series, engineer the training run for throughput, scaling, and fault tolerance (bf16, torch.compile, DDP/FSDP, profiling), adapt it to industrial tasks (remaining-useful-life prediction, anomaly detection, forecasting), and deploy INT8 models (TensorRT on GPU, ONNX Runtime on CPU) behind a streaming scoring service.
>
> **Owner:** Dhrumil Panchal. **Implementer:** Claude Code, phase by phase under the owner's supervision.
>
> **Core claim this repo must be able to defend:** "I pretrained a time-series foundation model myself, made the training run measurably efficient and robust across multiple GPUs, compared it honestly against much larger public foundation models, and deployed it under edge constraints with measured quality loss."
>
> **Independence:** This project uses only public data and public code. It must not use any PRISM Lab code, data, or simulators.

---

## 0. Operating rules for Claude Code (read first, follow always)

1. **Phase by phase, task by task.** Pass acceptance criteria before moving on. After each task: tests, lint, commit `phaseN.taskM: <summary>`, append to `PROGRESS.md`.
2. **Never fabricate results.** All numbers come from scripts → `results/*.json` → generated reports/README (`scripts/build_report.py`).
3. **Targets are not results.** Targets only in `docs/targets.md`.
4. **Verify external facts at runtime.** Dataset locations/formats (LOTSA, C-MAPSS, UCR Anomaly Archive, bearing data), library APIs (FSDP2 vs FSDP1, torch.compile, TensorRT, ONNX Runtime quantization, Chronos/Moirai/TTM packages) change. Inspect versions/docs/samples first; record in `docs/decisions/`. Follow reality over this plan and document deviations.
5. **Stop at every `🛑 HUMAN CHECKPOINT`.** Never rent GPUs or spend money without approval. Every multi-GPU job must print an estimated cost (hours × `GPU_PRICE_PER_HOUR` × n_gpus) before launch.
6. **Reproducibility.** Hydra configs; MLflow/W&B logging of config, git hash, data manifest hash, seed, hardware; outputs to `runs/<run_id>/`.
7. **Debug first.** Every training path runs on CPU or 1 GPU with a tiny model and synthetic data in under 2 minutes (CI uses this).
8. **Measure before optimizing.** Any performance change is justified by a profiler trace before and after, saved under `results/profiles/`.
9. **Code quality.** Python 3.11, types, `ruff`, `mypy --strict` on `src/`, `pytest`.
10. **Explainability.** Module docstrings explain *why*; ADRs for non-obvious decisions. The owner must be able to explain FSDP sharding, MFU math, and every profiler finding.

---

## 1. Environment and compute

- Python 3.11, `uv`. Deps: `torch` (recent, with FSDP2 `fully_shard` if available), `numpy`, `pandas`, `pyarrow`, `datasets`, `huggingface_hub`, `hydra-core`, `mlflow` and/or `wandb`, `einops`, `scipy`, `scikit-learn`, `statsmodels` (ARIMA/ETS), `onnx`, `onnxruntime`, `tensorrt` + `polygraphy` (GPU env only), `fastapi`, `uvicorn`, `prometheus-client`, `typer`, `rich`, `matplotlib`. Optional: `triton` (bundled with torch), `chronos-forecasting`, `uni2ts` (Moirai), `granite-tsfm` (TTM) for baselines.
- Single-GPU dev: one ≥24 GB GPU. Scaling runs: rented node with 4 GPUs of the same type (owner approves at checkpoint).
- `deploy/Dockerfile.train` (CUDA + torch), `deploy/Dockerfile.serve` (CPU ORT), `deploy/Dockerfile.trt` (TensorRT).

`.env.example`:
```
HF_TOKEN=
WANDB_API_KEY=
MLFLOW_TRACKING_URI=file:./mlruns
GPU_PRICE_PER_HOUR=
DATA_ROOT=./data
```

---

## 2. Repository structure

```
sensorfm/
├── README.md
├── CLAUDE.md
├── PROGRESS.md
├── Makefile
├── pyproject.toml
├── .github/workflows/ci.yml
├── configs/
│   ├── config.yaml
│   ├── data/{synthetic_debug,pretrain_main}.yaml
│   ├── model/{tiny,s20m,m60m,l150m}.yaml
│   ├── train/{single_gpu,ddp,fsdp}.yaml
│   ├── finetune/{cmapss_rul,ucr_anomaly,forecast}.yaml
│   └── export/{onnx_fp32,ort_int8,trt_fp16,trt_int8}.yaml
├── data/                 # raw/, shards/, downstream/ (DVC)
├── docs/{targets.md,dataset_card.md,model_card.md,decisions/,reports/}
├── results/{eval,profiles,bench,scaling}/
├── runs/
├── scripts/
├── src/sensorfm/
│   ├── utils/{seed.py,dist.py,hardware.py,flops.py,timing.py,io.py}
│   ├── data/{lotsa.py,synthetic_physics.py,filter.py,shard.py,loader.py,normalize.py,dedup.py}
│   ├── downstream/{cmapss.py,ucr_anomaly.py,bearing.py,forecast_bench.py}
│   ├── model/{patching.py,revin.py,transformer.py,heads.py,sensorfm.py}
│   ├── pretrain/{objectives.py,train.py,checkpoint.py,schedule.py}
│   ├── kernels/{fused_op.py}            # only if profiling justifies (Task 3.4)
│   ├── finetune/{rul.py,anomaly.py,forecast.py,linear_probe.py}
│   ├── baselines/{naive.py,stats_models.py,patchtst_scratch.py,foundation_zero_shot.py}
│   ├── eval/{metrics.py,bootstrap.py,slices.py}
│   ├── export/{onnx_export.py,ort_quant.py,trt_build.py,calibrate.py,verify.py}
│   ├── serving/{app.py,stream_consumer.py,windowing.py,metrics.py}
│   └── monitoring/{drift.py}
├── bench/
├── deploy/
└── tests/{unit,integration,fixtures}
```

---

## PHASE 1 — Data

### Task 1.1 — Scaffold
- Structure, Makefile (`setup lint test data-debug data-pretrain data-downstream pretrain-debug pretrain finetune eval profile scale export bench serve report`), CI (lint, mypy, pytest, `pretrain-debug` on CPU), `CLAUDE.md`, `PROGRESS.md`, `docs/targets.md` (Section 9).

### Task 1.2 — Synthetic physics generator (`data/synthetic_physics.py`)
- Public-equation simulators implemented from scratch: damped harmonic oscillators, coupled mass-spring-damper chains (n DOF, random stiffness/damping, random forcing: impulse, white noise, sinusoid sweep), rotating-machinery vibration (shaft frequency harmonics + bearing defect frequencies + noise), slow degradation processes (gradually changing stiffness/frequency). Integrate with `scipy.integrate.solve_ivp` or exact discrete-time state-space updates. Randomize sampling rates and sensor noise (SNR 10–40 dB).
- Use: (a) debug/CI data, (b) optional synthetic mix in pretraining (ablation E5).
- Tests: energy decays for damped free vibration; FFT peak at the configured natural frequency within tolerance.

### Task 1.3 — Pretraining corpus (`data/lotsa.py`, `data/filter.py`, `data/shard.py`)
- LOTSA (Salesforce; the Moirai pretraining collection) is on Hugging Face (as of planning: `Salesforce/lotsa_data`) as many sub-datasets. **Verify** the list, sizes, domains, frequencies, and licenses. Select sub-datasets from energy, transport, industrial/IoT, nature/climate sensors; exclude finance/sales-like series unless needed for volume. Record the selection with reasons and per-dataset licenses in `docs/decisions/001-corpus.md`.
- Normalize into univariate float32 series with metadata (dataset, frequency, length). Drop series shorter than the minimum context, constant series, and series with >20% missing; impute short gaps linearly; record drop counts.
- **Decontamination (`data/dedup.py`):** remove any series overlapping downstream datasets (by source name and by hashing of normalized windows).
- Write streaming shards (`data/shard.py`): fixed-size shards (e.g. 256 MB) of pre-windowed or raw series in Arrow/NumPy memmap with an index; shuffle at shard + buffer level. Manifest with total observations, series count, per-domain counts, hashes → `data/shards/manifest.json`.
- Target size: billions of observations if storage/time allow; start with ~100M observations, scale up after Task 3.1 measures throughput (choose the final size from the compute budget, then document).

### Task 1.4 — Data loader (`data/loader.py`, `data/normalize.py`)
- `IterableDataset` that streams shards, samples random windows (context length L=512 default, configurable), applies per-window instance normalization stats (RevIN handled in-model; loader returns raw window + mask), supports DDP/FSDP rank/worker sharding without duplication (test it), pinned memory, `num_workers` and prefetch configurable.
- Benchmark: samples/s per worker; the loader must exceed the model's consumption rate (verified in Task 3.1).

### Task 1.5 — Downstream datasets (`downstream/`)
- **C-MAPSS** (NASA turbofan RUL; FD001–FD004). Verify the current official download source. Standard protocol: piecewise-linear RUL cap at 125, per-operating-condition normalization for FD002/FD004, sliding windows; metrics RMSE and NASA score function on official test units (last cycle).
- **UCR Anomaly Archive** (Wu & Keogh's benchmark of 250 univariate series, created because older anomaly benchmarks were shown to be flawed). Verify location/format. Protocol: train on the anomaly-free prefix, score the rest, accuracy = whether the top-scoring location falls within the labeled anomaly region ± tolerance (follow the archive's stated protocol). **Do not use point-adjusted F1.**
- **Bearing vibration** (one public dataset, e.g. CWRU or Paderborn — verify license and access; pick one and document): fault classification from short windows, split by **run/load condition**, not by window, to avoid leakage.
- **Forecasting**: 3–5 standard public sensor-like datasets **not** in the pretraining corpus (e.g. ETT-style electricity transformer data if excluded from pretraining; verify exclusion). Metrics MASE and CRPS (quantile loss averaged over quantiles 0.1…0.9).
- Dataset card `docs/dataset_card.md`.

🛑 **HUMAN CHECKPOINT 1:** Owner reviews corpus selection, decontamination, and downstream protocols.

---

## PHASE 2 — Model and pretraining (single GPU)

### Task 2.1 — Model (`model/`)
- Input: univariate window x ∈ R^L with observation mask. RevIN (instance normalization with stored mean/std). Patching: patch length P=16, stride 16 (configurable) → N=L/P tokens; linear patch embedding + learned positional embeddings.
- Encoder: pre-LN transformer, `torch.nn.functional.scaled_dot_product_attention`, GELU MLP (4×), dropout, configurable depth/width:

| Config | d_model | layers | heads | ≈params |
|---|---|---|---|---|
| tiny (CI) | 64 | 2 | 2 | <1M |
| s20m | 384 | 8 | 6 | ~20M |
| m60m | 512 | 16 | 8 | ~60M |
| l150m | 768 | 18 | 12 | ~150M |

(Compute exact counts in code and report them.)
- Heads: masked-patch reconstruction head (pretraining); forecasting head producing quantiles for horizon H (pretraining aux + fine-tuning); task heads in Phase 4.
- `utils/flops.py`: analytic FLOPs per token for forward+backward (6·params·tokens approximation plus attention term), used for MFU = achieved FLOPs/s ÷ GPU peak bf16 FLOPs/s (peak from a config table per GPU type; state the source).

### Task 2.2 — Objectives (`pretrain/objectives.py`)
- Masked patch modeling: mask ratio r (default 0.4) of patches; MSE on normalized masked patches.
- Forecasting: given the first L−H points, predict the next H with quantile (pinball) loss.
- Total loss = α·masked + (1−α)·forecast (α configurable; ablated).

### Task 2.3 — Training loop (`pretrain/train.py`, `pretrain/checkpoint.py`, `pretrain/schedule.py`)
- AdamW (β2=0.95, weight decay 0.1, no decay on norms/biases/embeddings), linear warmup + cosine decay, gradient clipping 1.0, bf16 autocast, gradient accumulation, `torch.compile` flag.
- Logging every N steps: loss terms, lr, grad norm, tokens/s, samples/s, step time breakdown (data wait vs compute), GPU memory, MFU.
- **Checkpointing:** model, optimizer, scheduler, step, RNG states (python/numpy/torch/cuda), data loader position (shard index + offset). Atomic writes (temp file + rename), keep last K. **Resume test:** an integration test that kills training mid-run (subprocess + SIGKILL) and resumes, asserting the loss curve continues and step count is consistent; measure time-to-recover.
- Validation: held-out shards (by series, not by window), masked-reconstruction loss and forecasting CRPS.
- **Acceptance:** `pretrain-debug` passes in CI; the s20m model trains on 1 GPU with decreasing val loss.

---

## PHASE 3 — Training systems engineering

### Task 3.1 — Baseline throughput and profiling (`make profile`)
- Measure a naive reference configuration: fp32, eager, `num_workers=0`. Then profile with `torch.profiler` (Chrome trace + key averages table) and, if available, Nsight Systems (`nsys profile`). Save traces/summaries under `results/profiles/<config>/`.
- Identify the top bottlenecks (data wait %, kernel time breakdown, CPU overhead, memory copies).

### Task 3.2 — Optimization ladder (each step measured: tokens/s, MFU, peak memory, val loss parity)
1. Data loader: workers, pinned memory, prefetch, shard caching.
2. bf16 autocast.
3. `torch.compile` (mode configurable; record compile time).
4. Fused AdamW (`fused=True`) if supported.
5. Larger micro-batch enabled by memory savings; activation checkpointing only if memory-bound.
6. Check whether loss/val metrics stay within noise of the reference after optimizations (parity test on a fixed seed short run).
- Output `results/profiles/ladder.json` + `docs/reports/training_efficiency.md` (generated table and a bar chart of tokens/s per step of the ladder).

### Task 3.3 — Distributed training (`utils/dist.py`, `configs/train/{ddp,fsdp}.yaml`)
- `torchrun` launch; DDP wrapper; FSDP (prefer FSDP2 `fully_shard` if available in the installed torch; else FSDP1 with `ShardingStrategy.FULL_SHARD`), mixed-precision policy (bf16 params/compute, fp32 reduce if needed), sharded/distributed checkpoints (`torch.distributed.checkpoint`) with resume across the same world size.
- Correctness test: 2-process CPU (gloo) run on `tiny` matching single-process loss within tolerance for the first N steps with the same global batch.
- **Scaling experiment** on the rented 4-GPU node: 1, 2, 4 GPUs × {DDP, FSDP} × model sizes {m60m, l150m}; fixed per-GPU micro-batch; report tokens/s, scaling efficiency = tokens/s(n) ÷ (n × tokens/s(1)), memory per GPU, MFU. Explain honestly where DDP is sufficient at this scale and where FSDP's memory savings matter. Output `results/scaling/*.json`, `docs/reports/scaling.md`.

🛑 **HUMAN CHECKPOINT 2:** Before renting the multi-GPU node: print the planned run list with estimated hours and cost; owner approves.

### Task 3.4 — Custom kernel (conditional)
- Only if Task 3.1/3.2 traces show a fusible hotspot that `torch.compile` does not already fuse (e.g. RevIN + patch-embedding normalization, or masked loss reduction). Implement a Triton kernel in `kernels/fused_op.py` with: numerical tests vs PyTorch reference (forward and backward if used in training), microbenchmark, and end-to-end tokens/s impact. If no such hotspot exists, write an ADR explaining why no custom kernel was written — that is a valid, defensible outcome.

### Task 3.5 — Main pretraining runs
- Train s20m, m60m, l150m with a fixed token budget each (chosen from measured throughput and the approved budget). Log everything. Produce a small scaling plot: val loss vs params (and vs tokens if checkpoints allow).

---

## PHASE 4 — Downstream adaptation and evaluation

### Task 4.1 — Baselines (`baselines/`)
- Seasonal naive and ARIMA/ETS (`statsmodels`) for forecasting; PatchTST trained from scratch per task (same architecture family, random init — isolates the pretraining effect); zero-shot public foundation models where packages are available (Chronos-2/Chronos-Bolt, Moirai, TTM — verify current package names/APIs; skip any that won't install and document it).
- For C-MAPSS: also a gradient-boosting model on hand-crafted window features, and a from-scratch CNN/LSTM per common practice.

### Task 4.2 — Fine-tuning protocols (`finetune/`)
- Three regimes per task: linear probe (frozen encoder), full fine-tuning, and few-shot (10% of training data) to show the value of pretraining under data scarcity.
- RUL: windows → encoder → pooled embedding → regression head; early stopping on validation units (split by engine unit, never by window).
- Anomaly: score = forecasting or reconstruction error with a rolling normalization; evaluate per the UCR protocol.
- Bearing: classification head; split by run/condition.
- Forecasting: zero-shot and fine-tuned.

### Task 4.3 — Evaluation (`eval/`)
- Metrics: MASE, CRPS (forecast); RMSE, NASA score (RUL); UCR accuracy (anomaly); macro-F1 (bearing). 3 seeds for fine-tuning; mean ± std; paired bootstrap over series/units for claimed differences.
- Slices: domain, sampling frequency, horizon, operating condition (C-MAPSS FD00x).
- Output `results/eval/*.json`; generated `docs/reports/downstream.md` with a size-vs-quality plot including public foundation models (x = parameters, log scale; y = metric).

### Task 4.4 — Ablations

| ID | Variable |
|---|---|
| E1 | Pretrained vs from-scratch (same architecture) |
| E2 | Model size s20m / m60m / l150m |
| E3 | Patch length 8 / 16 / 32 |
| E4 | Mask ratio 0.2 / 0.4 / 0.6 |
| E5 | Synthetic physics data in pretraining: off / on (10–20% mix) |
| E6 | Objective mix α ∈ {0, 0.5, 1} |
| E7 | Fine-tune regime: linear probe / full / few-shot |

### Task 4.5 — Error analysis
- RUL: error vs true RUL bucket (early-life vs late-life), per operating condition; anomaly: failure categories (missed subtle anomalies, false alarms at regime changes). Write findings and one targeted improvement with before/after.

---

## PHASE 5 — Deployment under edge constraints

### Task 5.1 — Export (`export/`)
- Export fine-tuned models (RUL and anomaly/forecast) to ONNX (fixed and dynamic batch); verify ONNX vs PyTorch outputs (tolerance test).
- CPU path: ONNX Runtime FP32 → dynamic and static INT8 quantization (static calibration with 512 representative windows).
- GPU path: TensorRT FP16 and INT8 engines (`trt_build.py`, entropy/percentile calibrator in `calibrate.py`). Verify current TensorRT Python APIs.
- Keep sensitive ops (normalization, final head) in higher precision if INT8 hurts regression quality; document per-layer precision choices.

### Task 5.2 — Benchmarks
- For each variant: latency p50/p99 at batch 1 and batch 32, throughput, memory/engine size, and task metric delta vs FP32 PyTorch. Hardware specs logged. Output `results/bench/*.json` and `docs/reports/deployment.md` (quality-vs-latency Pareto plot).

### Task 5.3 — Streaming scoring service (`serving/`)
- `stream_consumer.py`: consumes sensor readings (simulated producer replaying a held-out C-MAPSS unit and UCR series; Kafka/Redpanda via Compose, or an in-process async queue in debug mode), maintains per-sensor sliding windows (`windowing.py`), scores with the INT8 ORT model, emits anomaly/RUL alerts with thresholds.
- FastAPI `app.py`: `POST /score` (batch windows), `GET /healthz`, `GET /metrics` (latency, windows scored, alerts, model version).
- Drift (`monitoring/drift.py`): PSI on window mean/std/spectral-energy features vs training reference; gauge + report; demonstrate it fires when switching operating condition.
- Tests: windowing correctness, end-to-end replay on fixture.

---

## PHASE 6 — Documentation and polish

- `scripts/build_report.py` generates all reports and README tables from `results/`.
- README first screen: problem, Mermaid diagram (corpus → pretraining → adaptation → export → streaming service), headline results (generated), key decisions, "what I'd do next".
- `docs/model_card.md`; `docs/reports/technical_writeup.md` (1,500–2,500 words) including the training-efficiency story with profiler screenshots.
- Fresh-clone check: `make setup && make test && make pretrain-debug` passes.

---

## 8. Definition of done per stage

| Stage | Done when |
|---|---|
| Repository foundation | Phase 1 (debug data + loader), Task 2.1–2.3 debug |
| Resume-ready | s20m pretrained on 1 GPU; Task 3.1–3.2 ladder; C-MAPSS fine-tune vs from-scratch PatchTST with seeds; README v1 |
| Interview-ready | Task 3.3 scaling study, Task 3.5 sizes, Phase 4 complete |
| Full | Phases 5–6 complete |

## 9. Targets (copy to `docs/targets.md`; NOT results)

| Metric | Target |
|---|---|
| Throughput vs naive fp32 eager | ≥ 3× (MFU reported) |
| Scaling efficiency at 4 GPUs | ≥ 85% |
| Downstream vs from-scratch PatchTST | Better on RUL and anomaly tasks, especially few-shot |
| vs Chronos-2 zero-shot | Competitive MASE at a fraction of its parameters |
| INT8 deployment | ≥ 3× lower latency, ≤ 2% metric degradation |

## 10. Resume bullet → evidence map (`docs/resume_evidence.md`)

| Claim | Evidence |
|---|---|
| Pretraining scale (params, observations) | `data/shards/manifest.json`, model param counts in run logs |
| Throughput gain and MFU | `results/profiles/ladder.json`, traces |
| Scaling efficiency (DDP/FSDP) | `results/scaling/*.json` |
| Crash-tested resume | resume integration test + log |
| Downstream comparisons and ablations | `results/eval/*.json`, `docs/reports/downstream.md` |
| INT8 latency/quality | `results/bench/*.json`, `docs/reports/deployment.md` |
