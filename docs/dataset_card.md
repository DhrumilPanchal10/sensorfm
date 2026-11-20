# Dataset card

Status: **draft for 🛑 Checkpoint 1.** Numbers marked *(from manifest)* get filled by the real
build on Kaggle; nothing here is a training result.

## 1. Pretraining corpus

- **Source:** selected subsets of LOTSA (`Salesforce/lotsa_data`, Hugging Face). Selection,
  rationale and license status: [decisions/001-corpus.md](decisions/001-corpus.md); config:
  `configs/data/lotsa_selection.yaml`.
- **Domains:** energy, industrial/renewables SCADA, machine telemetry, transport sensors,
  climate/nature sensors. Sales, finance, web, epidemiology and mixed competition sets excluded.
- **Unit of data:** univariate float32 series (multivariate subsets split per variate).
- **Cleaning:** NaN-edge trim; drop > 20% missing; linear impute gaps ≤ 32; split at longer gaps;
  drop segments < 128 points and constant segments. Per-subset drop counts *(from manifest)*.
- **Decontamination:** name blocklist + z-normalized 64-point window hashing against every
  downstream series (30,054 reference series, ~18.7M window hashes). Low-information windows
  (< 16 distinct quantized values) are not hashed: they produced false matches on real PEMS04 /
  BDG-2 data before this rule was added (16 false hits on 1,012 series → 0 after; 50/50
  planted copies still detected). Limitation: heavily quantized copies (e.g. integer counts with
  few distinct values) may evade detection; downstream data here is float-valued.
- **Split:** 1% of series held out for validation by stable hash of `dataset/item_id`.
- **Size:** target ≈ 100–150M observations *(from manifest)*.
- **Synthetic physics (optional, ablation E5):** `data/synthetic_physics.py` — SDOF oscillators,
  n-DOF chains, rotating machinery with bearing defects, degradation; fs log-uniform 1 Hz–25.6 kHz,
  SNR 10–40 dB.

## 2. Downstream datasets

All sources verified live on 2026-11-20; loaders checked on the real files.

### C-MAPSS (RUL) — `downstream/cmapss.py`
- NASA PCoE turbofan simulation, FD001–FD004 (100/260/100/249 train units; readme says 248 for
  FD004, the file has 249).
- Piecewise-linear RUL capped at 125 on training; official last-cycle RUL on test (uncapped by
  default; capped variant also reportable). 6 operating conditions (FD002/FD004) found by rounding
  the settings — exactly 6 clusters on the real data; per-condition z-normalization with
  train-only statistics. Data-driven constant-sensor removal keeps 15/17/16/17 sensors (literature
  often uses 14 on FD001; difference is s6, std ≈ 0.001).
- Validation split by engine unit. Metrics: RMSE, NASA score.

### UCR Anomaly Archive — `downstream/ucr_anomaly.py`
- 250 series (6,684–900,000 points; 19.4M total). One anomaly per series, anomaly-free training
  prefix encoded in the file name.
- Scoring exactly as in the archive's own slides: predicted location P is correct iff
  `min(begin−L, begin−100) < P < max(end+L, end+100)`, L = anomaly length. Metric: accuracy over
  250. **No point-adjusted F1.**

### CWRU bearings — `downstream/bearing.py`
- 12 kHz drive-end, 10 classes (normal + IR/ball/OR@6 × 0.007/0.014/0.021"), loads 0–3 hp,
  40 recordings. Normal-baseline files are 48 kHz (verified: shaft 1× appears at the right
  frequency only under that assumption) and are decimated to 12 kHz.
- **Split by load**, never by window (window-level splits leak and give ~100% accuracy).
- **Label physics check** (`results/eval/cwru_label_check.json`): RPM matches nominal load for
  40/40 files; envelope-spectrum peak matches the labeled defect frequency for 4/4 normal (1×),
  11/12 inner race, 8/12 outer race, **0/12 ball**. Ball faults (and the 0.014" outer-race set)
  don't show their textbook signature — consistent with CWRU's known reputation for weakly
  diagnosable ball faults. We keep the published labels and report this caveat with any
  bearing result. Metric: macro-F1.

### Forecasting — `downstream/forecast_bench.py`
- ETTh1, ETTh2, ETTm1, ETTm2 (CC BY-ND 4.0). Not in LOTSA (checked against the full subset list).
- Standard 12/4/4-month split; each of 7 columns univariate; rolling origins in the test span.
- Metrics: MASE (seasonal-naive scale, m = 24 hourly / 96 for 15-min) and CRPS ≈ 2 × mean pinball
  loss over q = 0.1…0.9 (plus scale-free WQL).
- Only one dataset family so far (plan asks 3–5 datasets): a 5th non-LOTSA set is an open
  question for Checkpoint 1.
