# Running SensorFM on Kaggle (free tier)

Kaggle Notebooks give free CPU sessions and a weekly GPU quota (T4 ×2 or P100). Check the current
quota in your Kaggle settings; it changes. Nothing here costs money.

## One-time setup

1. Push this repo to GitHub (private is fine). If private, create a fine-grained read-only token
   and add it as a Kaggle **Secret** named `GH_TOKEN` (Add-ons → Secrets).
2. Optional: add `HF_TOKEN` (read-only Hugging Face token) as a Secret. It raises Hub rate limits
   for the LOTSA download.

## Step 1 — build data (CPU session, Internet ON) — notebook provided

1. Kaggle → **Create → New Notebook → File → Import Notebook** → upload
   `kaggle/notebooks/01_build_data.ipynb` from this repo.
2. Right panel: **Accelerator: None**, **Internet: On**; **Add-ons → Secrets**: `GH_TOKEN`
   (and optionally `HF_TOKEN`), each toggled on for this notebook.
3. **Run All**. Expect ~1–2 h (downloads ~5 GB of LOTSA arrow files to `/tmp`, writes ~0.5–1 GB
   of shards). The last cell prints the corpus summary and manifest hash.
4. **Save Version → Save & Run All (Commit)**. When it finishes: notebook **Output** tab →
   **New Dataset** → name `sensorfm-data`, **Private**.

Validated beforehand on 5 real subsets locally (multivariate + covariate formats, decontamination
against the full downstream reference, training on the resulting shards).

## Step 2 — training (GPU sessions) — added in Phase 2/3

GPU notebook scripts live next to this file once Phase 2 lands. Every GPU job prints its estimated
GPU-hours against the weekly quota before it starts (cost is $0).
