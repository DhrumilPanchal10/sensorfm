# Running SensorFM on Kaggle (free tier)

Kaggle Notebooks give free CPU sessions and a weekly GPU quota (T4 ×2 or P100). Check the current
quota in your Kaggle settings; it changes. Nothing here costs money.

## One-time setup

1. Push this repo to GitHub (private is fine). If private, create a fine-grained read-only token
   and add it as a Kaggle **Secret** named `GH_TOKEN` (Add-ons → Secrets).
2. Optional: add `HF_TOKEN` (read-only Hugging Face token) as a Secret. It raises Hub rate limits
   for the LOTSA download.

## Step 1 — build data (CPU session, Internet ON)

New notebook → Settings: Accelerator **None**, Internet **On**. One cell:

```bash
!git clone https://$GH_TOKEN@github.com/<you>/sensorfm.git && cd sensorfm && bash kaggle/build_data.sh
```

(For a public repo drop `$GH_TOKEN@`. Read secrets in a Python cell first:
`from kaggle_secrets import UserSecretsClient; import os;
os.environ["GH_TOKEN"]=UserSecretsClient().get_secret("GH_TOKEN")`.)

Outputs land in `/kaggle/working/`: `downstream/`, `shards/main/` (with `manifest.json`),
`results/`. Click **Save Version → Save & Run All**, then from the notebook output choose
**New Dataset** → private dataset `sensorfm-data`. Later GPU notebooks attach it as input.

Disk: Kaggle gives ~20 GB in `/kaggle/working` (persisted) plus temporary space. The build
downloads at most ~12 GB of arrow files into the temp HF cache (`/kaggle/tmp`) and writes only
the ~0.5–1 GB of shards to `/kaggle/working`.

## Step 2 — training (GPU sessions) — added in Phase 2/3

GPU notebook scripts live next to this file once Phase 2 lands. Every GPU job prints its estimated
GPU-hours against the weekly quota before it starts (cost is $0).
