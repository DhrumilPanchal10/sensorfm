# ADR 001 — Pretraining corpus selection

**Status:** proposed — awaiting 🛑 HUMAN CHECKPOINT 1 (2026-11-20)

## Verified facts (checked at runtime, 2026-11-20)

- LOTSA lives at `Salesforce/lotsa_data` on the Hugging Face Hub; last modified 2025-01-21;
  repo-level license `apache-2.0`. 6,388 arrow files, **~925 GB total**, 170+ subsets.
- Format: one folder per subset of HF-`datasets` arrow files in IPC **stream** format. Columns:
  `item_id: string, start: timestamp[s], freq: string, target: list<float>` (univariate) or
  `list<list<float>>` (multivariate; e.g. PEMS04, PEMS08, residential_*_power, china_air_quality,
  subseasonal, era5_*, alibaba, borg). Some subsets have `past_feat_dynamic_real` (ignored).
- Reader tested end to end on `bdg-2_bear` (91 series → 91 kept, 1.48M obs after edge trimming)
  and `PEMS04` (307 items × 3 variates → 921 series, 15.6M obs).
- No ETT / ETTh / ETTm, no C-MAPSS, no UCR, no CWRU subsets exist in LOTSA (checked against the
  full subset list), so the planned forecasting benchmarks (ETT family) are not in the corpus by
  name. Window-hash decontamination is still run (it catches renamed copies).

## Decision

Use a **selection** of LOTSA (`configs/data/lotsa_selection.yaml`), restricted to sensor-like
domains, with per-subset caps so that no single source dominates:

| Domain | Subsets (abridged) | Why |
|---|---|---|
| energy | buildings_900k (2 files), largest_2017, residential load/PV, London smart meters, BDG-2, IDEAL, GEFCom load | Real metered physical signals with strong daily/weekly structure |
| industrial | wind_farms, kdd2022 (wind-turbine SCADA), solar_power, wind_power | Closest public analogue to industrial SCADA |
| telemetry | azure_vm_traces_2017 (1 file), alibaba_cluster_trace_2018 (1 file) | Machine telemetry: bursty, non-seasonal — diversity |
| transport | PEMS03/04/07/08, PEMS_BAY, LOS_LOOP, LOOP_SEATTLE, traffic_hourly, Q-TRAFFIC, pedestrian counts | Physical loop-detector sensors |
| climate | weather, temperature_rain, oikolab, air quality, subseasonal, era5_2018 (2 files) | Natural sensor signals, many frequencies |

**Excluded:** sales/retail (favorita, m5, hierarchical_sales, car_parts, restaurant), finance
(bitcoin, fred_md), web traffic (wiki, kaggle_web_traffic, extended_web_traffic), epidemiology
(covid, cdc_fluview, project_tycho, hospital), demographic (us_births), mixed-domain competition
sets (M1/M3/M4, tourism, cif_2016, nn5), human-demand counts (taxi, uber, rideshare, metro
ridership), CMIP6 (simulated climate model output; also 12.7 GB per year), borg (900 MB per file).

**Caps:** `default_max_obs` 4M per subset; larger caps (6–12M) for the big high-quality sets.
Caps take the **first** N items of a subset's file order (deterministic, but not a random sample —
documented limitation). Expected corpus size ≈ 100–150M observations, which is ~0.5 GB as float32,
small enough for Kaggle RAM and a private Kaggle Dataset. Grow it after Task 3.1 measures
throughput (plan §1.3).

**Cleaning** (`data/filter.py`): trim NaN edges; drop > 20% missing; linear-impute gaps ≤ 32
points; split at longer gaps; drop segments < 128 points and constant segments. Drop counts per
subset are written to the manifest.

**Split:** 1% of series held out for validation by stable hash of `dataset/item_id`.

## Licenses — ⚠️ NOT YET VERIFIED PER SUBSET

The LOTSA repo is tagged Apache-2.0, but the underlying datasets carry their own terms (the Moirai
paper's appendix lists sources). Each still needs checking at Checkpoint 1. Best current knowledge,
**unverified**:

| Subset | Believed upstream / license | Verified? |
|---|---|---|
| buildings_900k | NREL BuildingsBench (EULP), CC-BY-4.0 | no |
| bdg-2_* | Building Data Genome 2, CC-BY-4.0 | no |
| lcl / london_smart_meters | UK Power Networks via London Datastore | no |
| PEMS*, traffic_hourly | Caltrans PeMS (public) | no |
| era5_2018 | Copernicus C3S licence | no |
| kdd2022 | Baidu KDD Cup 2022 wind data (competition terms) | no |
| azure_vm_traces_2017 | Azure Public Dataset | no |
| alibaba_cluster_trace_2018 | Alibaba clusterdata | no |
| others | see Moirai paper appendix | no |

This is a non-commercial portfolio project; any subset whose terms turn out to forbid research
use will be removed before the real build.

## Open questions for the owner

1. Approve the domain inclusion/exclusion above (especially telemetry and CMIP6 exclusion).
2. Approve ~100–150M observations as the first corpus size.
3. OK to drop any subset whose license can't be confirmed quickly?
