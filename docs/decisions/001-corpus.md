# ADR 001 — Pretraining corpus selection

**Status:** accepted — Checkpoint 1 approved by the owner 2026-11-20 (defaults: keep `largest_2017` (CC BY-NC) and the code-license-only subsets; non-commercial portfolio use).

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

## Licenses — checked at the primary source (2026-11-20)

Method: Zenodo records API (license field), GitHub repo license + README data terms, OEDI dataset
metadata, and the Moirai paper's collection tables to map each LOTSA subset to its source.

| Status | Subsets | Evidence |
|---|---|---|
| ✅ CC BY 4.0 | weather, temperature_rain, pedestrian_counts, australian_electricity_demand, kdd_cup_2018, saugeenday, solar_power, wind_power, wind_farms, london_smart_meters, traffic_hourly, oikolab_weather, elecdemand (Monash archive) | Zenodo records 4654822, 5129073, 4656626, 4659727, 4656719, 4656058, 4656027, 4656032, 4654909, 4656091, 4656132, 5184708, 4656069 |
| ✅ CC BY 4.0 | buildings_900k, bdg-2_*, lcl, smart, ideal, borealis (BuildingsBench) | OEDI submission 5859 metadata (doi 10.25984/1986147) |
| ✅ CC BY 4.0 | residential_load_power, residential_pv_power | Zenodo 8219786 |
| ✅ CC BY 4.0 | azure_vm_traces_2017 | Azure/AzurePublicDataset repo license |
| ⚠️ CC BY-NC 4.0 (non-commercial) | largest_2017 | LargeST README §4 — kept, flagged: fine for this portfolio project, blocks commercial use of the trained model |
| ❌ no license → **removed** | alibaba_cluster_trace_2018, Q-TRAFFIC | alibaba/clusterdata: no license, README "encourage … study or research"; JingqingZ/BaiduTraffic: no license |
| ◻️ code license only, data terms not separately stated | PEMS03/04/07/08, PEMS_BAY, LOS_LOOP, LOOP_SEATTLE (LibCity, Apache-2.0 repo; PeMS is public Caltrans data); gfc12/17, spain, bull, hog (ProEnFo, BSD-3 repo); kdd2022 (PaddleSpatial, Apache-2.0 repo); era5_2018 (ClimateLearn, MIT repo; ERA5 itself is under the Copernicus licence, attribution required); subseasonal* (SubseasonalClimateUSA) | repo licenses via GitHub API |
| ◻️ not confirmed | beijing_air_quality (UCI 501; UCI's API exposes no license field), china_air_quality (Zheng et al. 2015) | — |

This is a non-commercial portfolio project. Attribution for all CC BY sources goes in the
dataset card and README credits before any public release.

## Open questions for the owner

1. Approve the domain inclusion/exclusion above (especially telemetry and CMIP6 exclusion).
2. Approve ~100–150M observations as the first corpus size.
3. OK to drop any subset whose license can't be confirmed quickly?
