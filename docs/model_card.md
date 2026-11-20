# Model card — SensorFM (draft)

Fields marked *pending* are filled from results files once the Kaggle runs exist; no number here
is typed by hand.

## Model

- **Architecture:** RevIN → non-overlapping patches (P = 16) with a per-point observation flag →
  linear patch embedding + learned positions → pre-LN transformer (SDPA, GELU MLP 4×) →
  per-patch reconstruction head and monotone quantile head (q = 0.1 … 0.9).
- **Sizes:** s20m / m60m / l150m — exact parameter counts in `results/model/params.json`.
- **Context:** 512 points (32 tokens); shorter inputs use right-aligned positions.
- **Pretraining objective:** masked-patch MSE on hidden context patches + pinball loss on a hidden
  64-point horizon, from one forward pass; α = 0.5 by default.
- **Pretraining data:** selected LOTSA subsets (energy, industrial SCADA, telemetry, transport,
  climate), decontaminated against all downstream data — see the [dataset card](dataset_card.md).
  *Size: pending (manifest).*

## Intended use

Research / portfolio: representation learning and zero-/few-shot adaptation for univariate and
channel-independent multivariate sensor signals at 1 Hz–25 kHz-like sampling rates.

## Not intended for

Safety-critical maintenance decisions without domain validation; financial or other non-sensor
series (excluded from pretraining); multivariate cross-channel reasoning (the encoder is
univariate; channels are combined only in task heads).

## Evaluation

*Pending* — generated tables in the README results block and `docs/reports/`.

## Known limitations

- Absolute level/scale are removed by RevIN; downstream heads receive them as explicit features.
- Forecasting at H = 96 sees 416 context points vs 512 for baselines (context length shared with
  the horizon) — a disadvantage stated with every forecasting result.
- CWRU ball-fault labels lack their physical signature in the data (dataset card); bearing results
  are reported with and without those classes.
- Licenses of individual LOTSA subsets are still being verified (Checkpoint 1).
