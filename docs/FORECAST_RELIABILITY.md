# Model 01 — Source-time-safe forecasting reliability adapter

**Status: implementation prototype, no trained TAC checkpoint changed, no original/independent sensor evaluation run.** Original ME111 gap PM2.5/NO2 coverage failure is still binding. This is an interval *postprocessor and diagnostic*, not an improved predictor or production safety guarantee.

## Purpose

The first specialist model is sensor/operational forecasting with explicit, honest uncertainty. This adapter takes an existing frozen model's point/interval predictions and estimates **nonnegative widening corrections** on a *separate* chronological, genuinely source-authorized calibration period. It must never fit an adjustment to the repeatedly inspected ME100–111 2015 DEV cases and subsequently represent a rescore of those same cases as new evidence.

### Contract

Create a JSON contract like:

```json
{
  "schema": "forecast-interval-contract-v1",
  "source_id": "publisher/immutable-source-version",
  "model_sha256": "64 hex original trained checkpoint SHA256",
  "model_fit_end": "2024-12-31T00:00:00Z",
  "calibration_start": "2025-01-01T00:00:00Z",
  "calibration_end": "2025-04-01T00:00:00Z",
  "evaluation_start": "2025-04-02T00:00:00Z",
  "alpha": 0.1,
  "min_cell_n": 30,
  "channels": ["pm25", "no2"],
  "gap_kinds": ["gap", "ordinary"],
  "horizons_hours": [6]
}
```

This example is schematic, **not** the original ME111 real source time split; use only actual lawful, unexposed original data and freeze source and model identities before calibration.

Each JSONL calibration or evaluation row must contain exactly `row_id`, `entity_id`, `issued_at`, `outcome_available_at`, `model_sha256`, `channel`, `horizon_hours`, `gap_kind`, `point`, `lower`, `upper`, `truth`. Source timestamps must be UTC; the true value must have been available within the calibration period before that period ends. Duplicate cases and unregistered bins fail, and all declared channel×horizon×missingness cells must have sufficient calibration observations.

The `infer` request excludes `truth` and `outcome_available_at`, and source-time checks fail closed if missing. The only inference update is the interval expansion:

[
q_{g,h,c}=operatorname{orderstat}_{lceil(n+1)(1-alpha)ceil}
left(max(ell_i-y_i,y_i-u_i,0)ight),quad
[ell',u']=[ell-q,u+q].
]

This is the ordinary split-conformal nonconformity construction; coverage guarantees rely on exchangeability assumptions **that temporally correlated, changing operational data may violate**. No conditional coverage, future performance, or legal source-provenance proof is inferred from a valid correction. Widening can worsen interval score even if coverage improves.

### Commands

```sh
python specialist_models/forecast_reliability.py calibrate --contract SOURCE_LOCK.json --records CAL_ONLY.jsonl --out artifact.json
python specialist_models/forecast_reliability.py evaluate --artifact artifact.json --records LATER_REPORT.jsonl
python specialist_models/forecast_reliability.py infer --artifact artifact.json --request UNLABELED_REQUEST.json
python -m unittest discover -s tests -p 'test_forecast_reliability.py' -v
```

All commands are standard-library only. No network, protected source retrieval, model fitting, artifact overwrite, deployment or state mutation. The CLI records the calibration input file SHA256 and source-model hash asserted by the operator but **cannot independently authenticate** them.

### Promotion gates

1. Identify a genuinely untouched real calibration and separately untouched later evaluation cohort, entity/site/time/label availability audit, publisher checksum/licence, normalization as-of cutoff and evaluator-owned outcome release. An exposed 2015 sensor cohort cannot become unseen by a new preregistration.
2. Use actual frozen ME111 (or separately authorized next trained model) predictions with exact model identity, no posthoc seed selection. Retain raw point forecast and intervals; publish both before and after metrics for *every* channel, gap stratum, horizon and seed.
3. Report actual separate-cohort point MAE, proper interval score, width, nominal/empirical coverage, clustered by independent site/time where possible and out-of-source transfer. If one critical group remains unsafe/under-covered, complete model promotion still fails.
4. Compare with causal missingness-aware baselines (e.g., GRU-D, persistence, ridge) under fair model-family/parameter/data/training budgets for architecture superiority. No declaration of a winner from this postprocessor alone.
5. Only after a model genuinely wins its required gates, measure physical-device or fast-cloud inference including the cost of the calibration wrapper.

**Fixture CI tests verify schema/timestamps/finite-sample mechanics only**. Current model status: not independently qualified; real external forecast reliability improvement: NOT RUN. See [ME111 draft #442](https://github.com/FPC-effortless/TAC-OSM/pull/442), [native diagnosis #453](https://github.com/FPC-effortless/TAC-OSM/pull/453), [product issue #450](https://github.com/FPC-effortless/TAC-OSM/issues/450) and [company strategy](../research/MASTER_RESEARCH_PLAN.md).
