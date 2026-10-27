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


## First original-source Model 01 baseline: UCI374 (2026-10-27)

**Reproducible genuine data + exposed DEV evidence; NOT a trained TAC-OSM improvement.**
[GitHub Actions source and score run #40466514610](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40466514610) downloaded the original [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) archive (CC BY 4.0, DOI 10.24432/C5VC8G).
Original ZIP SHA256: 2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a. Compressed 11,979,507 bytes.

The baseline uses original date and Appliances fields only for a one-hour-ahead forecast. Candidate methods: last observation and same time previous day. Selection occurred on earlier CAL observations, not DEV. Row indexes: unused FIT [0,9867), exposed CAL [9867,13814), exposed DEV [13814,15788). The protected tail [15788,19735) was not parsed or used for scoring or model selection, although original compressed ZIP bytes necessarily were downloaded.

| Baseline | Real DEV cases | MAE (Wh) | Observed nominal 90% coverage | Width (Wh) | Interval score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Last observation (CAL-selected) | 1,968 | 50.9400 | 90.9045% | 360.0 | 620.7724 |
| Same time previous day (unselected reference) | 1,968 | 56.9207 | 91.8699% | 420.0 | 663.1911 |

CAL n=3,941 observations and a finite-sample nominal 90% calibration residual quantile q=180 Wh for last-observation. 11 constructed input/holdout regression tests passed in the source workflow. These are real *DEV* scores only; the subsequently exposed DEV must not be reused as an independent confirmation population. Single household, serially correlated samples and overlapping future horizons limit nominal interval guarantees. Historical timestamps are naive original site-local readings, not proven publisher live available-at times.

**No original TAC-OSM training, ME111 forecasting improvement, cloud/mobile benchmark, independently validated customer demand or protected TEST scoring** was performed by this baseline. Do not conflate UCI energy persistence intervals with ME111's all-channel sensor uncertainty failures. Continue with original frozen model/source qualification and a newly authorized, truly untouched cohort for independent scientific claims.
