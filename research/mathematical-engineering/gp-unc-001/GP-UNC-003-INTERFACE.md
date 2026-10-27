# GP-UNC-003 — prediction-time isolation and delayed feedback (constructed no-fit)

**Scope:** source-time software boundary for eventual prequential uncertainty evaluation, not a fitted real-data model. New code: `src/tac_osm/gp_prequential_nofit.py`; tests: `tests/test_gp_prequential_nofit.py`.

## What this actually implements

- Each original frozen-base forecast row carries explicit `feature_at`, `issued_at`, `outcome_observed_at`, and `outcome_available_at` numeric timestamps.
- Reject rows where future outcome exists at or before forecast issuance, source features are issued after the prediction, any nonfinite timestamp/target exists, duplicate row IDs exist or global issue chronology reverses.
- Predict using only corrected GP state constructed from *previous* issued forecasts whose source outcome became available by the **current** forecast issuance. Delayed events remain in a separate bounded pending-result heap until eligible.
- Maintain separate Bayesian filters by original source `entity_id`. No state from entity A may enter B's filter. Original inducing embeddings must be supplied externally and frozen before evaluation.
- Score a base point forecast with a GP residual mean only as a **constructed diagnostic**. The filter consumes `truth - base_prediction` after `outcome_available_at`, never the corrected model's own forecast as a target.
- The finite pending-label queue is additional runtime memory, separately bounded from the GP's O(m²+md) numerical state. A deployment with long label delays must profile and count that queue.

## Test gates

Seven additional unittest methods cover delayed-label eligibility, poisoning unavailable future truth, distinct entity state, variance changes only after legal feedback, duplicate/chronological integrity, bounded pending heap, and rejection of nonfinite inputs.

The adapter does not inspect publisher receipts, permissions, checkpoint hashes, actual origin of frozen predictions, provenance of prior inducing points, nonstationary drift, observed vs candidate actions or sealed TEST policy. Those are mandatory external and source-specific integrations. Merely calling this function with real data would **not** authorize a real-source fit or make an independent validation result.

An evaluation with online Bayesian updates on historical development or test data is *adaptive prequential evaluation*, not frozen-model IID evaluation. It must have its own **pre-registered** allowed history/update rules, eligible timestamps, budget matching, competitor adaptation parity and independently validated source and labels. Do not use it to bypass the existing 95k–105k trained-model policy or signed independent preflight gate.

### Coverage by family

The generic contract is a reusable diagnostic component across System 1, Enterprise, Temporal, Physical, Edge-Lite and Counterfactual. Actual six-family adapters are **NOT INTEGRATED**. In Enterprise, censoring and zero duration need distinct likelihood semantics; for Physical, recorded versus candidate actions must not be conflated; Counterfactual potential-outcome uncertainty is unidentified without causal assumptions and observed propensity support.

Results remain **NO FIT, NO QUALIFICATION**. See [GP-UNC-002 diagnostics](GP-UNC-002-RESULTS.md), [TAC draft PR #491](https://github.com/FPC-effortless/TAC-OSM/pull/491), [PNDS draft #45](https://github.com/FPC-effortless/pnds-lean/pull/45) and [issue #492](https://github.com/FPC-effortless/TAC-OSM/issues/492).
