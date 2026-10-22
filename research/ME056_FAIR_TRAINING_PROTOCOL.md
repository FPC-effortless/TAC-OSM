# ME-056: fair training qualification protocol — proposed amendment

Date: 2026-10-22. Source PR: https://github.com/FPC-effortless/TAC-OSM/pull/334

## Decision to implement in experiment branch
Separate **scientific comparability** from **deployment execution efficiency**. Preserve the registered 5% CPU threshold as a failed engineering gate, but do not make it a prerequisite for a matched scientific training experiment. This is a proposed protocol amendment, **not yet an applied change to PR #334's code or CI**.

### Mandatory training eligibility
1. Trainable parameter count matched within registered tolerance (reported 38,266 vs 38,295, 0.076%).
2. Profiled FLOPs per training update matched within registered tolerance (42,257,505 vs 42,546,619, 0.680%).
3. Identical authentic data, train/validation/test split and preprocessing, action/authority/servo-prior inputs, supervised targets, observed outcome availability, number of updates and sample exposures, optimization-selection budget, random seeds and evaluation.
4. Report measured CPU per update and total wall-clock cost, but **do not equalize wall-clock time or disqualify a model for exceeding the engineering CPU target**.
5. Record pre-takeover-only factual evaluation; do not infer closed-loop policy quality. Declare grouped-four adjacent-episode split and lack of verified operator/session independence.
6. Preserve immutable registration, exact commit SHA, artifacts, checkpoint hash, metrics, all seeds, failed runs, and raw paired timing receipts.

### Engineering gate (separate, still failing)
ME-056 anchored recurrence: TAC 0.038523 s/update, GRU 0.035837 s/update; relative overhead 8.007%, paired bootstrap 95% CI [6.35%, 10.00%]. The 5% engineering gate remains **FAIL**. Prior ME-055 remeasured overhead 16.17%; relative reduction ~50.5%. No change to those facts.

### Train and evaluate only after implementing the amended code/CI contract
- Verify data availability, storage accessibility, runnable training workflow and budget.
- Train both TAC and GRU under identical sample and update budgets and fixed seed list.
- Score persistence x_next=x_t, GRU, TAC on untouched held-out factual trajectories with MAE per physical channel, pooled MAE, horizon-specific MAE, calibration/uncertainty if available, and cost of computation.
- Report bootstrap paired confidence intervals with appropriate sequence/session clustering; disclose inability to ensure independent operator sessions.
- Compare against recovered ME-054/ME-056 developmental values only if splits and protocols are confirmed identical; otherwise separate tables.
- If persistence still wins, do not promote architecture as a real-world forecasting success.
- Validate all inherited CI; do not assume 960 passing local tests means all upstream checks passed.
- ME-054 completed workflow artifacts must be retrievable and verified before any claimed result.

### Hypotheses for subsequent ablations (after fair training)
A. Memory retrieval in physical forecast; B. fixed 128-frame slow anchors versus learned retention; C. concept CED per-pass gain; D. batched dense transition supervision. Match params, FLOPs, data and budgets in each experiment.

### Audit status
This file records protocol design only. No CPU gate changed, no training launched, no new measured result, no model promoted.
