# DEV-051A — learned public-state action policy in native Veritas maintenance

**Status: COMPLETED / independently audited developmental evidence; NOT confirmatory, NOT a trained TAC-OSM world model, NOT canonical Veritas.**

Registered protocol: [research/dev051/study-protocol.json](../research/dev051/study-protocol.json). Original frozen training and evaluation source `d514fb5bd34b2de3932925cd1d1beb3b480c85f3`. The original complete training + separate-process audit is [GitHub Actions 39045522413](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39045522413), which completed **successfully**. Checkpoint/source audit uses pinned Veritas revision `b21d829b62d3cab2811e746267ca305e110ed004`. No original DEV-049 or DEV-050 numerical evidence was altered.

## Primary measured outcomes

All model initializations 7101, 7102 and 7103 were evaluated on the **same 128 new public Veritas worlds** (4 generation seeds x 32 worlds). World outcomes are paired across policies and reused across model seeds, so 384 model-seed/world outcomes are **not 384 independent external validation samples**.

| Policy | Cost/entity (each of the 3 trained seeds) | Residual faults / 128 worlds | Fully completed worlds | Blocked actions | Mean tool calls/world |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native inspect–verify–retry reference | 0.907421875 | 0 | 128/128 | 0 | 5.7265625 |
| Explicit public-belief rule | 0.907421875 | 0 | 128/128 | 0 | 5.7265625 |
| Behavior-cloned policy | 0.907421875 | 0 | 128/128 | 0 | 5.7265625 |
| On-policy student-state aggregated policy | 0.907421875 | 0 | 128/128 | 0 | 5.7265625 |
| Calibration-gated student policy | 0.907421875 | 0 | 128/128 | 0 | 5.7265625 |
| Inspect/repair once | 1.20625 | 33 | 97/128 | 0 | 3.96875 |
| Trust stale maintenance records | 1.7609375 | 71 | 64/128 | 0 | 2.171875 |

Each trained seed saw 983 supervised public-state teacher examples, and student-state aggregation collected 496 additional public-state examples. Both behavior cloning and aggregation had **160 optimizer updates each** from identically initialized heads; aggregation used extra native simulator interactions and so was not equal wall-clock or total compute. Initializations were independently seeded 7101, 7102 and 7103.

The calibration-only confidence gate chose **0.0 for all three seeds**, and the final fallback fraction is **0.0**. Thus the fallback itself **did not produce** the zero-failure result: the trained learner's chosen actions matched the explicit public-state teacher under the tested distribution. It also showed **zero** improvement in cost, faults or tool calls over the teacher. The aggregate audit's strict reliability **and** lower-cost success criterion is false for every trained arm.

## What was learned, and what was not

* **Positive:** A compact 12-feature / 32-hidden-unit public-state neural policy can imitate this verified multi-step reference on new episodes from the **same known synthetic generator**, while tracking public inspection outcomes, action acceptance and ordered repair attempts. Its own policy ran, incurred physical native action costs, and produced no observed residual faults.
* **Negative:** Student-state aggregation did not improve the calibrated native outcomes over plain supervised teacher imitation; this is a meaningful null result, not evidence that on-policy learning is universally useless. With this small deterministic task, learned action selection has not earned its complexity over a compact explicit state controller.
* **No learned predictive world model attribution:** The 12 input features are hand-engineered public belief state fields; the trained component is an action classifier. It does not learn event-to-state PLM dynamics, forecast `P(s_future | do(a))` under alternate interventions, learn entity discovery, or demonstrate a complete CDL–OSM–CASM loop.
* **No risk certificate:** Accurate inspection, at-most-two service attempts, two known public entities, exclusive slot isolation, and guaranteed second repair are fixed construction assumptions. Three head initializations on 128 repeated evaluation worlds cannot establish any real-world failure-rate upper bound.
* **No external Veritas qualification:** This is a newly authored public task, not a sealed Veritas benchmark. The native runtime and seven-dimensional verifier execute and record effects; policy receives only public records and charged observable receipts, not the hidden oracle.
* **No cross-study leaderboard:** DEV-050A's 0.917578 reference was measured on a different cohort; its absolute cost should not be compared as an improvement from this study's 0.907422.

## Integrity and reproducibility

* **1,277 regression tests passed** including nine new public feature/action/Veritas boundary checks.
* All three pretrained heads, raw decision traces, calibration candidate costs and episode verifier outcomes were preserved as per-seed model/checkpoint reports.
* A **separate-process independent audit passed**: reloaded each original checkpoint, reconstructed calibration choice, evaluated all seven fixed policies, replayed native Veritas tools and compared every recorded action, acceptance/inspection response, residual fault and verifier result. The audit did not independently repeat the optimization trajectory.
* Earlier static-analysis failure due to a missing timing import is preserved in CI history; it occurred before scientific training. Source was frozen at `d514fb5` for all three science jobs. No protocol or model recipe was changed after seeing held-out results.
* [Original complete evidence archive](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39045522413/artifacts/12104804673) includes all three original per-seed output directories and the independent aggregate `analysis.json`. GitHub Actions retention is finite; the separate evidence release will preserve it as a GitHub prerelease.

## Next falsifiable research gate — DEV-052

Register a **new** world-generation distribution in which sensors can be noisy, second repairs can fail, tools have different costs, and shared slots are contested. The teacher from DEV-050 can no longer rely on guaranteed repair. Compare a calibrated explicit Bayesian belief-state planner, a cost-aware learned state-action policy and a learned recurrent PLM + OSM with identical information availability. Evaluate: cost, unsolved incidents, trajectory-wide invariant violations, coverage (including unresolved referrals), value of inspections, out-of-distribution shifts and measured compute. Use fresh splits, configuration-disjoint seeds, unseen IDs and repeatable native effects. Existing DEV-051 results stay unchanged; the new domain and its cost convention must be preregistered before implementation.

**Interpretation:** Learnt public action selection has cleared a bounded engineering/evaluation milestone, not a superior prediction, architecture or commercialization milestone. The exact-state reference remains the comparator to beat.
