# DECISION-SYS1-001 — efficient decision-model recovery

Woyengi Intelligence Labs, 2026-11-06. **UNTRAINED engineering candidate.**

## What is implemented

`src/tac_osm/decision_sys1_001.py` contains two separately named candidates:

1. `NativeRiskStateModel(212,301)`: exact native fast32/slow32/matrix8x8 recurrence from the merged predictive-state agent (PR320), a read-only associative query, and a 284 -> 301 -> 4 risk head. Four interval hazards yield five ordered normalized probabilities via log-sigmoid survival products. Actual parameter count **99,963**. Carried state: **128 scalars / 512 FP32 tensor bytes per entity**, excluding model weights, metadata, indexing and transient runtime. This is a nonlinear recurrent associative-memory model, not a Transformer, GRU or standard Mamba/SSM layer. The delta-rule memory has established related prior art; no novelty claim.
2. `SurpriseGatedPredictiveStateAgent`: original observation/prediction methods and parameter keys, with a changed feedback operator. The parent correction writes nonzero latent changes even when outcome equals prediction. The candidate multiplies corrections by `rho=mean(tanh(outcome-stopgrad(prediction))^2)` and selects the original row exactly when feedback is missing or rho is zero. Missing values are sanitized before learned arithmetic, including action inputs. No new parameters. Confirmation logging/uncertainty updating is separate; zero-residual no-op is a residual-correction contract, not a universal Bayesian rule.

The risk model has **no** tokenizer, CASM graph, learned entity identification, semantic retrieval index, action-policy head or feedback update. It is not full PNDS. Preserve the distinction from old DEV067: PR303 documents a GRU encoder and fixed event FIFO, not this native graph.

## Actual execution

`PYTHONPATH=src python -m unittest discover -s tests -p 'test_decision_sys1*.py' -v`

32 local numerical checks passed in Python3.13.5 / PyTorch2.10.0+cpu, one thread. Tests cover exact parent Git-blob identity, inherited observation/prediction parity, read-only queries, normalized probabilities, every risk-parameter tensor's gradient reachability, FP32/FP64 zero-residual identity, finite masked gradients, NumPy feedback correspondence, one finite-difference gradient entry, bounds and snapshot replay. The 25,023-parameter correction fixture is a 4-input/2-action/1-outcome geometry, not a trained checkpoint.

Five deterministic numerical fixture seeds601..605 all reproduced parent zero-residual state changes and exact candidate no-ops. Parent max absolute matrix changes: .0767788735, .0894280051, .1487293155, .1299663682, .1161789989; candidate: all0. These are dimensionless random-weight software diagnostics, NOT error rates, learning results or a real-data improvement.

**0 optimizer updates; 0 dataset rows scored; 0 trained checkpoints; no GPU run.** Full-repository tests, CI, independent review and full Lean/source correspondence were not executed here. Passing these tests cannot qualify learned usefulness or superior efficiency. The extra guards may themselves cost runtime; no speedup is claimed.

## Exact provenance and open mathematics

Parent commit `5f288776f87be1423ec607f5077b96400654ffdc`, parent file `src/tac_osm/predictive_state_agent.py`, Git blob `00e7c715d280e8195a38c473db5b2ef3a90291ff`. Parent bytes/checkpoints are unchanged.
New module SHA256: `af2f40ecdac845804d9977a72b9c63909877bf90249bce9b03d567ec25c3fedf`.
`reference.py` is an independent NumPy reference for the correction operator ONLY. Companion PNDS work is partial; full inherited graph, IEEE/AD, time/ownership/receipt legality and reciprocal canonical variant-index integration remain OPEN. No compiled Lean proof is claimed.

## Real-data protocol draft — admission BLOCKED

First task: observed repair/replacement risk on SCANIA Component X v3 (DOI10.5878/bnh5-ka77). It is not causal repair benefit or real monetary savings. Historical SCANIA ME001B exposure and its negative result remain evidence; TRAIN must not be called wholly unseen.

Proposed212-feature interface:105 train-normalized measurements,105 missingness flags, log1p time gap, log1p observed age. Confirm semantics/dimensions/time units from the publisher BEFORE fitting. IDs are caller-owned state keys, not target-bearing learned features. No static specifications initially. No future imputation. Keep publisher vehicle splits; audit prior exposure before claiming an untouched final cohort. Do not open TEST now.

Train on prospective as-of-readout targets with censor-aware interval likelihood. Censoring before a horizon is UNKNOWN, not no-failure. Publisher6/12/24/48 cutoff alignment is OPEN. No retrospective labels may be injected as online verification feedback. Correction utility requires separately qualified outcome-arrival/action evidence and is NOT scored in the initial risk lane.

Primary rivals: competent causal Transformer with incremental cache; multilayer GRU; faithfully implemented selective SSM (explicit Mamba-2 anchor or source-justified newer revision). Include a contemporary gated-delta reference when feasible. Baseline geometries/kernels/counts are not yet qualified. No toy attention/SSM surrogate. Practical prevalence/tree/current-readout controls are separate, unmatched references.

Intended paired seeds6101..6105. Initial declared target100,000 with +/-5% absolute band and <=1% cross-arm trainable/deployed gap. This voluntary study size is NOT a global100K ceiling. Count all deployed weights and remove dormant padding. Exact sources, feature/example IDs, updates, optimizer, tuning budget and FLOP ceilings must be frozen after source/shape qualification and BEFORE fitting. Both equal-exposure and equal-compute analyses are needed; do not hide unmatched compute behind equal steps.

Metrics: class log loss, Brier, rare-class precision/recall, calibration and alert coverage. Publisher costs remain PROXY scores. Proposed systems gate: <=1% relative log-loss degradation and <=.005 absolute Brier worsening with >=30% lower complete per-event inference CPU cost versus the strongest eligible rival; finalize practical margins on FIT/DEV before confirmatory freeze. Do not hide p99 or rare-class harm. Use paired vehicle-cluster intervals plus seed variability.

Measure ingest/preprocess/lookup/update/query/serialization, cold replay versus warm incremental paths, weights/state/index/runtime memory, CPU p50/p95/p99 and throughput, GPU throughput only on authorized hardware. Give the Transformer its cache. Energy requires measurement. Per-key fixed state is not proof of general sublinear semantic retrieval; all rivals get the same entity-key store.

**Next packet:** TRAIN-only source/exposure/schema/censoring audit and exact causal feature adapter; source-pinned competitor instantiation and actual resource accounting; independent signed FAIR_COMPARISON, compute-planner and PNDS gates; only then paired real FIT/DEV training. No training-ready registration, automatic job, merge or deployment is created here.

## Sources

- Native integration https://github.com/FPC-effortless/TAC-OSM/pull/320
- DEV067 audit https://github.com/FPC-effortless/TAC-OSM/pull/303
- Historical SCANIA negative https://github.com/FPC-effortless/TAC-OSM/pull/246
- Publisher https://researchdata.se/en/catalogue/dataset/2024-34/3
- SSD/Mamba-2 https://proceedings.mlr.press/v235/dao24a.html
- Gated DeltaNet https://arxiv.org/abs/2412.06464
- Related2026 prior art, not adopted https://arxiv.org/abs/2605.22791
