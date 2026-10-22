# Research Ideas Registry — TAC-OSM and related programs

This is a **starter registry in TAC-OSM**, pending creation of a separate cross-project ideas repository. The registry does not claim that proposals were implemented or evaluated.

| ID | Idea | Origin | Status | Evidence or next test |
|---|---|---|---|---|
| TAC-001 | Direct associative retrieval for physical forecast | ME-054 review | Proposed | Ablate on authentic robot trajectories |
| TAC-002 | Multiscale learned retention | MTSK transfer | Proposed | 128-state matched comparison |
| TAC-003 | Prediction-error responsibility writes | MTSK transfer | Proposed | Key collisions and long-horizon error |
| TAC-004 | Entity-local memory updates | MTSK transfer | Data-dependent | Requires reliable persistent IDs |
| TAC-005 | Bounded norm projection versus tanh compression | ME-054 mathematics | Proposed | Memory stability and forecast MAE |
| TAC-006 | Concept routing diversity and specialization | ME-056 | Diagnostic proposed | Effective rank, entropy, error attribution |
| TAC-007 | Cross-context concept attention | LLM routing analogy | Proposed | Equal-budget routing ablation |
| TAC-008 | Adaptive recurrent decoder depth | LLM latent scratchpad / CED | Proposed | Per-pass held-out forecast and cost |
| TAC-009 | Hierarchical latent-first dynamics | JEPA/world models | Proposed | Action-conditioned multihorizon real forecasts |
| TAC-010 | Elapsed-time liquid-inspired recurrence | LTC/CfC | Conditional proposal | Irregular timing analysis first |
| TAC-011 | Memory/working-state role separation | KV-cache analogy | Proposed | Retrieval and interference tests |
| TAC-012 | Counterfactual decision-preserving retention | TAC-OSM research | Research program | Logged actions, costs, identification limits |
| TAC-013 | Bounded verification and correction | ME-054; DEV-035 | Existing principle | Compare real correction errors |
| TAC-014 | Hardware-efficient recurrent execution | LLM/SSM engineering | Proposed | FLOPs and wall time, secondary to capability |
| TAC-015 | Structured entity-factorized state | DEV-017/020 | Historical baseline evidence | Verify protocols and replicate real-data relevance |
| TAC-016 | Fully learned neural action/stop control | DEV-058/061 | Synthetic evidence only | Real decision-cost validation |
| TAC-017 | Native and Concept-CED real-world benchmarking | ME-054/056 | Developmental results | Persistence beats both reported MAEs |
| TAC-018 | Retrieval versus retained-state diagnostic | Berkeley stale binding analogy | Proposed | Probe memory contents vs actual readouts |
| TAC-019 | Unproductive refinement loop detection | RADAR analogy | Proposed | Pass-wise prediction gain |
| TAC-020 | Long-horizon execution reliability | NVIDIA long transduction analogy | Proposed | Real long-history error and action robustness |
| CROSS-001 | CDL/Hebbian-style relevance learning | Cross-project idea | Unverified | Formalize update and benchmark |
| CROSS-002 | CASM decision mechanism outside ML | Cross-project idea | Unverified | Define domain-specific prototypes |
| VER-001 | Absorb MiMo RL-environment design into Veritas | Veritas research | Proposal | Verify repo design and dependencies |
| CROSS-003 | Unified reusable GitHub skills | Cross-repo workflow | Proposal | Repo-wide compatibility audit |

## Historical experiment index (verify source evidence before interpreting)
| Experiment | Reported result | Limitations |
|---|---|---|
| DEV-016 | Joint 73.64%, frozen writer 59.32% | Predictive reliability gate failed |
| DEV-017 | Structured 91.84%, TAC 57.90% future prediction | Experimental scope must be retained |
| DEV-020 | Structured probes 100%, TAC ~50-51% | Probe failure does not identify unique cause |
| DEV-043 | Pairwise ranking 100%, hard accuracy 50% | Calibration/representation concern |
| DEV-058 | 576/576 default, 560/576 repeat | Synthetic, no real-world claim |
| DEV-061/R1 | Reported 100% primary completion | Synthetic; persistence wins real forecasting |
| ME-054 | MAE 0.07881 vs GRU 0.09232 | Persistence 0.06695; parity needs audit |
| ME-056 | MAE 0.08037 vs GRU 0.09082 | Persistence 0.06695; parity needs audit |

## Record schema for future ideas
Each idea should include: ID; title; problem; proposed mathematics; origin; relevant repositories; relation to existing ideas; hypotheses and failure modes; status (proposed/diagnostic/implemented/trained/verified/rejected); commit/PR; datasets and splits; parameter count; training budget; baselines; seeds; metrics; confidence and limitations; next experiment.

No synthetic result may be labeled real-world success. Never erase negative findings. Source code, checkpoints, reports and runs should be linked rather than duplicated without provenance.

## New evidence entry — ME-057 prospective scientific qualification (2026-10-22)

- **ID:** TAC-021.
- **Lineage:** Anchored native ME-056 in [PR #334](https://github.com/FPC-effortless/TAC-OSM/pull/334), carried into [draft PR #335](https://github.com/FPC-effortless/TAC-OSM/pull/335).
- **Idea:** Separate compulsory matched scientific resources (parameters, data/order, FLOPs/budgets, evaluation) from CPU speed, which is an independent efficiency outcome.
- **Implementation status:** Prospective parity-v3 validator changes, unit tests, and real FIT original-source audit script committed; no model training outcome.
- **Evidence preservation:** Older parity-v2 CPU gate remains a documented historical failure (8.007% overhead, 95% CI 6.353–9.999%). Never reinterpret old results as passing.
- **Next experiment:** Verify source audit for all 228 authentic FIT episodes and then preregister a real matched two-arm multi-seed prediction study; measure persistence, GRU and TAC.
- **Known blockers:** No verified new FIT receipt yet; source independence across operators/sessions cannot be proven from published metadata; TEST remains sealed.

### ME-056 lineage warning
There are **two distinct experimental designs called ME-056**:
1. **Concept-Swarm/CED ME-056**, [PR #332](https://github.com/FPC-effortless/TAC-OSM/pull/332): 6×15 concept states and shared-weight refinement loop, separate original robot source.
2. **128-frame anchored native ME-056**, [PR #334](https://github.com/FPC-effortless/TAC-OSM/pull/334): fast32/slow32 plus 8D relevance trace; 38,266 native parameters, dense transition auxiliary.

Do not merge the physical state dimension, memory type, model counts, dataset splits or measured scores from one lineage into the other. Registry entries must identify the exact branch/commit and source task.

### TAC-022 — Causal pre-feedback dense supervision

- **Origin:** Code audit of anchored ME-056 PR #334's `boundary_objective`; tracked in successor [PR #335](https://github.com/FPC-effortless/TAC-OSM/pull/335).
- **Finding:** Recorded transition snapshots had already ingested the next measured physical state through verification, while the auxiliary regressed a target from that same transition. This is a hindsight-information leak for forecasting supervision.
- **Fix:** Save snapshot immediately **before** consuming the next physical measurement; regression test mutates the next-state target and checks that the corresponding pre-feedback snapshot is invariant.
- **Status:** Implemented and submitted to CI; not trained or evaluated for predictive skill. Reprofile parameter/FLOP and runtime receipts for the corrected computation before any new comparison.
- **General rule:** Distinguish causal forecasting losses, after-the-fact state verification, and reconstruction of already seen targets. They must not be mislabeled as interchangeable prediction evidence.
