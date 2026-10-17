# DEV-044–048 audited development results

17 October 2026. Six bounded studies, three training seeds per study. The continuation fits 96 new model/head configurations; DEV-044A contributes 27 earlier checkpoints. All six studies pass their separate replay audits. These are development findings, not sealed confirmation.

The simplest exact-history controller remains the strongest decision/control reference in these restricted simulators. Lower neural average cost often comes with additional failures. No integrated-system superiority claim passes the evidence gates.

## Branches and review

| Study | Review | Scope |
| --- | --- | --- |
| DEV-044A | [PR #200](https://github.com/FPC-effortless/TAC-OSM/pull/200) | Original gain/retention pilot; source and evidence preserved |
| DEV-044A additional audit | [Replay branch](https://github.com/FPC-effortless/TAC-OSM/tree/research/dev044-independent-replay-2026-10-17) | Checkpoint-to-probability/calibration replay for all 27 models |
| DEV-044B0 | [PR #207](https://github.com/FPC-effortless/TAC-OSM/pull/207) | Separate writer and internal-reader interventions |
| DEV-045B | [PR #203](https://github.com/FPC-effortless/TAC-OSM/pull/203) | Frozen readout feature and probability-only heads |
| DEV-046B | [PR #204](https://github.com/FPC-effortless/TAC-OSM/pull/204) | Forecast, decision, joint and simulator-teaching objectives |
| DEV-047A | [PR #205](https://github.com/FPC-effortless/TAC-OSM/pull/205) | Entity/identity/history/queue shifts and retention probes |
| DEV-048A | [PR #206](https://github.com/FPC-effortless/TAC-OSM/pull/206) | Sparse prediction, fallback, constrained execution and verification |

The research PRs remain draft and unmerged. Earlier DEV-015–043 evidence is unchanged.

## DEV-044A attribution qualification

Selected trainable memory costs **0.830223** per entity versus **1.000000** for frozen memory, with failures **89 / 4 / 45** versus **0 / 0 / 0**. The guard fails. Schema41 costs approximately **0.530816**. Crucially, the frozen memory module also froze its internal `query` and `query_prior` read projections. This contrast cannot isolate writer learning. DEV-044B0 tests those parameter groups separately.

## DEV-044B0 writer-by-reader factorial

108 updates, 128-event training histories, 512-event calibration/evaluation; base memory profile only. External event/query encoders, adapter and head train in every arm. Writer and internal-reader hashes and gradients are checked independently.

| Writer / internal reader | Mean cost/entity | Failures by seed | Writer guard within reader stratum |
| --- | --- | --- | --- |
| writer_train_reader_train | 0.867477 | 39 / 0 / 13 | False |
| writer_frozen_reader_train | 0.961227 | 0 / 0 / 26 | True |
| writer_train_reader_frozen | 0.971933 | 1 / 0 / 28 | False |
| writer_frozen_reader_frozen | 1.000000 | 0 / 0 / 0 | True |

The factorial does not establish reliable writer superiority; it also does not replace matched baseline tuning, longer optimization or ten new independent seeds.

## DEV-045B frozen-representation interfaces

Each head is compared with the independently calibrated original threshold on its own memory and fresh cohort. Every head, including negative results, is retained.

| Memory | Head | Mean cost/entity | Failures by seed | Failure guard vs original |
| --- | --- | --- | --- | --- |
| selected_trainable | forecast | 0.727720 | 8 / 57 / 15 | False |
| selected_trainable | decision | 0.914062 | 0 / 0 / 4 | True |
| selected_trainable | contrastive | 0.738137 | 21 / 68 / 15 | False |
| selected_trainable | joint | 0.722512 | 8 / 40 / 34 | False |
| selected_trainable | probability_joint | 0.825231 | 10 / 57 / 10 | False |
| selected_trainable | original_tuned_threshold | 0.823495 | 18 / 20 / 10 | True |
| selected_frozen | forecast | 1.000000 | 0 / 0 / 0 | True |
| selected_frozen | decision | 1.000000 | 0 / 0 / 0 | True |
| selected_frozen | contrastive | 0.992477 | 0 / 0 / 42 | False |
| selected_frozen | joint | 1.000000 | 0 / 0 / 0 | True |
| selected_frozen | probability_joint | 1.000000 | 0 / 0 / 0 | True |
| selected_frozen | original_tuned_threshold | 1.000000 | 0 / 0 / 0 | True |
| schema41 | forecast | 0.543981 | 3 / 1 / 0 | False |
| schema41 | decision | 0.544271 | 2 / 1 / 2 | False |
| schema41 | contrastive | 0.543403 | 0 / 0 / 0 | True |
| schema41 | joint | 0.543981 | 0 / 0 / 0 | True |
| schema41 | probability_joint | 0.539641 | 2 / 0 / 0 | True |
| schema41 | original_tuned_threshold | 0.541377 | 2 / 2 / 0 | True |

Raw-feature heads can improve cost without solving reliability. For example, the joint head on selected learned memory has 82 failures versus 48 for the original threshold. No general readout impossibility follows from either this result or the earlier calibration failures.

## DEV-046B predictive/decision objectives

| Arm | Mean cost/entity | Failures by seed | Failure guard vs forecast in same freeze condition |
| --- | --- | --- | --- |
| forecast_trainable | 0.656250 | 10 / 4 / 11 | True |
| decision_trainable | 0.939236 | 6 / 0 / 0 | True |
| joint_trainable | 0.605324 | 9 / 32 / 11 | False |
| teacher_trainable | 0.574074 | 16 / 27 / 12 | False |
| forecast_frozen | 0.980903 | 6 / 0 / 0 | True |
| decision_frozen | 0.999711 | 0 / 0 / 0 | True |
| joint_frozen | 0.991319 | 0 / 0 / 0 | True |
| teacher_frozen | 0.989294 | 20 / 0 / 0 | False |

The teaching objective uses extra labeled student-encountered receipt states and additional computation. It is ActionOPD-inspired simulator supervision, not an ActionOPD language-agent reproduction.

## DEV-047A entity and queue shifts

Train at 8 entities; evaluate 8/16/32, 128/512-event histories, unseen registered public IDs and multiple queued interventions. The exact public reference costs **0.510355** on average with zero failures. Stored state is 25 scalars/entity: 200, 400 and 800 scalars, respectively.

| Variant | Mean cost/entity | Failures by seed | Failure guard vs public event queue |
| --- | --- | --- | --- |
| multiscale_gru | 0.607205 | 132 / 9 / 12 | False |
| fast_gru | 0.716642 | 14 / 507 / 12 | False |
| selective_diagonal | 0.780072 | 0 / 12 / 10 | False |
| temporal_order | 0.532676 | 12 / 24 / 17 | False |

Shared weights and permutation checks pass; new identities use an explicit public catalog. This is not learned identity discovery. The fast/temporal writer replacements are architecture comparisons, not byte-matched single-channel interventions; see the reproduction qualifications. The selective diagonal arm is not Mamba. The earliest-event retention probe does not train retention and may remove an event that is not decision-critical.

## DEV-048A complete-loop accounting

Development selected `exact_history` in all three seeds before evaluation. Costs include post-horizon verification inspections; sensing fallback also pays its pre-action inspections. These cohorts differ from DEV-047, so cross-study cost differences are not treatment effects.

| Candidate | Mean cost/entity | Failures by seed | Coverage / qualification |
| --- | --- | --- | --- |
| exact_history | 0.608371 | 0 / 0 / 0 | complete |
| route2_exact | 0.610603 | 0 / 2 / 0 | complete |
| route4_exact | 0.619531 | 0 / 8 / 0 | complete |
| route8_exact | 0.651525 | 2 / 8 / 0 | complete |
| route2_inspect | 0.797173 | 8 / 19 / 0 | complete |
| route4_inspect | 0.787574 | 8 / 21 / 0 | complete |
| route2_review | unknown | unresolved | 14.58% mean entity coverage per world; no complete cost claim |
| route4_review | unknown | unresolved | 29.17% mean entity coverage per world; no complete cost claim |

Routing reduces the entity positions entering the first OSM call. Inspection fallback predicts the remainder, restoring full entity prediction count and adding sensing work. History fallback scans history. Runtime includes encoding, routing, fallback, simulated execution and verification; model initialization/worker startup is excluded. Timings were collected on a shared CPU environment and are descriptive, not FLOP-matched performance evidence.

## Verification and artifacts

* [1,253 tests and training smokes passed](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39014569984). Focused writer/query independence tests pass.
* DEV-044A original Actions run [39011950314](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39011950314) passed; original raw ZIP digests were verified byte for byte locally.
* Separate audits reconstruct checkpoint forecasts and calibration and replay simulator interventions. DEV-048 replays every candidate and charges sensing/verification. Audits do not independently repeat optimizer steps.
* [Complete evidence branch](https://github.com/FPC-effortless/TAC-OSM/tree/evidence/dev044-048-development-2026-10-17): original DEV-044A seed ZIPs, continuation checkpoints/raw outcomes, source bundle, protocols, audit outputs, CI logs and SHA256 manifest. GitHub binary release upload returned HTTP 401, so the archives are preserved on this dedicated branch; main and the research source branches contain no evidence binaries.

Interpretation and remaining gates are in [the research discussion](DEV-044-048-INTERPRETATION.md), [reproduction guide](DEV-044-048-REPRODUCTION.md) and [source review](DEV-044-048-RESEARCH-SOURCES.md). No result establishes learned memory-item retention, calibrated uncertainty/value-of-information, original CASM/CDL equivalence, GEPA LLM-reflection effectiveness, industrial twin validity or untouched-domain superiority.

Failure vectors are descriptive seed-level outcomes; individual worlds are not independent training replications. Cryptographic hashes establish evidence identity, not scientific validity.

DEV-049 reporting correction: DEV-048 review coverage is the mean fraction of entity decisions resolved within each world, not the fraction of completely resolved worlds. Numeric values and original evidence are unchanged. The implementation computes `(actions != HUMAN_REVIEW).float().mean()` per world.
