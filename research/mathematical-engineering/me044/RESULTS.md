# ME044: separate multiscale native innovation decoder

Complete architecture-first training on actual measurements. Historical development only; no newly sealed test, merge or promotion.

The model retains the original context predictor and gives native memory a separate bounded forecast decoder. Six observed-token fast/slow/associative updates predict16 spline/seasonal coefficients. Both cadence priors and forecast basis are shared with equally sized context and GRU competitors. Native expert learning occurs every fourth update; inference always executes the expert.

**80 complete downstream fits +8 new AR parent fits**. Three capacity-matched systems and two unmatched practical references. Fixed1000-update and separate10optimizer-CPU-second training lanes; two seeds, horizons96/720, two measured domains. No post-score changes or early stopping.

## etth1: registered primary MSE

Measured oil temperature. Train-only standardized units, lower better. Do not compare raw scores across domains.

| Complete system | 1000 updates | 10 optimizer CPU seconds | Mean CPU at1000 updates |
|---|---:|---:|---:|
| original_context | 0.154661 | 0.155516 | 1.819s |
| previous_scheduled | 0.154419 | 0.157474 | 2.601s |
| pyramid_context | 0.153875 | 0.153316 | 2.128s |
| pyramid_gru | 0.151785 | 0.155873 | 2.347s |
| pyramid_native | 0.151925 | 0.153572 | 3.343s |

Original fixed-update checkpoints and per-horizon/per-seed scores are retained in results.json. Matched systems have active parameter gaps<0.04%; original_context and previous_scheduled are deliberately unmatched practical references.

| Registered native gate | Result |
|---|---|
| all | FAIL |
| equal_CPU_accuracy | FAIL |
| no_large_horizon_regression | PASS |
| one_percent_accuracy | FAIL |

## tetouan: registered primary MSE

Three measured power zones. Train-only standardized units, lower better. Do not compare raw scores across domains.

| Complete system | 1000 updates | 10 optimizer CPU seconds | Mean CPU at1000 updates |
|---|---:|---:|---:|
| original_context | 0.108966 | 0.103030 | 1.858s |
| previous_scheduled | 0.107785 | 0.103404 | 2.645s |
| pyramid_context | 0.110430 | 0.105801 | 2.162s |
| pyramid_gru | 0.111009 | 0.101859 | 2.413s |
| pyramid_native | 0.109533 | 0.102906 | 3.396s |

Original fixed-update checkpoints and per-horizon/per-seed scores are retained in results.json. Matched systems have active parameter gaps<0.04%; original_context and previous_scheduled are deliberately unmatched practical references.

| Registered native gate | Result |
|---|---|
| all | FAIL |
| equal_CPU_accuracy | FAIL |
| no_large_horizon_regression | PASS |
| one_percent_accuracy | FAIL |

## Original model verification

- 80 original optimizers/checkpoints checked; 18,032,640 saved full-forecast values replay bit exactly.
- 24 actual fixed-update state comparisons confirm every shared cheap-path coefficient remains bit-identical to original_context. Cheap forecasts, actual truths and trust/router arrays also replay exactly.
- Independent float64 MSE/MAE recomputation uses canonical C-layout reductions and the preregistered epsilon/operation-count bound. No forecast replay tolerance relaxed. Original ME043 strict reduction-equality failure and ME035B ridge-equivalence exception remain unchanged.
- Per-parameter optimizer clocks verify native/context/GRU expert updates every fourth step and previous scheduled expert/router every eighth step. Actual process/wall time, samples, parent cost and peak process RSS are recorded.

## Actual full-window inference cost

Batch1, original first DEV window,5warmups then5x20calls; geometric mean of per-checkpoint median CPU across horizons/seeds. Includes normalization, full-history AR, context, native/GRU computation, basis decode and all actual outputs. Process CPU is not a streaming update claim.

| Domain / training lane | System | Full forecast CPU ms |
|---|---|---:|
| etth1 / exposure | original_context | 0.215 |
| etth1 / exposure | previous_scheduled | 1.256 |
| etth1 / exposure | pyramid_context | 0.293 |
| etth1 / exposure | pyramid_gru | 0.476 |
| etth1 / exposure | pyramid_native | 1.221 |
| etth1 / cpu | original_context | 0.214 |
| etth1 / cpu | previous_scheduled | 1.263 |
| etth1 / cpu | pyramid_context | 0.295 |
| etth1 / cpu | pyramid_gru | 0.479 |
| etth1 / cpu | pyramid_native | 1.225 |
| tetouan / exposure | original_context | 0.215 |
| tetouan / exposure | previous_scheduled | 0.790 |
| tetouan / exposure | pyramid_context | 0.288 |
| tetouan / exposure | pyramid_gru | 0.475 |
| tetouan / exposure | pyramid_native | 1.218 |
| tetouan / cpu | original_context | 0.211 |
| tetouan / cpu | previous_scheduled | 1.200 |
| tetouan / cpu | pyramid_context | 0.293 |
| tetouan / cpu | pyramid_gru | 0.476 |
| tetouan / cpu | pyramid_native | 1.217 |

## Optional compiled native execution

A separately registered post-fit execution optimization compiles only the native branch. Original learned weights and all other components remain unchanged; no new fitting or predictive accuracy result.

Recorded result: EXACT_REPLAY_PASS. Target >=1.25x geometric-mean full-forecast speedup; exact-output guard compulsory. Full details in [inference.json](inference.json).

All16 native checkpoints replay3,606,528 full forecast values exactly after compilation. Paired full-window CPU speedup **1.414x**, target **PASS**. Compilation time is separate, default backend unchanged.


## Comparison manifest and interpretation

- Parameter parity PASS only within the three registered pyramid systems. All modes/priors/readout information match there. Practical references are unmatched; no dummy weights.
- Data/exposure PASS within fixed1000-update matched systems: identical observed source rows, target windows, sampler sequence and expert schedule. Actual CPU differs. Compute-lane budget PASS with final-step overshoot if the recorded cap was reached; sample/expert exposures differ. Full three-way parity FAIL: **NO ARCHITECTURAL SUPERIORITY CLAIM**.
- Same fixed endpoint and parent selection rules, one model configuration per arm, both seeds and horizons preserved. No result-dependent target switching. Whole-study design follows repeated historical DEV exposure, not independent new-site evidence.
- Cheap-trajectory retention is a software/optimization guarantee, not proof that native state or its additional residual improves future prediction. Bounded corrections do not establish safety or calibration.
- Modular stop-gradient training preserves cheap-path weights; the full expert objective does not backpropagate through that cheap encoder. Inference always runs the expert, unlike ME043 hard routing. No persistent streaming, adaptive System2, causal action, counterfactual, commercial or foundation-model claim.
- All original weights, optimizers/RNG, losses, source hashes, genuine predicted/observed arrays, timing and negative gates are retained in repository-owned lossless indexed archives. Archive restoration and CI establish their actual delivery/replay status separately.
