# ME073: native autoregressive event/time trajectory development

## Registered whole-model objective

Build and train one integrated native TAC candidate that generates eight
successive event/time steps instead of independent direct horizon states.
Use authentic BPI2012 FIT and historically exposed DEV REPORT. Five fixed
seeds73001–73005, six complete epochs, no comparator/ablation matrix or winner
claim. Retain ME072 originals unchanged. This study strengthens architecture
and trained abilities; it cannot prove general intelligence or customer value.

Registration and original code must be committed before first fit, pass the
single-architecture `validate_research_parity.py --phase preflight` gate, and
preserve all five originals in Git before any REPORT score. `FAIR_COMPARISON.json`
is canonical for original source/split hashes, exact budgets and tolerances.

## Available-at-time source and coverage

Source: BPI2012 original financial institution process log, publisher file
24027287, DOI10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f,
SHA2565cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3.
4TU General Terms of Use; raw original bytes stay external. Reuse the inherited
7,096 FIT cases, frozen chronological exclusions and unused TEMP/CQR roles.
REPORT last925 DEV cases has been exposed in earlier studies; no new TEST
scoring, synthetic evidence, BPI2018 retry or independent transfer claim.
Original source is necessarily parsed to reconstruct the inherited split.

Trace keys include REG_DATE, AMOUNT_REQ and case identity. Their publication
does not establish prediction-time availability. Therefore **exclude all three
from learned inputs**. Initial fast/slow context uses the first actually
recorded activity, resource and timestamp-derived features, available by all
registered query ages. Resources for future events are unknown and not supplied.
No outcome, suffix length, terminal status or future timestamp enters generation.

Recorded absence beyond a source suffix is **EOR**, a retrospective published
record-boundary proxy. Its label availability cannot be backdated to the last
event. Neither censoring nor business completion is identified from this.
Score source-boundary mass separately; never call it successful workflow closure.

## Integrated architecture

Retain observed native fast64/slow64/8x8 associative dynamics and 32-event
truncated-backprop carry. Initialize fast/slow in[-.5,.5] from the first observed
event representation. This reorganizes available information, not new metadata.
The observed state is read into a shared SwiGLU step decoder. Each future step
predicts conditional source-record continuation and activity; a conditional
clock depends on the proposed activity as well as the latent state.

During teacher-forced FIT likelihood, feed the actual preceding future mark
and incremental delay into **separate functional copies** of native memory.
During free generation, feed the model's sampled mark and delay. Future resource
is unavailable; a learned imagination encoder consumes only the proposed mark,
log1p delay and zero-delay flag. State advances never alter observed memory.

The coherent joint path factors into eight conditional continuation, mark and
hurdle/lognormal-mixture increments, stopping at first EOR. Step delays are
nonnegative and summed, so each path has nondecreasing elapsed time. Dependence
between steps fixes independent readout trajectories; it does not enforce all
legal business transitions. Teacher-forcing exposure bias remains a limitation.

Three-component positive clocks use FIT-only current/next-activity pair priors,
backing off for fewer than16 pairs. Locations are FIT log-gap quantiles plus
bounded learned offsets[-2,2]. Scales remain in[.15,1.25]. **ME072 already bounded
scales**: its extreme intervals were not proof of unconstrained variance. The
new clock controls per-step locations and couples time to activity; total time
has no arbitrary72h cap and real extreme delays retain finite likelihood.
Student-t tails and a hard time cap are not adopted without evidence they help.

h1 mark logits use the existing FIT-only5-gram prior; later steps use FIT
transition priors and native imagined state. This is a predictive log pool,
**not** a pointwise baseline safety guarantee. Hard lowest-variance routing is
not used: unjustified confidence can select a systematically wrong expert.
The native nonlinear recurrence cannot generally be parallelized by affine
scan. An SSM core swap would be a separate registered branch, not this native
candidate. Detached state carry exists already and is not unlimited retention.

## Training and forecast contract

AdamW lr.001, weight_decay.0001, clip1; batch32 complete real cases, chunk32,
six epochs, immutable per-seed schedules. No DEV tuning, early stopping,
posthoc temperature/conformal fitting, seed/model selection or failed-fit erasure.
Observed prefix ages1–6 receive weight2; all others weight1. This fixed emphasis
is intended to address early forecasting. Count repeated exposures explicitly.
Each prefix supervises up to eight suffix factors, masking after first EOR.
Exact categorical/continuation loss +.2 zero/positive log-time density +.05
conditional mark Brier. Report original-seconds likelihood with Jacobian.

Freeze original model/optimizer/RNG bytes, all final checkpoints and training
receipts in repository before scoring. Report ages2/6/16/32/64 with reached
case counts925/555/165/52/2 and horizons1/2/4/8. Age64 is too sparse for a
memory claim. Age comparisons involve different surviving case populations.

Free-running horizons use256 deterministic scrambled Sobol trajectories,
seed73000, common points across cases, float64 inference from original float32
weights. Integrate survival with path weights; weighted activity masses and
elapsed-time quantiles summarize the same autoregressive generative law.
h1 activity probabilities are exact. h2/4/8 are finite-QMC approximations with
registered normalized epsilon1e-6 smoothing, not exact analytic marginals.
Conditional-on-survival quantiles can reverse across horizons because the
surviving population changes, despite monotone clocks on every single path.
Keep original forecast arrays and eight representative full paths per query.

Report all reached cases; real-future conditional activity accuracy/NLL,
activity/EOR NLL and Brier, clock median MAE,90%coverage, mean andp50/p90/p99
widths, proper interval scores and24h conditional-delay Brier. No time labels
invented for EOR. Exact factual suffix joint density NLL is reported separately
from free-running marginal scores; teacher-forced conditional accuracy must
never be advertised as a look-ahead forecast. Repeated seeds and query ages
are not independent workflows. Score uncertainty and width together.

## Acceptance, failure and preservation

Engineering acceptance: finite joint probabilities and active gradients,
normalized generated marginal mass, nonincreasing survival, nonnegative path
time, first-record-only initialization, causal/no-label generation, no factual
memory mutation, correct EOR likelihood mask and chunk-carry consistency.
Capability readout: publish all five original seeds and all age/horizon metrics,
including failures. Objectives are stronger early branches and more useful
distant event/time forecasts. No comparator or superiority acceptance gate.
A failed objective stays a negative result; do not retune on REPORT or TEST.

Bundle `research/replay-bundles/ME073/`, expected<15MB weights plus original
arrays/config/receipts; preserve original optimizer/RNG and recoverable failure
states. Numerical contract fixed before fit: output abs/rel1e-5, aggregate
abs1e-4/rel1e-5. Independent clean GitHub checkout must restore and recompute
all original distributions, paths and real REPORT scores without refitting;
otherwise REPO_REPLAY=FAIL/BLOCKED. No threshold relaxation after results.
Published prior methods and exact provenance are in FAIR_COMPARISON.json.
