# ME072 — native marked multi-horizon world-model development

Owner-directed integrated capability construction, five native seeds
72001–72005, one architecture. No comparator/lesion matrix, architecture-winner
claim or isolated-memory prerequisite. Apply the existing single-architecture
development gate before and after actual fitting. The research question is
whether one trained native state supports several observed future events,
coherent timing and explicit source-record boundary probabilities.

## Authentic source and labels

Reuse publisher-verified BPI2012 and the established chronological FIT and
exposed DEV split. Original gzip SHA256:
`5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3`;
DOI `10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f`.
4TU General Terms of Use apply; raw source stays external. The source loader
retains publisher MD5/size/SHA verification and a pinned byte-identical mirror.
Original TEST receives no new inference. Do not retry blocked BPI2018.

All vocabulary/statistical priors fit only the 7,096 FIT cases. Original
future event index and original timestamp differences give h=1/2/4/8 targets.
An h-th future original record beyond the case's recorded suffix is labeled
**end-of-record (EOR)**, with **no timing target**. This is a finite logged
trace boundary, not verified completion, cancellation, timeout, interruption
or success. Collection censoring cannot be identified from absence alone.
The model learns a record-availability proxy; no causal stop policy is claimed.

Include last recorded events during training rather than dropping them.
The same observed prefix can target a real next record at h1 and EOR at h8.
Source-file ingestion for existing chronological partitioning is disclosed.
No future length, target, event or timestamp enters the model input. Stable
case identity is an external routing key, never a learned ID embedding.

## Architecture and joint probability

Retain native elapsed-time fast64/slow64/8x8 associative state. Project observed
state, associative query and current event to a 96D latent. Learned horizon
embeddings and bounded gated residuals produce future latent states, decoded
by a shared width160 SwiGLU. This is direct factual horizon forecasting;
predicted events are not substituted for observations or committed to memory.

For each horizon:

    P(EOR) = 1 - S_h
    P(event=a, D) = S_h * P_h(a) * [p0_h * delta_0(D)
                      + (1-p0_h) * sum_k pi_k LogNormal(D; mu_hk,sigma_k)]

S_h is a product of learned conditional survival factors. p0_h decreases
through nonnegative log-hazard increments. Two mixture components share
weights/scales across horizons and have nondecreasing locations. Scales are
bounded in [.1,2] log-seconds. Thus mass normalizes; record survival cannot
increase, and conditional waiting quantiles cannot reverse over increasing
horizon. These are model constraints, not guarantees of prediction usefulness
or censoring-free population truth. Future marks remain distributions, not
verified intervention choices. Current FIT history prior conditions h1;
FIT multi-horizon mark frequencies condition later heads. Their deployment
state is external to neural parameter counts and remains repository-owned.

The same normalized clock defines zero-delay probabilities, likelihoods and
quantiles. There is no disconnected unconditional quantile head and no posthoc
temperature or CQR in this cycle. Train joint existence/mark log loss plus .1
time/occurrence proper scores and .1 joint categorical Brier. The target-only
positive log-time Jacobian is omitted for optimization and included for
reported original-seconds density scores. Reuse lognormal-mixture point-process
and predictive-process methods; see FAIR_COMPARISON.json. No generic novelty
claim is made for those established methods.

## Training and developmental readout

Six complete FIT epochs, batch32 real cases, full32-event truncated BPTT,
AdamW lr.001, weight_decay.0001, clip1. Carry and detach all native states at
chunk boundaries; reset at a new case. Fixed length-bucketed schedules per
seed. No outcome-based search, early stopping, weight or seed selection.
There are **126,285 distinct observed FIT prefixes** and six repeated passes:
757,710 prefix exposures / 3,030,840 horizon-target exposures per model.
These are not independent new observations.

REPORT is the existing last925 DEV cases, already exposed historically.
TEMP/CQR roles are retained in the identity manifest but are unused reserves;
no posthoc parameters fit there. Query ages are 2/6/16/32/64 observed events,
at true reached positions, never known-length midpoints. Include every
REPORT case reaching a query, including a last recorded observation; future
EOR is a scored proxy, not an invented future activity/time. Cases that never
reach a query are explicitly outside that readout. Preflight support is
925/555/165/52/2 cases, respectively. Age64 is therefore sparse engineering/
development evidence, never memory qualification. No age128/256 claim.

Report each age/horizon separately: reached cases, actual future-record and
boundary counts, joint activity/EOR log loss, accuracy, multiclass Brier,
boundary Brier, and conditional activity accuracy/loss. Timing is scored only
where actual h-th future records exist: proper original-seconds likelihood,
median MAE, interval score, coverage and width distribution. No EOR prediction
is counted as a correctly forecast real activity. Repeated ages/seeds are not
independent cases. No causal, cross-site, autonomy or commercial qualification.

## Preservation, numerical contract and checks

Freeze all five original model/optimizer/RNG checkpoints in Git before REPORT
scoring. Bundle: `research/replay-bundles/ME072/`, approximately12MB budget.
Preserve learning traces, actual updates/exposure/CPU, config, exact frozen
preprocessing, source identity and every original model. No selected-best
checkpoint differs from last; epoch saves retain the latest recoverable state.
Original full marked-distribution parameters and quantiles, scores and SHA
index are committed before independent clean GitHub replay. Source remains
lawfully external; inability to retrieve it blocks full score replay.

Freeze primary log-output tolerance at abs1e-5/rel1e-5 and aggregate metrics at
abs1e-4/rel1e-5 before fitting. Invert the same clock CDF in float64 using64
bisection steps. Preserve failed outputs and restore original score bytes;
do not loosen gates after seeing mismatches. Generated fixtures check joint
normalization, monotone horizons, inverse-CDF agreement, causal availability,
carried state, finite gradients and no fabricated time after EOR. Fixtures
cannot establish intelligence. Independent original-weight real score replay
is required before this model study is called complete.
