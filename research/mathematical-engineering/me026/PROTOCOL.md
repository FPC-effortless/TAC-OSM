# ME026 — registered execution engineering and ME025 evidence closure

This is a zero-fit engineering extension of ME025. Its completed nine original
fits are downloaded from successful workflow 39521144405, artifact 12311761679.
The immutable original ZIP digest is in contract.json. Do not retrain, add
seeds, reselect checkpoints or open any new cohort. Original ME025 core and
training source remain unchanged, including its negative primary finding.

First verify the original ZIP, nine checkpoint and prediction hashes before
loading responses. Restore the authenticated ME022 snapshot, requiring original
parent checkpoint hashes, exact snapshot SHA and canonical tensor/user/time
fingerprints. Replay all nine models on exactly 5,952 already-exposed DEV
windows. Maximum original probability error <=1e-6 and Brier/logloss/AUC/ECE
error <=1e-7. Reconstruct sampler RNG, enforce original exposures/steps/budgets,
verify FIT-only normalization/popularity and count actual trainable parameters.
No sealed cohort, source download or model fitting is permitted.

The original small MLP reference has only metrics, not saved weights or
predictions. Preserve it as an unverified historical reference; do not silently
refit it. The Transformer comparison remains descriptive; full compute parity
is unestablished. The protocol does not revisit ME025's failed accuracy gate.

Execution change: compile mathematically identical native recurrence and
readout, exposing a correct explicit incremental kernel-native forecast API.
The five rate summaries admit an associative affine segment composition;
derive and test it independently from the state-dependent native recurrence.
Do not claim that native TAC as a whole admits an associative scan. Preserve
fixed 128-event forecasts and all trained weights, including their feature
order and item-popularity lookup. Kernel summaries use only already-observed
feedback. State ownership and the separate causal 32-response prior are caller
responsibilities. Verify streaming/window equality on 32 genuine windows for
each of six native checkpoints. Bound logit error at 1e-5 and probability error
at 1e-6, before any timing. Replay all six standalone exports over the full DEV.

Strengthen constructed causal tests with a NONZERO head and state sensitivity;
a deliberately unmasked attention implementation must fail the leakage guard.
Artificial examples are software falsifiers only, never capability evidence.
Test kernel impulse derivatives, block affine composition, empty-segment
identity, real-arithmetic associativity and floating-point agreement. Synthetic
labels are not trained or scored as a capability experiment.

Primary engineering gate: every native checkpoint's median full-window
batch-128 forecast wall latency improves by at least 10%, with all numerical
gates passing. Benchmark all six checkpoints, batches 1/32/128, three warmups
and nine retained repeats/path. Alternate eager/compiled ordering by repeat.
Report CPU and wall latencies, compile time, raw repeats and hardware. Benchmark
one incremental transition+forecast separately; it processes different work
and must not be sold as an equal-work speedup. No hyperparameter tuning or
architecture accuracy ranking follows. Threads=1; same process/machine/dtype
for each paired measurement. Peak process RSS is not isolated inference memory.
Archive original and derived exports, all results and failures; publish a draft
stacked on ME025, without merge, deployment or promotion.
