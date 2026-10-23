# ME071: build a stronger persistent native event model

The owner explicitly prioritizes raw predictive capability and integrated
architecture development over attribution or another baseline study. This
registration trains **one architecture**, five seeds 71001–71005. No new GRU,
ablation model or architecture-winner comparison is run. The new explicit
`capability_development` validator lane checks authentic source, original
storage, distinct seeds, budgets and final exposure receipts; it rejects
multiple architectures. Existing matched-comparison rules are unchanged.

## Source, access and prediction task

Reuse publisher-verified original BPI2012 and the existing FIT cohort and
FIT-only five-event prior from ME070. Source SHA-256:
`5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3`.
Original DOI: `10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f`.
The original publisher terms are 4TU General Terms of Use. Raw event records
remain external; source acquisition/hash checks are preserved in ME069's
source loader. Do not retry the blocked BPI2018 source in this cycle.

Train on all genuine next-event/time targets within each eligible FIT case,
six complete epochs, AdamW lr .001, weight decay .0001, clip norm 1, batch 32
cases, backward chunks of 32 events. Length-bucketed real-case batches and
their deterministic per-seed schedules are frozen before fitting. Never mix
states across cases. Carry **all** native fast/slow/associative values between
chunks with stop-gradient, reset only at the start of a new case. No future
events, case IDs or future commands enter the neural feature representation.

The existing DEV is already exposed historical development data. Divide it
chronologically into first quarter TEMP, second quarter CQR, last half
REPORT. Different cases fit class temperature and waiting-interval correction.
REPORT is a development readout, not untouched confirmation. Predict after
exactly six observed events when a next record exists, without hindsight
midpoint selection. REPORT cases too short for this query have no target and
are counted as ineligible. Original TEST receives no new inference or tuning.
Ingestion of the single source file for the existing split is disclosed.

## Integrated model

Retain native elapsed-time decay, bounded 64D fast and 64D slow state and
8x8 associative delta memory. The earlier capability map's 32D numbers apply
to older robot variants, not ME069/ME071. Expand the shared SwiGLU decoder
to width 128. Combine the existing repeat/transition/neural distribution
with the FIT five-event history prior using a learned log-linear blend.
This is useful conditioning, **not** a mathematical loss floor or safety
guarantee. The inherited count table's path, hash and deployed bytes are
charged separately from trainable neural parameters.

Use .05 label smoothing. Fit a scalar class temperature on TEMP only using
a fixed grid in [0.1,10] including the identity (161 distinct candidates). No neural
weight selection, early stopping or architecture sweep is allowed. Equal
accuracy and different NLL do not diagnose overconfidence by themselves;
report multiclass Brier and ECE alongside log loss.

Replace the Gaussian waiting-time parameterization with three **ordered
unconditional log1p-second quantiles**, levels .05/.50/.95. A nonnegative
median, a bounded fraction below it and a positive upper increment prevent
crossing. Joint loss is smoothed next-event CE + .3 mean quantile pinball
loss + .1 Bernoulli score for positive versus tied-zero delay. The occurrence
probability and three quantiles are separately supervised summaries, not a
new single full waiting-time density or neural point-process intensity.

Fit split CQR on distinct CQR cases in log1p seconds. Use the finite-sample
rank ceil((n+1)*.9) and a nonnegative correction, then evaluate seconds in
float64. CQR can widen intervals; no 50–80% shrinkage is promised. Its usual
finite-sample marginal coverage statement requires exchangeability, which
chronological process drift does not guarantee. Report actual coverage and
width distributions, positive-gap MAE, pinball and proper interval scores.

## Additional execution capabilities and supporting tests

Provide an inference entity-state bank with independent state per routing
key, explicit reset, ordered timestamps and pure forecast queries. Addressing
is an external verified identity boundary, not a learned entity embedding.
Interleaving equivalent streams must reproduce their isolated state values.
This does not prove learned separation of unlabeled subprocesses.

Provide masked-observation assimilation that accumulates elapsed time across
missing records. Supporting tests hide four intermediate observations and
check that hidden values cannot affect state; available history priors must
also omit hidden events. No missing event is magically reconstructed, and
GRUs are not assumed to collapse. Generated fixtures are engineering only.

Detached carry expands executed forward history, not backward credit beyond
32 events. BPI2012 supplies little >64-event support and no DEV/TEST >128;
neither this code nor synthetic long fixtures qualify 256-event memory.
The ME070 broad-memory rejection criterion is preserved for a later genuinely
source-qualified comparison; this single-model development makes no such claim.

## Preservation and checks

Publish five original model, optimizer, Torch/NumPy RNG and schedule states
under `research/replay-bundles/ME071/checkpoints/`, approximately 8MB total.
Each fit stores the latest recoverable epoch checkpoint; no best/early-stop
checkpoint is selected. Freeze all final original weights in Git **before**
TEMP/CQR fitting; commit calibration before REPORT inference. Preserve the
original class log-probability/log-quantile arrays, scores and every fit.
Replay original weights and all REPORT predictions in an independent GitHub
clean checkout, without training or calibration refitting. Record final
resource/exposure gate, complete restore/scoring receipts and failures.

Numerical contract fixed before fitting: per-element primary log-output
absolute/relative tolerances 1e-5/1e-5; aggregate metric absolute/relative
1e-4/1e-5. Seconds are derived in float64. No bitwise cross-CPU guarantee.
Do not replace original scores with replay outputs; retain failures.

Prior methods: Guo et al., ICML 2017 temperature scaling;
Romano et al., NeurIPS 2019 conformalized quantile regression;
Dai et al., ACL 2019 segment recurrence. These are reused methods, not TAC
discoveries. TAC remains a native recurrent predictive architecture, not
an SSM/Transformer conversion, policy optimizer or language model.
