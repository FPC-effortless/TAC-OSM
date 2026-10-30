# LANGACQUIRE097: teach acquisition before changing the memory architecture

Prospective internal FIT/DEV research, not independently qualified. All source,
cohort, code, seeds, objectives, budgets and decision rules are committed before
the first optimizer step. No protected repository TEST, model promotion,
independent reward certificate, GPU quota or general-intelligence certification
is authorized by this study. LANGPERSIST096 remains immutable negative evidence.

## Question and reuse of established methods

Can query-focused training teach an existing compact recurrent architecture to
use information acquired earlier, improving real code completion and a separate
task-heldout transformation diagnostic? The earlier 500K checkpoints learned
independently reset short-window prediction; their null results did not isolate
whether acquisition-to-query training would help. This study changes the loss
placement on the same chronological inputs, not the TAC memory equations.

Episodic memory learning, backpropagation through recurrent state and in-context
adaptation are established methods, not proposed TAC discoveries:

- Santoro et al., *Meta-Learning with Memory-Augmented Neural Networks*, ICML2016:
  https://proceedings.mlr.press/v48/santoro16.html.
- Dai et al., *Transformer-XL: Attentive Language Models Beyond a Fixed-Length
  Context*, ACL2019: https://aclanthology.org/P19-1285/.
- von Oswald et al., *Transformers Learn In-Context by Gradient Descent*, ICML2023:
  https://proceedings.mlr.press/v202/von-oswald23a.html.

The unresolved question is whether the exact source-pinned TAC086 graph learns
causally useful acquired information on the registered source data and budget.
A learned query alone is not new: LANGMEM086 already has one. No additional
memory, hidden trained encoder, oracle operator library or architectural graph
is added to TAC in this trial.

## Frozen cohorts and information boundary

The primary data are original licensed Python source bytes at CPython commit
`60403a5409ff2c3f3b07dd2ca91a7a3e096839c7` (PSF-2.0). Ten whole files are FIT;
four different whole files are DEV. Source URL, Git blob, byte count, SHA256 and
license evidence are in `SOURCE_PROVENANCE.json`. These public files were FIT
for historical models; all097 models start from random initialization and never
train on097 DEV. No claim of independently blind evaluation is made.

The lexical cohort contains128 FIT episodes (64 positive,32 negative,32 uniform)
and64 DEV episodes (32 positive,16 negative,16 uniform). Each is an authentic
contiguous64-byte acquisition A,32-byte query Q and4-byte target Y. Positive
cases begin at complete ASCII Python identifiers of length4–24, absent in full
from Q but previously observed in A. Negative identifiers are absent from A and
Q. DEV positive/negative names are also absent from the complete FIT identifier
inventory. Uniform positions have no answer-stratum selection. Selection is
hash-ordered with salt `LANGACQUIRE097-DATA-97001`; windows are disjoint within
each original file. No outcome-dependent reselection or relaxed eligibility.

The secondary diagnostic uses the human-authored ARC-AGI public training archive
at `399030444e0ab0cc8b4e199870fb20b863846f34` (Apache-2.0). Original task IDs
separate24 FIT tasks from8 DEV tasks, excluding every ARC task from096. This is
an authored benchmark, not naturally collected operational outcomes and not
primary real-world capability evidence. Held-out task IDs do not guarantee
held-out semantic rule families or unseen compositions; those obligations stay
OPEN.

ARC presents at least two original input/output demonstrations in A and an
unseen original query input in Q; only training/scoring code sees its official
output Y. Grids have at most6rows/6columns. A uses at most192 bytes and A+Q+Y
at most272 bytes. The fixed lossless ASCII format is rows/columns followed by
row-major color digits, with input/output/demonstration delimiters. It changes
the interface from096's natural-language prompts before any097 fit; improvement
over096 would therefore confound interface, model size, exposure and training.
Only the paired097 objective contrast isolates the registered loss treatment.

The initial128/208-byte ARC proposal had only21 eligible tasks for32 required;
its insufficiency is retained. Before any model observation, source inspection
found35 eligible tasks at192/272, allowing24FIT+8DEV+3unused. A256/336 option
was not selected. No geometry was chosen using model outcomes.

`FIT.json` has152 complete training episodes. `DEV_ACTOR.json` has72 A/Q-only
records; `DEV_JUDGE.json` owns targets, original identities, strata and cluster
metadata. Actor IDs are opaque hashes. File names, offsets, target names/lengths,
operator labels and the scorer never become neural input features. Acquisition
and query use separate fresh processes; the query receives only Q, a complete
checkpoint-bound state and anonymous ownership/clock metadata. A/Y remain
unavailable to that actor. Source authentication, train/DEV separation, frozen
future-only poisoning and source-level feature allowlists precede scoring.

## Forty fitted models and two objective conditions

Use seeds1701–1705 and four independently registered model families:

| Family | Active / deployed learned parameters |
| --- | ---: |
| Unchanged CompactDualClippedTAC086 |100,521 |
| GRU |100,661 |
| Selective SSM |100,404 |
| Causal Transformer |100,484 |

Every fitted model must be in95K–105K; all pairwise gaps are below1%. Live
prefit and final counts, all heads, fixed buffers, complete carried state,
clock/owner metadata, activation/KV storage and process RSS are recorded.
Transformer KV storage grows with the observed prefix; matched weights do not
mean matched persistent memory. No unused padding weights or excluded encoder.

For every family/seed, initialize full-loss and query-loss models identically.
Their random source, source episodes, order, batch geometry, optimizer and
updates match. Use256 AdamW updates, batch4, lr0.001, weight decay0.01,
betas(0.9,0.999), epsilon1e-8, global gradient clip1, no foreach/fused optimizer,
no tuning trials, no early stopping and no checkpoint selection.

The sampler uses NumPy `default_rng(seed)`. Every update samples two distinct
lexical and two distinct ARC FIT episodes without replacement within domain;
order is lexical then ARC. Publish each seed's ordered-ID hash, authentic input
bytes and domain/target exposure. Both conditions receive the same observed
data, including the same teacher-forced training prefixes.

For training, concatenate z=A||Q||Y and predict z[1:] from z[:-1]. End-pad inputs
to4×272 only after authentic bytes. Mask all padded targets; padding must not
alter earlier logits. Compute the same per-position cross-entropy and operation
shapes in both arms. The full condition averages valid chronological targets;
the query condition averages only targets in Y. Do not detach acquisition state:

\[
 S_A=F_\theta(A),\quad
 \mathcal L_{query}=\frac{\sum_t m^Y_t\,
       [-\log p_\theta(z_{t+1}\mid z_{\le t})]}{\sum_t m^Y_t}.
\]

Loss masks change the number and distribution of supervised targets. Balanced
episode counts do not give balanced target weights: longer ARC answers receive
more query-target weight. Publish both observed-input and supervised-target
counts by domain/condition. This is a loss-placement/weighting treatment; no
claim that an effect isolates only retention pressure.

Actual FIT query gradients must reach acquisition writes/state before fitting;
record module connectivity, finite activations/gradients, clipping and nonzero
weight updates. A running optimizer without connected learning is failure.

## Compute gate and finite execution

There are four separate `within_family_objective_comparison` registrations,
each ten models (two objectives×five seeds). `FAIR_COMPARISON.json` is an index,
not a mixed-family single-model loophole or a cross-architecture parity PASS.
Run each family's preflight before its first optimizer step and final gate on
its actual checkpoint/resource receipts.

Same executable graph and dense shapes allow a complete mathematical ATen
operator signature/shape/count comparison of full/query forward and backward
work. Freeze actual FIT profiles for all five seeds before fitting. Every
training step must match that frozen ledger; actual first/warm AdamW ledgers
must match between objective conditions. Opaque operators cancel only within
the same source-pinned model. This establishes the registered same-graph work
comparison, not exact hardware instructions, energy or absolute FLOPs.

Cross-family training FLOPs are not matched. Absolute training FLOPs remain
INCONCLUSIVE rather than invented from a partial matrix-only profiler. Thus no
architecture winner, efficiency superiority or complete equal-FLOP benchmark
can be claimed. A future cross-architecture study needs separately registered
exposure-matched and compute-matched regimes and complete accounting.

Before any real-data model prediction or optimizer step, a constructed
complete-ATen recording profile estimated that the initial512-update plan
would consume46.4 minutes per seed before optimizer overhead, exceeding
the40-minute ceiling. The fixed plan was therefore reduced to256 updates
before source sealing. No cohort, seed, model, endpoint or threshold was
changed, and no trained result informed this budget correction.

The subsequent original FIT-only GitHub profile measured approximately7.1
seconds per TAC update under complete operator recording. Eight256-update
fits in one seed job would take approximately72 minutes, so the five seed-job
layout is not eligible. Retain256 updates and all forty models; do not reduce
learning to fit that layout. Before any optimizer step or DEV result, split
the work into ten objective legs (five seeds times two objectives), each with
the four family fits. This amortizes CI setup while giving every objective
condition the same family-specific and leg-wide limits. Runner/ledger source
changes and the changed runtime recipe invalidate the earlier profile binding;
preserve that attempt and repeat original FIT-only no-optimizer profiling
before enabling any fit. This is a prefit execution amendment, not a change
selected using learning or DEV outcomes.

First run source-authenticated, zero-optimizer GitHub CPU profiling and the
PNDS/Lean checks. Commit the original measured profiles and their exact source,
data and training-recipe bindings. Only then enable the forty fixed fits.
Each of ten objective-leg jobs has a shared40-minute experiment ceiling for
four family fits. Family-specific fit caps are35 minutes for TAC,8 for GRU,
8 for SSM and4 for Transformer, with identical caps for full/query conditions.
The shared leg ceiling still applies when those per-family ceilings would sum
to more than40 minutes. No fit may continue by silently reducing its256 steps.
The legacy `max_seed_seconds` field is retained only for compatibility; the
ten-leg workflow uses the explicit objective-leg ceiling.

The maximum aggregate fit-stage budget is400 minutes, plus at most15 original
preparation/profile minutes and40 evaluation minutes:455 experiment-stage
minutes. CI package installation, data download, startup and artifact upload
are separately measured orchestration overhead, not hidden inside that455.
Each GitHub job has a45-minute outer timeout. Preserve incomplete checkpoints
and failures if a cap is reached; never replace a failed arm or seed.

Runs use the same pinned one-thread CPU FP32 software environment. Record the
actual physical CPU/runtime identity for every fit; separate GitHub virtual
machines may have different physical CPUs. Complete mathematical operator
signatures/counts match within each source-pinned objective pair; this does not
establish identical hardware instructions or FLOPs. CPU wall/process times
are efficiency observations and are not required to match. No GPU quota is
allocated or substituted.

All forty original final checkpoints must seal before any DEV predictions.
Retain model/optimizer/Python/NumPy/Torch RNG, input order, source bindings,
actual steps, partial failures and every checkpoint hash. No failed seed is
replaced. Cold reload and streaming/full FIT parity precede DEV access. A
failed fit or missing final receipt blocks a complete matrix claim.
Before the first optimizer step, the actual live initial-model digest must
match its frozen original profile. Assembly rechecks equality of each full/query
initial-model pair across the ten legs; a seed label alone is not initialization
evidence. All ten leg artifacts must assemble into forty sealed checkpoint
receipts before any DEV prediction.

## Acquired-state and binding controls

On the same frozen DEV cases compare carry, complete content reset, a real
other-owner donor and norm-preserving coordinate corruption, retaining clocks
and legal ownership/bindings. Include every actual model tensor/KV/history.
Donors must have the same domain and observed acquisition length; selection is
hash-fixed without future Y or answer-stratum filtering. All64 lexical donors
exist; only3of8 ARC matched-clock donors exist. Keep the five unavailable ARC
donor rows explicitly INCONCLUSIVE with missing metrics, never pad a KV cache,
silently drop tasks or substitute selected cases. Supported donor subsets are
descriptive only and cannot stand for full-cohort attribution.

For ARC, additionally acquire a format/clock-matched alternative A that cyclically
rotates already observed demonstration outputs across the original inputs.
It reads no query output or future target, preserves acquisition length and
delimiters, and changes input-output bindings. Its query worker receives only
the exported alternative state and the same Q. All eight selected cases have
an informative changed control; any future unavailable/identical control is
INCONCLUSIVE without replacement. Unknown-rule attribution must beat this
binding shuffle, not merely a reset that also removes formatting information.

The lexical fixed raw-history Witten–Bell predictor is a compulsory untrained
reference, using only A/Q and its own generated bytes. Count its allocated ring,
headers, temporary counts, imports/setup, scanning, state, shared priors and
process RSS. The ARC untrained reference copies the last observed demonstration
output from A, with no query Y access or learned operator library. It has exact
grid outcomes only; do not fabricate a probability distribution/NLL for it.
Neither reference is a parameter-matched neural architecture.

## Outcomes, uncertainty and fixed decisions

Query actors output first-target-byte log probabilities and exactly4 lexical
or40 ARC greedy bytes, without gold-length stopping or teacher-forced Y. The
separate evaluator owns all targets and decodes the frozen ARC format. Report
format compliance separately from full exact grid correctness. ARC first-byte
NLL is a format diagnostic, not a reasoning score. No pooled AGI/IQ metric.

The primary family is TAC. All GRU/SSM/Transformer findings are reported without
selecting a favorable family, seed, checkpoint or endpoint. Primary lexical
statistics use the32 positive DEV cases; negative/uniform cases remain
diagnostics. Average five seeds within each case, then bootstrap the four
original code-file clusters5,000times with seed97001. ARC statistics similarly
resample the eight original tasks. Do not count seed copies or bytes as
independent cases. Few clusters make these exploratory internal intervals,
not independent generalization qualification.

All gain signs mean improvement. Within each family:

- Lexical NLL learning requires full-carry minus query-carry NLL >=0.01nat and
  its paired95% lower bound>0. Query reset-minus-carry must meet the same rule.
  At least4of5 seeds and at least75% of original clusters must improve for
  both effects. Donor-minus-query-carry NLL must also have95% lower bound>0.
- Lexical completion utility separately requires positive query-carry minus
  full-carry, reset and donor exact-four gains, each with95% lower bound>0.
  An NLL gain alone is not useful completion.
- ARC learning requires positive query-carry minus full-carry and reset
  exact-grid gains, each95% lower bound>0, at least4of5 positive seeds and75%
  positive task clusters. Binding-specific attribution additionally requires
  those consistency/interval rules versus the format-matched binding shuffle.
  An unavailable required control makes attribution INCONCLUSIVE. All eight
  tasks remain in the denominators; partial donor effects do not replace it.

The joint learning gate requires the lexical NLL/completion and ARC learning/
binding gates. Practical advantages over the raw-history and copy-last-output
references are separate reported hurdles, with full resource accounting; a
joint within-model learning effect does not establish superiority to these
references or cost savings. Missing source, split, information-boundary, actual
size/exposure/compute, gradient or replay evidence prevents a positive claim.

Freeze all thresholds before predictions; preserve negative/inconclusive runs,
per-case outputs, per-seed effects, cluster effects and all costs. No gate is
rescued with uniform-cohort performance, a favorable seed or unregistered task.
Learned verification/repair, false-write acceptance, sustained continual
improvement, unseen semantic rule families and full intelligence theory remain
OPEN regardless of this pilot's outcome.
