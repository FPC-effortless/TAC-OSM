# Next-stage proposal: masked MTSK and verified decision learning

Status: proposal, not an implemented or registered completed experiment. Keep
optimization and PNDS conformance work in progress. Do not introduce locality,
timescale diversity, global relations, teacher supervision and active sampling
simultaneously: their contributions would be inseparable.

## Entity-local MTSK candidate

Maintain per-entity memory AND per-entity responsibility/temporal-prior state.
Gather only the publicly addressed entity, propose its MTSK transition, then
scatter only that entity's accepted augmented state. Unaddressed entities must
remain bitwise unchanged, including decay, responsibilities and any persistent
normalization state. A rejected experience commit must preserve the whole
augmented transition. Masking only the additive write is insufficient because
SwiLAMTSK._step applies decay independently of its update_gate. Existing native
MTSK, VerifiedMTSK and masked-local TAC remain separate untouched controls.

Invariants before training: mixed entity IDs within a batch; hundreds of unrelated
events; rejected commits; non-finite proposal rejection; read-only queries; entity
permutations; gradients confined to the addressed state; reset independence.
Shared learned weights do not imply shared recurrent state. Query timestamps may
change predictions while protected memory remains unchanged; do not confuse
state isolation with prediction invariance as real time passes.

Strict public entity assignment is an assumption, not learned entity discovery.
Fast/medium/slow priors do not establish semantic specialization. Measure event
retention, replacement after cancellation/reissue, responsibilities, per-timescale
ablation and actual predictive/decision consequences.

Capacity requires a real kernel redesign. With DEV-032's d_model24, four heads
and six experts, the MTSK memory tensor alone uses864 scalars; eight copies
use6912 before auxiliary state, not the768-scalar TAC budget. The streamed
responsibility tensors add another144 scalars per copy (1152 across eight). Decouple memory key/value rank from shared encoder and
predictor width, and count responsibility state as well as memory matrices.
Freeze exact layouts before outcomes. Report parameters, optimizer state,
measured compute, latency, memory and total tuning cost. If simultaneous state
and compute matching is infeasible, provide separate matched controls and state
the residual confound; do not call nominal state equality compute equality.

Required references: native MTSK, masked-local TAC, schema41, explicit table plus
learned head, tuned GRU/delta. For a locality-timescale interaction claim, add a
matched MTSK factorial: global/local addressing crossed with tied/independent
retention parameters. Equal initialization alone only tests retention priors;
untied parameters can subsequently learn distinct regimes. Tied retention is
also not proof of a single effective timescale when responsibilities vary.
Frozen inference ablations and equally tuned retraining answer different questions.

The optional global relational memory is a separate arm with a separately charged
state budget. The current independent-entity generator cannot demonstrate useful
cross-entity causal reasoning. A relational task needs observable interactions,
new namespace, oracle specification and independently frozen evaluation first.

## Simulator-verified decision supervision

Keep future-state prediction loss and test an additional action/cost objective.
Train on student-visited training histories; query a trusted copied simulator for
action outcomes. The student receives only allowed public observations. Privileged
world state may construct teacher labels during training, but must never enter
student features, evaluation adaptation or sealed test selection.

Compare state-only learning, finite-action teacher supervision, and cost/regret
supervision under equal training examples and oracle-query budgets. Treat
cost-equivalent actions as a set; one teacher action is not the unique correct
label. Audit teacher actions against environment consequences. Report regret,
prediction accuracy, cancellation, absolute failures, unnecessary services,
inspections, action costs and training/inference costs separately. Where public
histories alias hidden states, privileged teacher agreement may be unattainable;
do not attribute unavoidable disagreement to memory failure.

An action teacher alone can encourage shortcut imitation without a world model.
Test counterfactual forecasts independently. Active querying versus passive
sampling is another experiment with equal oracle interactions, not a simultaneous
change to memory and policy losses. Generated simulation evidence is still data.
Charge sandbox calls and collection costs separately from deployed physical costs.

## Source verification and terminology

The cited [ActionOPD paper](https://arxiv.org/abs/2610.19820) was reachable and its
official repository was inspected at
[`d447bdf2ad3158ae7c315cb96ce0a180f94459b1`](https://github.com/thu-nics/ActionOPD/tree/d447bdf2ad3158ae7c315cb96ce0a180f94459b1).
Its README describes on-policy semantic action agreement with black-box language
teachers, reports3.3–6.9 percentage-point four-benchmark gains and explicitly notes
lower repeated-success pass-four reliability than the distillation baselines.
These are author-reported results, not independently reproduced here. Agreement
is not guaranteed correctness. Direct cross-entropy on finite simulator actions
is an inspired control, not a reproduction of ActionOPD's semantic REINFORCE method.
No external implementation was installed or executed.

The retinal result motivates an optional event-order channel, not evidence it
improves neural memory. The referenced videos/transcripts are not attached to
this task; no reproduction of their implementation or numerical claims occurs.
No Veritas repository or deployed system is changed by this proposal.

Before scientific execution, freeze the selected candidate's executable protocol,
validation-only tuning and equal search budgets; use H128/H512, registered physical
costs and all-role metrics; then freeze recipes before untouched evaluation and
fresh-seed confirmation. Twenty seeds are a reliability target, not a guarantee
of power. Larger evaluation and discovered/unseen entities remain separate tasks.
