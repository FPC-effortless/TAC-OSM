# DEV-061: memory isolation before architectural expansion

The implementation tests entity-local recurrent memory with the existing learned
CDL-style retrieval, neural CASM, action-consequence heads and learned policy.
It does not insert a repair planner, latest-state action script, stop threshold,
or retry fallback into a TAC policy. An independent public-rule controller and a
public-feature MLP remain explicitly labeled controls.

This follows the bounded Karpathy autoresearch cadence in
[program.md](../research/autoresearch/program.md): register, commit, fit the fixed
matrix, preserve failures, restore and independently execute, then keep or reject.
It adapts that cadence to this CPU research program; it does not run the upstream
GPT trainer or claim the upstream project's performance. The protocol was first
committed at `86425fd`; fitting code and the completed protocol at `1e27956`.
The final results, including failed promotion gates, belong in DEV-061-RESULTS.md.

## Learned components and declared priors

| Surface | What is trained | What is fixed |
|---|---|---|
| Admission | Public signal/type classifier | Four public event types; pretrained then frozen |
| Generic storage | Addressing and event encoder | 32 slots and binary decoder |
| Entity storage | Shared recurrent GRU update | Public categorical-ID routing; 8 entity, 8 relation, 16 episodic slots |
| Episodic retention | Optional keep gate from event and write-time entity prefix | FIFO replacement; hard gate at inference |
| CDL-style read | Queries, keys, values and ranking | Two heads, top-four retrieval; scans the bounded bank |
| Neural CASM | Conditional operators, selector and halting scores | Three operators and two composition depths |
| World prediction | Two action-conditioned latent steps; outcomes, completion, costs | Fixed action/event/query feature API and training simulator |
| Controller | Action, correction and stop scores | Argmax over supplied public candidates; external action budget |
| Verification | None in the policy | Separate simulator/native execution audit |

“Learned policy” describes where decisions originate. It does not mean learned
entity discovery, raw text perception, symbolic program synthesis, unknown action
vocabularies, or an autonomous discovery of simulator laws. Public-ID routing is
an explicit structural prior, as requested by the memory-isolation experiment.
The learned agent never calls the public-rule comparator.

## Controlled matrix

Seven arms receive the same controller-training snapshots and public interface:
`generic`, `entity`, `entity_aging`, `entity_retention`,
`entity_aging_retention`, `structured_gru`, and `public_mlp`.
The four entity variants form the aging/retention factorial. Controller losses
are shared across the TAC variants. Structured GRU removes CASM before fitting;
it retains the other prediction and retrieval machinery. It is not a plain
single-cell GRU policy. The MLP deliberately has a public latest-state feature
parser and actor-focused losses, a strong task-specific control with an explicit
optimization/representation advantage.

The parameter budget is matched within 5%. All four entity factorial cells use
width 47, so aging/retention contrasts hold that width fixed. The generic model
uses width 48 and structured GRU width 61; their recurrent-state capacity and
FLOPs are not identical. Exact counts and tensor-state sizes are recorded in
`research/dev061/model-budgets.json`. Admission pretraining has the same update
count and type-label distribution; generic addressing consumes extra random
samples. Its exact synthetic pretraining samples differ from entity admission.
The main controller snapshots are identical across arms.

Each arm gets two learning-rate pilots on seed 9100 and ten fresh fits on seeds
9101–9110. Admission: 600 updates; belief/read warmup: 256; controller: 1,200.
No test checkpoint selection or favorable-seed selection is allowed. CPU caps
and unsuccessful fits are retained; an incomplete matrix blocks promotion.
“Fixed updates” is an exposure budget, not proof of optimal convergence.

## Current belief, evidence and retention

Entity updates modify only the routed current-state slot. A separate episodic
store retains encoded observations. This prevents *write interference* between
entity slots; it does not guarantee correct learned beliefs or prevent learned
retrieval from choosing stale episodic evidence. Those are separately measured.
The current-state probe is auxiliary and does not feed an explicit probability
vector directly into the policy. That outcome-to-policy intervention remains
DEV-064; this study changes memory construction first.

Optional retention deletes one retained episodic item during training with entity
states held fixed. Its target combines the next supervised prediction/decision
loss change and true expected one-step action-risk regret. Gate inputs use the
entity prefix at the record's write time. Future/private labels supply training
losses only. The target is a local model-counterfactual proxy, not measured
long-horizon causal regret. Only already-retained items are sampled, so its
selection bias must be acknowledged. DEV-063 remains open.

All admitted signal events still update recurrent entity state. A skipped
*episodic* write is not a skipped current-state update. Inference really skips
the episodic encoder when the keep gate closes, but pays the gate and recurrent
update costs. Admission still scans every observation. None of this establishes
C5 indexing or physical DRAM savings.

## Causal diagnostic and metrics

Frozen DEV-058 checkpoints remain byte-for-byte intact. A neutral wrapper first
passes exact learned-read equivalence tests. It then compares original learned
reads with forced latest status/latest relation, and oldest status/the same
latest relation, using only retained public records. These interventions change
read concentration and addressing. Latest-minus-oldest tests freshness within
that forced format; neither contrast alone identifies the universal cause of
all stopping failures. Missing latest evidence cannot be conjured from hidden
state.

The primary outcome is the fraction of ten seeds **strictly above 95%** clean
completion, with paired seed tables and the fraction passing every N/H cell.
Report physical repair separately from valid stop/completion, early stops,
healthy extra actions, missing stops and censored healthy-to-stop delay.
Completion and stop-probability Brier/ECE describe held-out calibration; proper
training scores do not guarantee calibration or perfect empirical agreement.

Fixed-query Brier/error and retrieval diagnostics use shared histories; visited
state statistics also depend on each policy's trajectory. `fixed_rows.latest_hits`
in raw evidence is a preferred supervised-slot hit, which can differ from
latest-time correctness when duplicate latest records exist. Analysis independently reconstructs full public fixed histories and measures
latest-time correctness, staleness and latest-record availability against those
histories, including when the newest record was dropped from memory. Raw
`fixed_rows.stale` and `latest_available` concern retained-bank targets only and
are preserved as such; they do not replace full-public-history diagnostics.
Reported controller work counters are declared policy-path counts, exclude the
auxiliary current-state read, and do not count every padded tensor operation.
CPU timings include those auxiliary computations; profiler-supported FLOPs are
an incomplete operator estimate. Tensor state bytes include public metadata.
DRAM traffic remains UNKNOWN without hardware counters.

## Scope and next experiments

Clean primary cells keep N=2/4/6 and H=32/128/512/1024 separate. Unseen categorical
IDs and 10% initial/receipt-bit noise are separate OOD stress families. The old
receipt schema reveals which entities changed and omits explicit action
provenance in memory updates. Thus these noise families do not establish valid
Bayesian belief tracking or learned value of information. Missing initial states,
continuous identity association and unseen cancellation histories are not trained
or validated here. The latter two are explicitly NOT_SUPPORTED in this API.

DEV-062 must first establish provenance-aware public-only belief updating and
paid inspection under genuine partial observability, comparing to exact public
posteriors and information-matched filters. Then DEV-063 retention,
DEV-064 outcome interventions and CASM attribution, DEV-065 training strategy,
and DEV-066 integration may proceed under separate frozen protocols.
[CASM's claim policy](../research/dev061/casm-claim-policy.json) names DEV-064,
`DEV-064-CASM-LONG`, and `public-action-consequence-chain-N4-6` as the deadline
for superiority over a tuned MLP on the **same** learned belief/outcome features.

Reusable residual TAC blocks at depths 2/4, later 8 if justified, are a separate
future experiment with Transformer/GRU controls. Network depth remains distinct
from action-imagination horizon. No stacking, JEPA pretraining, natural-language
interface, C5 indexing, or commercial benchmark was silently added to DEV-061.
Archived DEV-015–060 studies and original exports are preserved.

## Implementation qualifications fixed before learned holdout analysis

The retention risk target is explicitly `(candidate_cost + 20 * expected_remaining_faults) / 20`.
`candidate_cost` already includes terminal faults for stop, so a broken stop gets
an additional fault penalty here. This is an asymmetric **risk surrogate**, not
literal charged economic regret; it must not be reported as the latter. The
formula was fixed in the committed fitting source, not changed after outcomes.
Economic acceptance always uses actual charged costs and final physical faults.

Exactly tied validation rankings retain the first pilot in recorded completion
order. That order is archived in `tuning.json`, along with `selection.json`.
Replaying those records reproduces selection; rerunning the pilots on another
scheduler need not resolve an exact tie identically. No test outcome resolves a
tie. Future protocols should explicitly use a deterministic learning-rate tie
break. No current fits or selected rates are replaced after the fact.

The registered generic wrapper stores raw public metadata for diagnostics. It
unnecessarily concatenates that metadata even on a closed admission gate. A
separate inference-only subclass, `InferenceMemoryAgent`, returns the existing
state on that path. State and action equivalence are tested; weights and fitting
source remain unchanged. The portable generic artifact uses this optimization.
