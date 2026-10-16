# DEV-029: adequately trained baselines and complete-system integration

Registered 2026-10-16, before pilot outcomes. DEV-028 module-initialization
attribution is deferred until optimization controls establish whether it is
an informative next experiment. Historical protocols and results are preserved.

## Current scientific position

TAC-OSM investigates learned predictive memory under long-horizon interference
in synthetic action-conditioned environments. Entity-indexed memory has
produced high development performance, but fresh configurations expose major
reliability failures. Current experiments do not establish superiority over
tuned recurrent, structured, or explicit-state baselines, learned entity
discovery, or capability of the complete TAC-OSM system.

DEV-023's 98.54% short result remains recorded adaptive development evidence.
DEV-024's 74.58% long result remains a failed fresh replication (median 68.75%,
range 51.04–95.83%, two of five seeds above 80%). Neither replaces the other.
DEV-027 identifies a whole-model initialization effect in its tested factorial;
it does not establish an optimization ceiling. The old 216-update recipe
exposed 432 long-history episodes. Chance performance under that recipe is
not evidence that an architecture cannot represent the solution.

The immediate question is **learned memory under interference, compared against
sufficiently trained and tuned baselines**. Supplied entity partitions and
known transition rules must be identified explicitly wherever they are used.

## Track A: mechanism and optimization

The machine-readable registration is
[protocol.json](../research/dev029/protocol.json). It registers native TAC,
masked-local TAC, learned schema41, GRU768, equation-level gated delta, and a
classical event table with a learned prediction head. The exact-rule oracle
is an upper-bound/solvability reference, never a competing learned model.

Each arm receives two learning rates (0.0005, 0.002), two horizon curricula,
three tuning seeds, and nested checkpoints at 216, 648 and 2,160 updates.
The curriculum switches from H128 to H512 after update 720, independently
of the checkpoint being evaluated. Other settings and training receipts are
matched. Every registered trial must finish: no pruning from early scores.
Equal trial opportunities and updates are **not equal FLOPs**. Parameters,
CPU and wall time, events, state bytes and inference time must be reported.
Measured FLOPs and total tuning/inference cost remain required before an
architectural efficiency claim; neither is inferred from state size.

New train/validation/test namespaces replace reuse of known development grids.
Only validation chooses recipes. `select_recipes` fails closed unless the
complete six-arm, four-recipe, three-seed, three-checkpoint matrix is present;
it rejects test and smoke records. Final-checkpoint mean primary accuracy
selects, then all-query CE and registration order resolve ties. Freeze a
separate recipe manifest in a commit and audit matrix completeness before
unlocking a separate test evaluator. The present pilot has **no test path**.
A selector is an integrity check, not a security boundary against code changes.

After freezing recipes, train twenty fresh initializations (1001–1020) per
arm with fresh matched streams. Report mean, median, sample standard deviation,
IQR, worst observed score, fraction passing every original gate, and paired
seed-level bootstrap confidence intervals. Twenty seeds do not establish a
population guarantee. Preserve all failures and raw records. Do not treat
within-seed correlated episode counts as independent seed replications.

Economic reporting must include distributions of absolute failures, unnecessary
services, inspections, total intervention cost, and selected minus model-only
cost, with paired seed uncertainty. Keep the historical strict gate unchanged;
report the magnitude of tiny cost differences alongside binary outcomes.

### Bounded first pilot

Within the existing approximately 30-minute batch, run all six arms at one
registered recipe: seed 311, H512, LR0.002, 216 updates, B2. This checks training,
public receipts, validation, raw predictions, physical-cost replay and evidence
capture. It is **not the tuning study**, not recipe-selection evidence, and
not a fresh twenty-seed reliability result. Save model and optimizer states;
continuation requires a subsequent frozen runner and independent audit. An
incomplete pilot must not be described as a completed experiment.

### Classical reference boundary

`EventTablePredictor` stores current bit, generation, eligible due time,
queued value and queued generation for each of eight entities, plus time:
41 float32 scalars (164 recurrent bytes). Cancellation and public entity
addressing are provided by the algorithm. A small shared MLP learns prediction
from these raw statistics and action/query features; it receives no exact
future predictions. It is neither schema41's learned GRU representation nor
an entity-discovery method. CPU work and parameter/optimizer storage remain
additional costs. Multiple simultaneously eligible schedules per entity are
outside this restricted table contract and fail explicitly.

### Structural-generalization stages (not implemented here)

1. Replace direct IDs with indirect but identifiable public cues. Register
   identity permutations, unseen identities and ambiguous cues. First verify
   an oracle can identify the relevant distinction from available observations;
   include irresolvable ambiguities separately rather than call them model
   failures. Separate learned association from supplied indexing.
2. Extend environment, event encoding, query and memory interfaces together
   for training on 8 and testing on 16/32 entities, including entities exceeding
   slots. Current eight-way features, queries and fixed key slices are not a
   variable-capacity implementation. Changing only evaluation entity count is
   invalid.
3. Register transition-rule and relationship generalization after optimization
   controls. Compare against classical tables appropriate to the information
   supplied, and tune all learned comparators with matched search opportunities.

Recurrent Entity Networks (Henaff et al., ICLR 2017), Recurrent Independent
Mechanisms (Goyal et al., ICLR 2021), and Slot Attention (Locatello et al.,
NeurIPS 2020) are relevant antecedents, not implemented baseline claims.
Entity factorization is not a new TAC discovery. Slot Attention alone does not
supply persistent identity tracking. Practical implementations need separate
registered compute and optimization budgets.

## Track B: complete-system integration

Maintain executable checkpoints with explicit boundaries; do not merge research
branches indiscriminately. A component can be preserved, wired, exercised,
causally connected to action, and scientifically useful at different levels.
Random-weight execution and gradient dependence establish wiring only.

| Candidate | Preserved implementation | Boundary and current limitation |
|---|---|---|
| Legacy integrated loop | `builder.py`, `model.py` | State, router, executor, environment, verification and bounded repair execute; action is the router's selected candidate, not executor output; no prospective OSM call |
| Original CASM contract and graph variants | `executor.py`, `casm_runtime.py`, `casm_adapter.py` | Deterministic contract adaptation, local graph runtime, and external-executor adapter are distinct implementations; not interchangeable learned models |
| Learned conditional operators | `development/conditional_compute.py`, `full_plm.py` | Neural operator bank; bounded streaming `decide` omits the reference planner |
| Learned integrated PLM reference | `development/full_plm_reference.py` | MTSK → learned router → learned CASM → OSM prospective action scores → action; post-outcome verifier, repair proposal and gated commit; prior capability failure remains |
| Original CDL teacher | `router.py` and original CDL repository | Runtime teacher stub is not an executable NLL scorer; cheap relational/learned routers do not establish successful teacher distillation |
| Fused multi-timescale specialists | PR #146, `5923ed0bf1aa2b9ae125c98d60dab25ec3cf07aa` | Separate branch with memory, priming, regime, routing, specialists, halting and repair; not present in this branch; no compatibility or advantage assumed |
| Original learned PLM | PR #143, `8792cbd37701b2d33c7c86db1b0514b6e33dd480`; PR #144, `91fec13eb32c1ced27149d046c66f120214485fa` | Preserved research lineage; initial failed capability results cannot be erased by wiring checks |

`check_integration_dev029.py` executes the preserved integrated PLM on synthetic
public-shaped tensors with random weights, counts actual boundary calls,
checks finite action/state/repair outputs, and checks CASM and OSM gradient
paths into action scores. It then supplies synthetic outcome evidence to the
verifier/repair/commit interface. It **does not execute a physical environment
or apply a repair action**, establish true CDL relevance, train the complete
system, or validate component advantage. These gaps are explicit output fields.

Before calling an integrated system validated: provide an actual environment
rollout, persistent state across episodes, prospective OSM comparisons,
action-conditioned consequence evaluation, explicit verification acceptance
and rejection, executed bounded repair, accounted routing/operator/rollout and
repair costs, matched learned baselines and component ablations. Compare graph
CASM, neural operators and fused specialists before selecting a shared executor
boundary. Register frozen seeds and end-to-end decision/cost regression gates.
Memory-only results must never populate these full-system capability fields.

The two tracks reconnect through a frozen public observation/state/query/action/
outcome interface. A mechanism enters the integrated checkpoint only after
compatibility checks and an end-to-end ablation measure its contribution.
