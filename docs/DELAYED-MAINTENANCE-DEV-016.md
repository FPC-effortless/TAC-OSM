# DEV-016 — Delayed maintenance benchmark and validity audit

**Status: benchmark locally audited; learned CPU workflow implemented and awaiting GitHub execution.**

Parent: DEV-015 commit `2433f494a717518170d35ea71ab4a36d03e530dd`, PR #171.
This is a new development instrument, not evidence that TAC-OSM memory works.

## Pre-experiment evidence gate

Inherited L0: DEV-015 supplies a differentiable 768-scalar memory-to-action
path and saved-checkpoint replay, but no robust writer improvement.
Unknown: whether finite learned state supports delayed interventions in a
different temporal process. This phase creates an L1 environment-validity row.
Later learned-model action and future-state contrasts would create L2/L3 rows;
none are claimed here. Another XOR/address coordinate experiment would not
resolve this blocker.

The machine-readable contract was written before the local grid audit.
It is a development protocol, not an externally timestamped preregistration.

## Environment and information boundary

Eight maintenance entities carry binary failure states and generation counters.
`inspect(entity)` exposes only the selected entity. `wait` exposes no state.
`service(entity, value, delay)` changes that entity only at maturity. An intervening
external change increments its generation and invalidates an older service.
Changes precede service maturity at the same tick. Event ticks are public.

The policy-facing immutable observation contains only tick, entity, kind and
observed value. Expected outcomes, hidden generations and pending interventions
remain environment/evaluator objects. `Case` is an evaluator record, never a
model input. Pre-action predictions are committed before fresh inspection;
the verifier compares that prediction with the public post-action receipt.
The verifier is hand-designed, not learned.

Opposite-past pairs share the present query and candidate action. Roles are old
retention, recent acquisition, and cancellation of an older service. Each role
has balanced binary wait-future targets. Both wait and zero-valued delayed
service are simulated to their horizons; the optimal maintenance action is wait
when healthy and service when failed, with unnecessary service disallowed.

## Frozen validity audit

3 procedural seeds × 3 histories (32/128/512) × 3 delays (4/16/32) ×
3 roles × 16 pairs × 2 partners = **2,592 episodes**.
These are generator seeds, not independent learned-model seeds.
History denotes the initial observation/distractor budget; intervention and
query-horizon events add work. Actual observation count is recorded per row.

Measured locally:

| Endpoint | Result |
|---|---:|
| Schema wait-future accuracy | 100% |
| Balanced constant/reset control | 50% |
| Schema service-future accuracy | 100% |
| Schema optimal action accuracy | 100% |
| Schema post-action failure rate | 0% |
| Fixed schema state | 16 scalars |
| Mechanism/interface tests | 9 passed |
| Exact source/grid/row replay | 2,592 episodes passed |

The schema reference is a hand-coded indexed sufficient statistic. Its 100%
score validates solvability and provides a strong finite-state comparator; it
does not establish learned representation or TAC-OSM capability. Constant reset
means a fixed prediction, not a reset neural checkpoint. Work remains O(H)
for streaming ingestion. No memory- or compute-efficiency superiority is claimed.

## Limits and next learned phase

### Amendment before learned training

The original 2,592-episode validity result above describes commit
`ad5729cf82ee8607874fda0bf6b3a9217010b360`. It is preserved as a historical
instrument result. Its GitHub validity run 38530240670 passed. The inherited
full-suite workflow failed because the development JSON did not conform to the
canonical contract schema and the mutable suite count had not been updated;
both defects are repaired in this follow-up.

V2 adds a fourth `pending_cancel` role before any learned training: both partners
initially inspect the same failed target, and schedule the same repair. One
historical reset invalidates that repair; the other partner changes a different
entity. At the shared prediction horizon, the repair matures only in the first
partner. Current-value recall alone cannot solve this condition. The scheduled
due tick is public, with no hidden generation or outcome in policy features.
The hand-coded reference now tracks five arrays and one public tick (41 scalars),
with at most one pending intervention per entity in the benchmark schedule.
The revised validity grid has **3,456 episodes**, with the same 100% schema and
50% balanced constant-control accuracies, checked locally before learned runs.

The learned interface streams 14 public features (8 entity indicators, 4 event
kind indicators, tick and scheduled due tick scaled by the fixed constant 2048)
and the public binary payload. `wait` has a zero payload plus an explicit wait
kind. Learned linear event/query maps translate to the inherited 12-dimensional
memory interface. Only a 4×8×24 tensor persists. Padding is masked. Candidate
queries contain entity, action kind, current tick and horizon, not target/role.

All four arms inherit the authenticated DEV-013 aligned_aux writer/read/CASM
weights per seed, start with identical total tensors, and train for 256 updates
of 8 episodes (all four roles per update), AdamW .002, weight decay .0001,
gradient clip 1. Frozen writer also freezes its new event map; query maps,
CASM reader and action head learn in every arm. Joint arms train the writer and
event map. All arms train the evaluated future head on true outcomes through
detached CASM features, preventing an untrained-head comparison. A second matched
auxiliary head sends true future gradients in joint_predictive, random future
gradients in joint_shuffled, and no auxiliary feature gradients in joint_task.
True wait-future labels correlate with optimal actions; service outcomes are
simple. This is additional offline supervision, not autonomous discovery.

Native sparse CASM executes for both candidate actions. New heads and adapters
increase model capacity relative to the parent, but capacity and initialization
match across these four arms. Trainable parameter counts and timed training work
are reported; no equal-FLOP training claim is made. No checkpoints are silently
substituted if the exact parent archive fails authentication.

GitHub Actions runs the full suite, four-arm smoke and saved-checkpoint replay
before the three model seeds (43/59/71) run independently. Each model evaluates
1,152 episodes (3 histories × 3 delays × 4 roles × 16 pairs × 2 partners),
for 12 checkpoints and **13,824 model-episode decisions**. Reset and swapped-state
diagnostics use the same weights and are not separately trained baselines.
Raw logits, pre-action classifications, realized post-action failure, public
verification, capacity, observation counts and sparse operator calls are saved.
Evaluation replay regenerates every row from saved checkpoints; it does not
reproduce SGD or constitute independent scientific replication.

The fixed schedule uses scripted inspection and intervention policies. The
environment supports action-dependent sensing, but autonomous exploration has
not been evaluated. Query horizons guarantee no new exogenous changes: this is
conditional prediction, not prediction of unknown stochastic shocks. Service
outcomes are simple and should be treated as a diagnostic. Binary targets and
opposite-past pairing retain similarities with prior memory tasks; the new
generation-guarded dynamics do not alone demonstrate domain generalization.

The next phase must adapt public event/action features to the actual TAC-OSM
writer and conditional executor, count adapter parameters, and keep every
recurrent policy tensor within 768 scalars. Train four byte-identical starts:
frozen writer, joint task, joint predictive supervision, joint shuffled labels.
Use the fixed training schedule and decision gates in `contracts/DEV-016.json`.
True future outcomes are offline labels only. Evaluate native pre-action future
and optimal-action predictions before verifier receipts or repair; save and
replay every checkpoint and row. Include independently trained finite-state
baselines before architecture comparisons. The current contract is an initial
development design; adapter/interface details must be frozen before training.

No PyTorch runtime is installed locally and shell HTTPS access to the configured
proxy failed. Repository source was inspected through the GitHub connector.
This directory is a patch overlay at the pinned parent, not a complete checkout.
The inherited full test suite will run in the CPU GitHub workflow. Results and
checkpoint availability are only established after that workflow completes.

## Reproduce the completed phase

```sh
python -m unittest discover -s tests -p 'test_delayed_maintenance_development.py' -v
python scripts/run_delayed_maintenance_dev.py
python scripts/run_delayed_maintenance_dev.py --replay
```

The report records source/contract SHA-256, all raw episode rows, actual
observation counts, state capacity and independently simulated service targets.
Replay checks the complete deterministic report, including source hashes.
It verifies evaluation regeneration, not training or independent replication.
