# DEV-016 — Delayed maintenance benchmark and validity audit

**Status: benchmark implemented and locally replay-audited; learned phase NOT RUN.**

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
The inherited full test suite was not run. No learned checkpoints were trained.

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
