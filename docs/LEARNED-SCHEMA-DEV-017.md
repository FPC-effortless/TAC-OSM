# DEV-017 — Learned schema state, pending dynamics, causal memory and decision cost

**Status: frozen pre-training ADAPTIVE DEVELOPMENT protocol; no outcome claimed.**
Parent: DEV-016 PR #172, commit `1083d7ad615ccc1758e68d929f67eeafccb2388e`.
The contract is committed with this implementation before automatic CPU training.
This is a development protocol, not external preregistration or confirmation.

## Evidence gate and research question

Inherited: DEV-016 passed checkpoint replay and showed joint native actions
73.64% versus frozen 59.32%, but the primary reliable-prediction gate failed.
Its joint intervention also changed the event map. Pending cancellation was
only 55–58%, and physical failures alone could reward excessive service.
This experiment is necessary to create new controlled L2/L3 rows: learned
schema sufficient state, unresolved temporal outcomes, memory-versus-encoder
attribution, and physical decision cost under state-dependent sensing.

## Learned 41-scalar baseline

Eight public entity-indexed five-dimensional GRU slots plus one learned global
GRU scalar: **41 recurrent scalars**. Both recurrent transitions, event encoder
and forecast readout learn from outcomes. There are **no** channel assignments
for bits, generations, active service or due times. Entity IDs provide schema
addressing; learning the address convention is not tested by this baseline.
The global scalar is learned, not a copied clock. Public query time is an input.

All models start freshly initialized; no arm inherits pretraining or a teacher
state. The baseline uses a standard MLP readout. TAC arms use the actual
4×8×24 delta PLM, CASM-aligned retrieval and sparse conditional execution.
New common public event encoders consume 8 entity indicators, 4 event-kind
indicators, tick/due divided by the fixed constant 2048, and binary payload
one-hot indicators. Queries expose entity, action kind, now and forecast time.
Expected outcomes, role, partner, generation, pending lists and evaluator worlds
never enter model.forward/update/decode.

| Arm | Encoder learns | Recurrent parameters learn | Reader learns | State |
|---|---|---|---|---:|
| schema41 | yes | yes, slot/global GRUs | yes, MLP | 41 |
| schema41_encoder | yes | no | yes, MLP | 41 |
| tac_reader | no | no | yes, native CASM | 768 |
| tac_encoder | yes | no | yes, native CASM | 768 |
| tac_memory | no | yes | yes, native CASM | 768 |
| tac_joint | yes | yes | yes, native CASM | 768 |

Tensors and total parameters match exactly within the two schema arms and
within the four TAC arms. Architectural parameter counts, recurrent capacities,
readout design and training FLOPs differ across schema/TAC; record these rather
than claiming equal-compute fairness. TAC memory includes shared write/read
keys; attribution is to the recurrent memory mechanism including addressing,
not exclusively to erase/write gates. Frozen-component hashes and gradients
must prove each intervention. Reset and pairwise state swaps test stored-past
dependence with unchanged weights and public queries.

## Temporal task and exactly 512 observations

Every history contains exactly H public observations at ticks 1..H, including
initial inspections and subsequent competing changes. H=32/128/512. Old
retention and recent acquisition remain controls. Pending-cancellation partners
share the same initial target failure and repair; a historical reset changes
whether that unresolved repair can mature. Reissued-cancellation partners
invalidate the first scheduled generation, schedule a new repair and differ
in whether the new generation is also invalidated. Obsolete interventions
remain in the private environment queue but cannot causally change the world.

Due delays at evaluation are 16/64/128 ticks after the final historical tick.
Forecast horizons are before due (D/2), at due (D), and after due (1.5D).
Before-due pending partners both remain failed; at/after partners have opposite
outcomes. The primary target is the **pre-sensing wait-future prediction on
at/after cases**, which is balanced. Before-due behavior is separately reported;
mixing it into a balanced chance claim would inflate scores. Generation labels
and named latent state channels are never supervised. No unknown new exogenous
changes occur during forecast horizons: these are conditional causal forecasts.

## Training and held-out grid

Seeds 43/59/71; **384 matched updates**, 8 episodes/update, one opposite-past
pair from each role. Training H=32/128, due delays 4/16/32, all deadlines.
AdamW LR .002, weight decay .0001, clip 1. All arms get identical offline CE
labels for current state, wait-future and service-future on all eight entities.
Current/future labels are explicit supervised training data, not privileged
inference inputs. The model receives new state bits at inference only when the
policy actually inspects an entity. Service-future outcomes are zero here and
remain an easy diagnostic; they cannot establish predictive competence alone.

Each update also includes two common, randomly selected late inspections per
training episode. All arms receive identical public receipts. The loss averages
pre-inspection and post-inspection forecast CE, teaching receipt-driven recurrent
updates without supervising a hidden generation or changing training data by
arm. Training receipt hashes are recorded. Evaluation still commits the primary
forecast before any policy-selected inspection; H counts only that exact
pre-sensing stream. Training sensing is random coverage, not learned RL.

Evaluation: 3 histories × 3 due delays × 3 deadlines × 4 roles × 8 pairs ×
2 partners = **1,728 episodes/checkpoint**. Six arms × three seeds =
**18 checkpoints and 31,104 model episodes**. Each episode has all-entity
pre-sensing forecasts and three separately branched sensing controls. Conditional
control trajectories from the same episode are not independent replications.

## Action-selected sensing and actual decision cost

The policy selects up to two uninspected entities using uncertainty from its
learned wait-future probabilities: maximize `4*p*(1-p)`, stop when the maximum
is at most .1, tie-break by public entity ID. Each chosen inspect action returns
only that entity's current bit; its GRU/PLM state is updated from that public
receipt. No generation, pending flag, future outcome or gold inspection target
is disclosed. Full sensing traces are saved. This is a fixed uncertainty
heuristic over learned beliefs, not a separately trained RL sensing policy or
an optimal value-of-information computation; a current inspection does not
directly reveal whether a future repair remains eligible.

Controls: no sensing; selected sensing; random inspections with exactly the
same per-episode inspection count as the selected policy. The policy never
reads the scripted initial history again after forming recurrent state.
Inspections are instantaneous and preserve world tick; this simplified cost
model does not include elapsed sensing delay.

Actions use public forecast probabilities: service an entity only if its
predicted wait cost `4*p_wait_failure` exceeds `1+4*p_service_failure`.
Actual realized cost is **4×failures + 1×services + .1×inspections**, normalized
by eight entities. Report failure rate, unnecessary-service rate, service rate,
inspection count and total cost separately. Always-service has no failures
but costs 1/entity before sensing and wastes services on healthy futures.
Fresh post-action inspection verifies a committed prediction; it never repairs
the initial action/forecast score and does not certify economic optimality.

## Frozen success rules

Each seed of a passing architecture must achieve primary forecast >=80% and
H512 pending/reissued forecasts at due delays64/128 >=75%. Its H512 selected
policy must have failure rate <=5%, unnecessary services <=10%, and cost/entity
<=.80. Mean selected cost must not exceed no-sensing cost. Report all seeds and
before/at/after conditions even when these gates fail.

TAC memory success additionally requires **both** joint minus encoder-only and
memory-only minus reader-only primary gains >=5 pp, with exploratory paired
three-seed 95% t interval lower bounds >0. Joint state-swap must reduce primary
accuracy by >=10 pp; reset mature-target accuracy must be exactly 50%. The
explicit swap threshold is implemented in the audit and committed before
training. Small seed count, adaptive task design and multiple conditions
still prevent confirmatory claims. A passing schema baseline does not promote
TAC memory, and improvements confined to an encoder control cannot be called
learned recurrent-memory gains.

## Evidence and reproduction

CPU GitHub Actions first runs the full suite and smoke, then trains six arms
per seed. Every checkpoint is authenticated and every forecast, chosen
inspection, service action, physical cost and verification row is regenerated
from saved weights. A separate standard-library auditor independently
regenerates temporal targets, validates component interventions, hashes all
18 files, rederives sensing choices and costs, and calculates seed statistics
and the fixed decision rule. Replay does not reproduce SGD or constitute
independent external scientific replication.

```sh
python -m pytest tests/ -q
python scripts/run_learned_schema_dev.py --seed 43 --output artifacts/DEV-017-seed43
python scripts/run_learned_schema_dev.py --seed 43 --replay --output artifacts/DEV-017-seed43
python scripts/audit_learned_schema_dev.py artifacts/DEV-017-seed43 artifacts/DEV-017-seed59 artifacts/DEV-017-seed71 --output artifacts/DEV-017-analysis.json
```

All previous DEV-015/016 evidence remains intact. Keep this research PR draft.
