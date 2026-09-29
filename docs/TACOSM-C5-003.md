# TACOSM-C5-003 — the integrated boundary, gated before the task population

**Status: pre-registered.** Not run. Amended four times before any run —
amendments A1, A2 and A3, below — and none is a revision from a result,
because no result exists to revise from. The successor to `TACOSM-C5-002`,
which terminated as `INSTRUMENT_INVALID` (`docs/TACOSM-C5-002-RESULT.md`).

Related: `docs/TACOSM-C5-001-RESULT.md` (void),
`docs/TACOSM-DEGENERACY-001.md` (the preflight infrastructure this experiment's
gate is built on), `src/tac_osm/degeneracy.py`.

## Purpose

C5-002's gate fired, and the firing was the result: a bridge that reports
0.84375 absolute output accuracy separates satisfier from a one-bit-flipped
violator in only 59 of 256 pairs. The diagnosis is a *specification* error, not
a learning failure — nothing in the bridge's training objective rewards
within-query ranking, so the model reaching 0.84375 absolute accuracy is
behaving as trained. The instrument was measuring the wrong thing, and it
terminated itself rather than publishing a number.

C5-003 is the first fresh capability experiment after that diagnosis. Two
things change, and only two:

1. **the bridge is trained on pairs, not on single candidates**, so the
   objective contains the quantity the downstream decision actually depends
   on; and
2. **the gate is a representability gate with eight registered criteria,
   evaluated before the task population runs**, so an instrument that cannot
   represent the task stops the run instead of producing an ambiguous table.

Everything else is deliberately unchanged from C5-002: the same candidate
population, the same relation, the same arms, the same work accounting, the
same separation contract, the same CASM-S pin, the same threshold. The one
exception is the bridge validation stream's seed, which C5-002 derived as
`BRIDGE_SEED + 1` — the gate's own seed, so its reported validation separation
and its gate verdict read one stream under two names. That is recorded as
amendment A1 rather than fixed silently, because a successor that corrects a
predecessor's design without recording it is indistinguishable from a successor
that changed the calibration. C5-002's design was right about the things that
produced a clean termination; what was wrong was the instrument, and that is
what this experiment replaces.

## The architectural change

C5-002 was registered as a *measurement* experiment whose gate was a single
separation criterion. C5-003 is registered as an **integrated-boundary**
experiment: it tests the boundary between the four stages as a single
causal chain, and the gate is the thing that decides whether the chain is
attached at all.

The loop, as four interfaces with four distinct failure modes:

```
(S, Q) -> R          addressing:     which candidates are even considered?
(R, Q) -> A          execution:      which structure is computed, and is it the relation?
(A, O) -> V          verification:   does an independent check confirm the computation?
R     -> W(R)        work:           what did computing R cost?
```

Each arrow is a place the chain can break, and each break has a different
remedy. C5-001's void was a break at `(S,Q) -> R` *disguised as* a break at
`(R,Q) -> A`, because the compound `success_rate` could not tell them apart.
C5-002 separated the endpoints and added a gate, and the gate found the break
was in the instrument's own representability — before any of the four.

The four stages are registered as separate endpoints for the same reason
C5-002 split `success_rate`: a number that merges them cannot distinguish an
addressing failure from an execution failure, and the distinction is what
makes a negative result interpretable.

## The representability gate

This is the part that makes C5-003 a new experiment rather than C5-002 with a
retrained bridge. The gate is **eight criteria, evaluated in order, on
held-out structures the task population never sees**, under the exact
execution path the task population uses. All eight are pre-registered here
before any run. A failure of any one terminates the run as
`INSTRUMENT_INVALID` and emits no capability number.

| # | criterion | what it rules out | registered constant |
|---:|---|---|---|
| 1 | frozen threshold constants | a gate whose bars moved between registration and run | `GATE_THRESHOLD = 0.5`, `GATE_MIN_ACCURACY = 0.5`, `MIN_SPREAD = 0.05`, `MIN_CLASS_SUPPORT = 2` |
| 2 | held-out pair accuracy | a bridge that does not separate satisfier from violator — the C5-002 failure, at 0.2305 | `GATE_MIN_ACCURACY = 0.5` over `GATE_STRUCTURES = 512` pairs |
| 3 | non-degenerate output spread | a constant or near-constant output — the C5-001 failure, `sd = 0` | `MIN_SPREAD = 0.05`, `ABSOLUTE_FLOOR = 1e-9` |
| 4 | actual verifier acceptance | a gate that passes a model the task stream's own verifier rejects | `verification_acceptance` on the satisfying half of each of the 512 held-out pairs, minimum `GATE_MIN_ACCURACY` (see amendments A2 and A3) |
| 5 | no dependence on `true_edge_set` | an instrument that reads the wiring it is supposed to be tested on | structural: the transported spec never carries `true_edges` (`docs/CASM-S-ADAPTER.md`) |
| 6 | no oracle information entering routing or execution | gold leaking into the addressing or execution stage | `leakage.py`'s `FORBIDDEN_FIELDS`, audited at the gate, not assumed |
| 7 | deterministic reproduction from the frozen checkpoint | a run whose numbers depend on which machine ran it | `checkpoint_sha256` recorded; `hash()` is per-process salted so structure keys are formatted strings, not hashes |
| 8 | pair-trained objective | a bridge whose objective does not contain within-query ranking — the diagnosed cause | `BRIDGE_OBJECTIVE = "pair_separation"`, registered as a constant, not a tuning knob |

Criteria 1–3 generalise the two observed failure modes and are implemented by
`src/tac_osm/degeneracy.py`'s `preflight`. Criterion 4 is new: C5-002's gate
tested the model's output against the threshold, but not against the verifier
that the task stream's own `verification_rate` endpoint reads, and the two
agreed in that run only because both were zero. It measures acceptance on the
**satisfying half** of each pair, not over both halves:
`RelationConstraintVerifier` is a *success* verifier, so on a descriptor the
public relation does not hold for it rejects unconditionally
(`structured_verifier.py`'s `relation_unsatisfied` branch) even when the
executed output agrees with the relation — two-sided acceptance is therefore
capped at one half by construction, and the registered minimum of 0.5 would
be a ceiling reached only vacuously. The satisfying half is the half that
moves: an inverted model is rejected there, where a threshold-only check sees
a model confidently wrong in the same direction on both. Amendment A3, below,
changes what the outcome's `value` field carries on that half, and records
that the alpha-scaled output does not reach the threshold from below either;
criteria 5 and 6 are structural and are checked on the objects that cross
the boundary, not on a prose assurance. Criterion 7 is what makes the run
reproducible rather than merely repeatable.

The gate runs **before** the task population. That position is the difference
between a terminated run and a voided one, and it is the one design decision
from C5-002 that is carried forward without re-litigation.

```
held-out structures -> pair-trained bridge -> CASM-S execution
   -> do all eight criteria hold?
      no -> INSTRUMENT_INVALID, report the gate, emit no capability number
      yes -> the task population runs
```

### Why the thresholds are what they are, and not tighter

`GATE_MIN_ACCURACY = 0.5` is C5-002's pre-registered value, unchanged. It was
fixed before C5-002's run and is not tunable after seeing 0.2305 — relaxing it
would convert the gate from a termination condition into a fitting knob, which
is the exact failure mode the gate exists to prevent. C5-002's result is the
reason the bar stays where it is, not a reason to move it.

`MIN_SPREAD = 0.05` is loose on purpose. C5-001's output was constant, `sd =
0`, so any positive value separates that failure from a working instrument;
the threshold catches the *class* of failure and is not fitted to sensitivity.

`GATE_THRESHOLD = 0.5` is not re-examined here, but amendment A3 records
something about it that a reader should know: the alpha-scaled soft Booleans
do not reach `0.5` on the satisfying half from below either — the product is
`≈ 0.480` at initialisation and `≈ 0.46` under the C5-002 trained checkpoint.
The threshold is nevertheless unchanged, for the reasons A3 gives, and the
possibility that criterion 2 fires on a well-separating instrument is part of
what that amendment records rather than what it quietly removes.

No threshold here was tuned against a run that passed it.

### A4 — the task-stream endpoint boundary and actual determinism check

A3 fixed criterion 4's use of the verifier by passing the model's own output as
`Outcome.value`. A static audit found that `run_cell` still passed the hidden
action answer as `Outcome.value`, so the measured `verification_rate` would
reintroduce the same `1e-9` output-consistency failure on real soft CASM-S
outputs. A4 carries the A3 convention into the task stream.

The audit also found that `execution_accuracy_rate` was measuring whether the
selected action was in `acceptable_actions`, which is an action-selection
quantity, not a computation quantity. A4 separates the two:
`execution_accuracy_rate` now compares the executed output at
`GATE_THRESHOLD` with the public relation, and `selection_success_rate`
separately measures whether the selected action is acceptable. Neither change
uses the hidden acceptable set to score the execution endpoint.

Finally, criterion 7 now executes the same held-out gate batch twice and compares
the output, gate and node-value records. The checkpoint hash remains recorded,
but provenance alone is no longer treated as proof of deterministic execution.
No confirmatory C5-003 run existed under the old definition, and no thresholds,
arms, population, bridge objective or capability decision rule changed.

## The pair-trained bridge

C5-002's bridge was trained on each candidate's Boolean output in isolation.
The loss was MSE against `all(descriptor[j] == reference[j] for j in marks)`,
a per-candidate regression target. Nothing in that objective rewards ranking a
satisfier above a violator *within one query*, which is what the downstream
argmax and the thresholded verification both depend on. A model trained to
that target and reaching 0.84375 absolute accuracy is behaving as trained;
the separation failure was a specification error in the target, not a learning
failure.

C5-003's bridge is trained on **pairs**: each training example is one
`(reference, satisfying, violating, marks)` tuple — the same construction the
gate evaluates, from a disjoint seed stream — and the loss is on the *margin*
`score(satisfying) − score(violating)`, not on the absolute level of either.
The registered objective is a single named constant, `BRIDGE_OBJECTIVE =
"pair_separation"`, so a change to it is a change to the pre-registration and
visible in the diff rather than a silent re-targeting.

The pair stream is generated by `degeneracy.make_pairs`, the same construction
the C5-002 gate used inline. That identity is not shared code — a runnable
script is the wrong dependency target for a library — and
`test_make_pairs_matches_the_c5_002_gate_construction` is the seam that keeps
the two in step.

Three seed streams are used, and they are disjoint by named constant rather
than by arithmetic: the task population's `SEEDS = (0, 1, 2, 3, 4)`, the
bridge's `BRIDGE_SEED = 20260929`, the bridge validation stream's
`BRIDGE_VALIDATION_SEED = 20261001`, and the gate's `GATE_SEED = 20260930`.
The validation stream is deliberately *not* `BRIDGE_SEED + 1`, which is the
gate's seed: a validation set that is the gate's stream with a count prefix
would make the reported separation and the gate's verdict the same number
twice, and would leave the gate's verdict unattributable to either.

## Arms

Unchanged from C5-002, and deliberately so. The arms were not the defect; the
instrument was.

**exhaustive** executes all H candidate structures. Its `coverage_rate` is 1
by construction; it is the work-scaling denominator and the execution
reference.

**exact-indexed** builds the content-addressed equality index once per static
population and executes only the retained K candidates. The index is exact on
this population, so the arm is the retention *control*: it measures the
retention boundary, not a learned capability, and its `coverage_rate` is 1 for
every registered K because the bucket *is* the acceptable set.

**representation-addressed** uses the fixed cosine-similarity baseline,
retaining K candidates. Its `coverage_rate` is the interesting one: how much
retention does a fixed similarity achieve when it is not exact.

For every arm the CASM-S model, checkpoint, graph compiler, runtime input path
and verifier are identical.

## Endpoints

The four stages of the boundary, as four endpoint families. The separation
contract from C5-002 is unchanged — `coverage_rate` reads no CASM-S output,
computed from retained indices and the hidden acceptable set alone.

| stage | endpoint | reads | must not read |
|---|---|---|---|
| `(S,Q) -> R` | `coverage_rate` (primary) | retained indices ∩ hidden acceptable set | any CASM-S output, selection or verification result |
| `(R,Q) -> A` | `execution_accuracy_rate` | public relation + selected descriptor + executed output | the hidden acceptable set |
| `R -> A` | `selection_success_rate` | selected retained index ∩ hidden acceptable set | CASM-S output; this is an evaluator/action-selection endpoint, not a computation endpoint |
| `(A,O) -> V` | `verification_rate` | the verifier's own constraint check on the computation's trace | the hidden acceptable set |
| `R -> W(R)` | `structures_executed_per_query` and the work units | the serialized graph submitted to CASM-S | the output's semantic value |

The work endpoints are the full C5-002 set: `active_nodes_per_query`,
`candidate_edges_per_query`, `gate_evaluations_per_query`,
`structural_operations_per_query`, `node_outputs_per_query`,
`casm_forward_batches`, `address_positions_per_query`,
`representation_candidates_scored_per_query`, `wall_clock_seconds`.

The complete registered set is in `contracts/TACOSM-C5-003.json`.

## The 128x observation, and its status

C5-001 measured a real reduction in submitted work at H=256, K=2:

```
W_exhaustive(256)  = 256 structures/query = 12288 edges/query
W_indexed(256, K=2) =   2 structures/query =    96 edges/query
ratio = 128x
```

It survives both voids because work is counted from the serialized graph, not
from the output. It is recorded as an **instrument-level observation** — a
property of the accounting and the pipeline, not of selective computation as a
capability. C5-003 measures the same work endpoints with the same accounting,
so under a gate-confirmed instrument the reduction is either reproduced or it
is not, and either way it stays on the instrument side of the register until a
gate-confirmed run says otherwise. This experiment is what would license the
promotion, if the gate passes and the capability endpoints hold.

## CASM-S pin

Unchanged. The runner requires the external checkout to be exactly:

```
FPC-effortless/cdl-attention-experiment
c31554413301e3c9d3e6b3f8c8c6be572a74a748
```

A different checkout fails closed.

## Reproduction

The compute-side entry point is `scripts/measure_c5_casm_003.py`, registered
with `contracts/TACOSM-C5-003.json`:

```
python scripts/measure_c5_casm_003.py \
  --casm-root /path/to/cdl-attention-experiment \
  --train-bridge \
  --checkpoint-out /path/casm-s-c5-003.pt
```

For a frozen model after calibration:

```
python scripts/measure_c5_casm_003.py \
  --casm-root /path/to/cdl-attention-experiment \
  --checkpoint /path/casm-s-c5-003.pt
```

The confirmatory run must use the pre-registered values in
`contracts/TACOSM-C5-003.json` unless an explicit deviation is recorded. A
`--smoke` run declares its deviation and is not a measurement; it is
contract-only and returns before constructing the CASM-S runtime, so it runs
torch-free on the control plane while still enforcing every `require_*`
against the registered design.

## Amendments

All three amendments were made **before any run**, which is the only window in
which a pre-registration may be amended honestly: there is no result to
protect, and the replaced definitions are kept readable in the contract rather
than overwritten. None changes the hypothesis, the arms, the endpoints, the
decision rule or the thresholds. The full text, including each amendment's
`old_definition`, `rationale`, `discovered` and `no_result_under_previous_version`,
is in `contracts/TACOSM-C5-003.json`.

### A1 — the bridge validation stream's seed

The registration asserted the bridge calibration was "unchanged from C5-002 in
every respect except the objective". It is not. C5-002 drew its validation
stream at `BRIDGE_SEED + 1`, which is `20260930` — the gate's own seed — so
its reported validation accuracy (0.84375) and its gate verdict (0.2305) read
the *same* structures under two names, and neither is an independent
observation of the other. The registered identity was false in one unnamed
respect, and the successor silently differed from the design it claimed to
repeat.

C5-003 names the validation stream's seed as its own constant,
`BRIDGE_VALIDATION_SEED = 20261001`, so the three streams' disjointness is
three visible values rather than an arithmetic offset that hid a collision.
The five constants C5-002 named — 512 training examples, 128 validation, 100
epochs, `lr = 2e-3`, training seed `20260929` — are unchanged. C5-002's *run*
is unaffected and stands as measured; the amendment records a defect in its
*design* that its own numbers could not have revealed.

### A2 — criterion 4 measures the satisfying half

The gate table's row 4 read as acceptance over both halves of every pair. It
cannot be. `RelationConstraintVerifier.verify_task` is a *success* verifier:
on a descriptor the public relation does not hold for, it returns invalid via
the `relation_unsatisfied` branch even when the executed output agrees with
the relation, so two-sided acceptance is capped at one half by construction —
and the registered minimum of `0.5` would then be a ceiling the criterion
could reach only by being vacuous.

The implemented criterion scores acceptance on the **satisfying half**: the
fraction of the 512 held-out pairs for which the verifier accepts the
satisfying descriptor on the model's own output. That is the half that moves —
an inverted model is rejected there, where a threshold-only check sees a model
confidently wrong in the same direction on both halves — and the criterion's
registered purpose ("a gate that passes a model the task stream's own verifier
rejects") is preserved by this reading and was never achievable by the
two-sided one. The implementation has always scored the satisfying half; the
amendment brings the registration to what the code does.

### A3 — criterion 4's outcome value, and the threshold the alphas do not reach

A2 resolved *which half* criterion 4 scores. It left the outcome's `value`
field carrying the programme's expected answer — `1.0` on the satisfying
half. That reading is not usable, for a reason in the verifier rather than in
the model: `verify()` enforces `abs(trace[0] − value) <= tolerance` at
`tolerance = 1e-9` *before* it reaches the relation branch, so passing `1.0`
as the value requires the model to emit **exactly** `1.0` on every satisfying
pair.

CASM-S's soft Booleans cannot, in general. They are alpha-scaled —
`alpha = c * softplus(alpha_eta)` per relation port, `c = 1.0`, initialised at
`alpha_eta = [0, 0]` so `alpha ≈ [0.693, 0.693]` — and the 2-mark relevance
circuit compiles to `INPUT ×4, EQ, EQ, AND`, with `EQ` expanded by the adapter
to `XOR + NOT`. The satisfying-half output is a product of the alphas, and that
product equals `1.0` only on a measure-zero manifold in parameter space. It is
*reachable* — `alpha0 · alpha1 = 1.0` at, say, `eta ≈ (0.60, 0.48)` — but no
float32 gradient path lands on it. At the initialisation itself the product is
`0.693² ≈ 0.480`, already below `1`.

The retained C5-002 checkpoint (`casm-s-c5-002.pt`, run `36536249559`,
artifact `11018687523`) makes this concrete rather than hypothetical: read
torch-free as a zip with a numpy-backed `_rebuild_tensor_v2` stub, its trained
parameters are `alpha_eta = [0.19835, 0.17771]`, so `alpha ≈ [0.797, 0.786]`,
and the satisfying-half output is `≈ 0.46`. Under the pre-A3 convention,
criterion 4 scored `0.0` on **every** pair and fired `INSTRUMENT_INVALID` on an
instrument whose separation is exactly what this experiment trains for.

The fix is narrow: the outcome's `value` now carries the model's own output,
the same float the trace holds. The consistency check then passes by
construction and the relation branch's `trace[0] >= 0.5` becomes the operative
test — which still rejects an inverted model, and now accepts `0.5`–`1.0` while
rejecting `0.03` and `0.34`.

Two consequences are recorded here rather than hidden:

1. **Criterion 4 becomes close to criterion 2.** With `value = trace[0]` it
   reduces operationally to "the satisfying-half output is at least the
   threshold", which is criterion 2's `hi >= threshold > lo` without the `> lo`
   term. Its non-redundant content is that it reads the decision through the
   *task stream's own verifier object* rather than through a threshold
   comparison, so a future change to the verifier's boundary is reflected
   here without this criterion being edited.
2. **The alphas do not reach the threshold from below either.** At the C5-002
   trained alphas the satisfying-half output is `≈ 0.46`, below
   `GATE_THRESHOLD = 0.5`; at the initialisation it is `≈ 0.480`, also below.
   The threshold is **deliberately not changed** by this amendment: it is
   frozen by criterion 1, it is the verifier's own decision boundary rather
   than an instrument-specific calibration, and moving it mid-series would
   break comparability with C5-002's `0.2305` separation and its gate verdict.
   The consequence is that criterion 2's absolute `hi >= 0.5` term is itself
   part of what A3 records: if the pair-trained bridge's satisfying-half
   output stays below `0.5`, criterion 2 is the criterion that fires and the
   run terminates as `INSTRUMENT_INVALID` — which is the honest outcome for an
   instrument that does not cross the verifier's boundary, and is a far better
   failure than criterion 4 terminating a run on a `1e-9` technicality.

## What this experiment does not establish

- **Not semantic retrieval.** The exact index is exact because the population
  is constructed so equality on two marked positions is hashable. A learned
  semantic addresser implementing the same interface remains future work
  (GitHub issue #5).
- **Not the full system-level C5 claim.** This is one controlled benchmark,
  one relation, one frozen model, one threshold.
- **Not a CASM-S capability verdict.** The gate decides whether the instrument
  *can* measure capability on this family. A gate failure is a finding about
  the measurement, not about the substrate.
- **Not a connection from TAC-OSM selective execution to a validated CASM-S**
  (GitHub issue #6).
- **Not a claim that the pair-trained bridge is the right bridge.** It is the
  bridge the diagnosis predicts. If the gate still fails, the next diagnosis
  is the soft-op `alpha` scaling, then gate routing, then the threshold — in
  the order C5-002's result registered them, not in an order chosen after
  seeing this one.

## The causal chain this experiment is designed to make

```
bridge cannot represent the task
        |
        v
   STOP — INSTRUMENT_INVALID, no capability number
```

rather than:

```
bridge cannot represent the task
        |
        v
   run the benchmark anyway
        |
        v
   ambiguous result, voided after the interpretation is written
```

C5-001 lived in the second chain and was voided. C5-002 built the first chain
for one criterion and the chain worked. C5-003 extends it to eight, and adds
the bridge the diagnosis predicted.
