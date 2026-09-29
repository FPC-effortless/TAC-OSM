# TACOSM-C5-003 — the integrated boundary, gated before the task population

**Status: pre-registered.** Not run. Not amended from a result. The successor to
`TACOSM-C5-002`, which terminated as `INSTRUMENT_INVALID`
(`docs/TACOSM-C5-002-RESULT.md`).

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
same separation contract, the same CASM-S pin, the same threshold. C5-002's
design was right about the things that produced a clean termination; what was
wrong was the instrument, and that is what this experiment replaces.

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
| 4 | actual verifier acceptance | a gate that passes a model the task stream's own verifier rejects | `verification_acceptance` over the same 512 held-out pairs, minimum `GATE_MIN_ACCURACY` |
| 5 | no dependence on `true_edge_set` | an instrument that reads the wiring it is supposed to be tested on | structural: the transported spec never carries `true_edges` (`docs/CASM-S-ADAPTER.md`) |
| 6 | no oracle information entering routing or execution | gold leaking into the addressing or execution stage | `leakage.py`'s `FORBIDDEN_FIELDS`, audited at the gate, not assumed |
| 7 | deterministic reproduction from the frozen checkpoint | a run whose numbers depend on which machine ran it | `checkpoint_sha256` recorded; `hash()` is per-process salted so structure keys are formatted strings, not hashes |
| 8 | pair-trained objective | a bridge whose objective does not contain within-query ranking — the diagnosed cause | `BRIDGE_OBJECTIVE = "pair_separation"`, registered as a constant, not a tuning knob |

Criteria 1–3 generalise the two observed failure modes and are implemented by
`src/tac_osm/degeneracy.py`'s `preflight`. Criterion 4 is new: C5-002's gate
tested the model's output against the threshold, but not against the verifier
that the task stream's own `verification_rate` endpoint reads, and the two
agreed in that run only because both were zero. Criterion 5 and 6 are
structural and are checked on the objects that cross the boundary, not on a
prose assurance. Criterion 7 is what makes the run reproducible rather than
merely repeatable.

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

No threshold here was tuned against a run that passed it.

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
| `(R,Q) -> A` | `execution_accuracy_rate` | the public relation, the selected descriptor, the executed output | the acceptable set |
| `(A,O) -> V` | `verification_rate` | the verifier's own constraint check on the computation's trace | the acceptable set, and any model output other than the executed one |
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
