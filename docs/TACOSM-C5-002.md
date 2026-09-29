# TACOSM-C5-002 — addressing and execution, separated and gated

**Status: pre-registered.** Not run. Not amended from a result. The successor to
`TACOSM-C5-001`, which is void; see `docs/TACOSM-C5-001-RESULT.md`.

## Purpose

C5-001 asked one question and measured one number, and the number it measured
was a compound of three quantities that can move independently. This
experiment is the same physical question — does reducing the executed
candidate set from H to K preserve what matters — with the quantities
separated into endpoints, plus a gate that closes the run before an ambiguous
number can be produced.

The pipeline is unchanged:

```
H -> address -> retained set R -> CASM-S execution of every member of R
```

What changes is what is measured at each stage, and when the run is allowed to
stop.

## Why a new experiment and not an amendment

A pre-registration revised to fit a result is not a pre-registration. The
C5-001 failure was in the design, not in a definition that could be reworded,
and its contract stands as registered. See `docs/TACOSM-C5-001-RESULT.md` for
the void record, the artifact and the four failure modes.

## The two-quantity separation

C5-001's `success_rate` was simultaneously *did addressing retain an
acceptable candidate*, *did execution compute the relation* and *did the
selection rule pick a good one out of the retained set*. In C5-002 these are
three different fields:

| quantity | endpoint | what reads it | what must not |
|---|---|---|---|
| retention / coverage | `coverage_rate` | retained indices and the hidden acceptable set | any CASM-S output, selection result or verification result |
| execution correctness | `execution_accuracy_rate` | the public relation, the selected descriptor, the executed output | the acceptable set |
| work scaling | `structures_executed_per_query` and the work units | the serialized graph submitted to CASM-S | the output's semantic value |

The load-bearing constraint is the first row. `coverage_rate` is computed from
the retained index and the hidden acceptable set **alone**. It cannot inherit
an execution failure, which is exactly what C5-001's `success_rate` did: its
selection rule was argmax over CASM-S output, the output was uninformative,
and the endpoint collapsed to the base rate.

In the exhaustive arm the retained set is the full population, so
`coverage_rate` is 1 by construction. That is the correct value and the
correct reading: the reference arm has no addressing step to fail, and the
analytic control for the addressing endpoint is not the exhaustive arm but the
exact index's own ceiling.

## The representability gate

This is the part that makes C5-002 a new experiment rather than a patched one.

C5-001 had no gate. Its bridge calibration reported 0.84375 validation
accuracy, the run completed, and the table looked plausible while
`verification_rate` was 0.0000 in all 75 cells. The calibration was measuring
the wrong thing — regression accuracy over the output distribution — while the
downstream decision is a *within-query argmax* followed by a *thresholded
verification*, which is a different and harder requirement. A model can be
84% accurate in absolute terms and still rank candidates arbitrarily within a
query. Nothing in C5-001's design could detect that, so it produced an
ambiguous result and had to be voided.

C5-002 runs the gate **before** the task stream, on structures the task stream
never sees, under the exact execution path the task stream uses:

```
held-out structures -> bridge calibration -> CASM-S execution
   -> is the required output produced under the threshold C5 uses?
      no -> terminate as INSTRUMENT_INVALID, report the gate, emit no capability number
```

The gate is not a calibration score. It asks the question the endpoint
actually depends on: does this frozen model, on this graph family, under this
threshold, separate the relation's satisfier from its non-satisfier? If it
does not, the instrument cannot measure execution capability on this family
and the run stops. A terminated gate is a result about the instrument, and it
is recorded as such — not as a CASM-S negative result, because the unresolved
causes (soft-op `alpha` scaling, unlearned gate routing, the 0.5 threshold,
the bridge training target) are causes of the *measurement*, not of the
substrate.

The gate is pre-registered, not tuned. Its pass condition is fixed here before
any run, so it cannot be relaxed to make a run succeed.

## Arms

**exhaustive** executes all H candidate structures. Its `coverage_rate` is 1
by construction; it is the work-scaling denominator and the execution
reference. It must not use a CASM-S output to select for the coverage
endpoint, because there is no selection to make when everything is retained.

**exact-indexed** builds the existing content-addressed equality index once
per static population, executes only the retained K candidates, and is measured
against its own analytic ceiling. The index is exact on this population — the
bucket for a query is exactly the acceptable set, verified in
`docs/TACOSM-C5-001-RESULT.md` — so the ceiling is 1 and the arm is a
*control*: it measures the retention boundary, not a learned capability.
Because the bucket *is* the acceptable set, any non-empty truncation of it
still intersects the acceptable set, so `coverage_rate` is 1 for every
registered K and the K level varies the executed-work budget rather than the
coverage. The work endpoints are what distinguish K=2 from K=4 on this arm.

**representation-addressed** uses a fixed cosine-similarity baseline over the
public query bit vector and the candidate descriptor, retaining K candidates.
It is explicitly not a learned semantic-addressing result, and its
`coverage_rate` is the interesting one: it measures how much retention a fixed
similarity achieves when it is not exact, which is genuine addressing evidence
in its own right.

For every arm the CASM-S model, checkpoint, graph compiler, runtime input
path, and verifier are identical.

## The selection rule, stated as the defect it replaces

C5-001 selected by argmax over CASM-S output:

```python
local = max(range(len(executions)), key=lambda i: (executions[i].result.output, -i))
```

With constant output, the argmax is deterministically index 0. The exhaustive
arm therefore always selected candidate 0, and its `success_rate` was exactly
the base rate — 0.2500, seed spread 0.0000, at every H.

C5-002 does not fix this by writing a better argmax. It removes the dependency
structurally: the addressing endpoint no longer reads a CASM-S output at all.
`coverage_rate` is a set intersection between the retained indices and the
hidden acceptable set. Execution is still needed for `execution_accuracy_rate`
and for the work endpoints, but a bad or uninformative execution can no longer
degrade the addressing measurement. The two halves are independently
interpretable, which is what makes the failure modes distinguishable rather
than merged into one uninterpretable number.

## Measured work

For each executed structure the runner records:

```
structures_executed
active_nodes
candidate_edges
gate_evaluations
executed_structural_operations
node_outputs
```

The aggregate is reported per query and across the cell. Work units derive
from the actual serialized graph submitted to the external model, not from
the Python candidate-count metric. `casm_forward_batches` is reported
separately because one CASM-S forward can process multiple retained
structures; it must never be read as the number of structures executed.

Addressing has its own work endpoints, registered separately because a single
work total cannot show which stage cost it: `address_positions_per_query` is
the candidate positions the content-address index inspects (zero on arms that
do not use it), and `representation_candidates_scored_per_query` is what the
fixed cosine scorer evaluates. Without them the representation arm's retention
would look free — a coverage number with no cost attached to it, which is
the shape of a claim this experiment is not making.

### The 128x observation is retained but not promoted

C5-001 measured a real reduction in submitted work at H=256, K=2:

```
W_exhaustive(256)  = 256 structures/query = 12288 edges/query
W_indexed(256, K=2) =   2 structures/query =    96 edges/query
ratio = 128x
```

That number survives the void because work is counted from the serialized
graph, not from the output. It is recorded in
`docs/TACOSM-C5-001-RESULT.md` and `docs/EVIDENCE_REGISTER.md` as an
**instrument-level observation** — a property of the accounting and the
pipeline, not of selective computation as a capability. C5-002 measures the
same work endpoints with the same accounting, so the reduction is either
reproduced under a valid instrument or not, and either way it stays on the
instrument side of the register until a gate-confirmed run says otherwise.

## CASM-S pin

The runner requires the external checkout to be exactly:

```
FPC-effortless/cdl-attention-experiment
c31554413301e3c9d3e6b3f8c8c6be572a74a748
```

A different checkout fails closed.

## Reproduction

```
python scripts/measure_c5_casm_002.py \
  --casm-root /path/to/cdl-attention-experiment \
  --train-bridge \
  --checkpoint-out /path/casm-s-c5-002.pt
```

For a frozen model after calibration:

```
python scripts/measure_c5_casm_002.py \
  --casm-root /path/to/cdl-attention-experiment \
  --checkpoint /path/casm-s-c5-002.pt
```

The confirmatory run must use the pre-registered values in
`contracts/TACOSM-C5-002.json` unless an explicit deviation is recorded. A
`--smoke` run declares its deviation and is not a measurement.

## What this experiment does not establish

- **Not semantic retrieval.** The exact index is exact because the population
  is constructed so equality on two marked positions is hashable. A learned
  semantic addresser implementing the same interface remains future work
  (GitHub issue #5).
- **Not the full system-level C5 claim.** This is one controlled benchmark,
  one relation, one frozen model, one threshold.
- **Not a CASM-S capability verdict.** The gate's job is to determine whether
  the instrument *can* measure capability on this family. A gate failure is a
  finding about the measurement, and the unresolved causes listed above stay
  unresolved by it.
- **Not a connection from TAC-OSM selective execution to a validated CASM-S**
  (GitHub issue #6). That is the long-running goal this experiment is one
  instrumented step toward.
