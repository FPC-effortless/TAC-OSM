# TACOSM-C5-001 — Result: VOID / instrument failure

Run: GitHub Actions 36512989760
Measurement commit: `78cbec7bba336c5f3fc2f09c2d81c5061051a70e`
Contract fingerprint: see `contracts/TACOSM-C5-001.json`
Artifact: `TACOSM-C5-001-78cbec7bba336c5f3fc2f09c2d81c5061051a70e`
(artifact 11009388891, 90-day retention — the raw cells and the frozen CASM-S
checkpoint are retained so this verdict is auditable rather than asserted)

**STATUS: VOID for capability inference.** No capability claim. No negative
capability claim. The successor experiment is `TACOSM-C5-002`.

## Why the run is void, in one sentence

The experiment was designed to measure whether reducing the executed
candidate set preserves capability, but its selection rule made the
capability endpoint depend on CASM-S's output being informative — and in this
configuration it was not, so the reference arm measured the population's base
rate and the indexed arm's success was guaranteed by the benchmark's own
acceptable-action construction rather than by execution.

## The measured numbers

Configuration: H = {8, 64, 256}, K = {2, 4}, seeds 0–4, 100 measured queries
per cell, warmup 3 excluded. Bridge calibration: 512 training examples, 128
validation examples, 100 epochs, final train loss 0.1797,
**validation accuracy 0.84375**.

Mean over the 5 seeds:

| arm | H | K | success | verification | structures/query | edges/query | wall (s) |
|---|---:|---:|---|---:|---:|---:|---:|
| exhaustive | 8 | — | 0.2500 | 0.0000 | 8.00 | 384 | 1.06 |
| exhaustive | 64 | — | 0.2500 | 0.0000 | 64.00 | 3072 | 8.77 |
| exhaustive | 256 | — | 0.2500 | 0.0000 | 256.00 | 12288 | 37.17 |
| exact-indexed | 8 | 2 | 1.0000 | 0.0000 | 2.00 | 96 | 0.28 |
| exact-indexed | 8 | 4 | 1.0000 | 0.0000 | 2.00 | 96 | 0.28 |
| exact-indexed | 64 | 2 | 1.0000 | 0.0000 | 2.00 | 96 | 0.28 |
| exact-indexed | 64 | 4 | 1.0000 | 0.0000 | 4.00 | 192 | 0.55 |
| exact-indexed | 256 | 2 | 1.0000 | 0.0000 | 2.00 | 96 | 0.31 |
| exact-indexed | 256 | 4 | 1.0000 | 0.0000 | 4.00 | 192 | 0.54 |
| representation-addressed | 8 | 2 | 0.3500 | 0.0000 | 2.00 | 96 | 0.28 |
| representation-addressed | 8 | 4 | 0.2000 | 0.0000 | 4.00 | 192 | 0.52 |
| representation-addressed | 64 | 2 | 0.6000 | 0.0000 | 2.00 | 96 | 0.28 |
| representation-addressed | 64 | 4 | 0.5000 | 0.0000 | 4.00 | 192 | 0.54 |
| representation-addressed | 256 | 2 | 0.6000 | 0.0000 | 2.00 | 96 | 0.31 |
| representation-addressed | 256 | 4 | 0.5500 | 0.0000 | 4.00 | 192 | 0.58 |

Seed spread on `success_rate` is 0.0000 for every exhaustive cell and every
exact-indexed cell. The `representation-addressed` spread is 0.10–0.26, i.e.
that arm has genuine variance and its numbers are not degenerate.

## The four failure modes

Each was verified locally against the same benchmark code the runner used,
because a verdict about an instrument has to be reproducible without the
compute.

### 1. The exhaustive arm collapsed to the population base rate

Selection in `run_cell` was

```python
local = max(range(len(executions)), key=lambda i: (executions[i].result.output, -i))
```

— argmax over CASM-S output, with first-index tie-breaking. If CASM-S's
output is constant across candidates, the argmax is deterministically index 0
of the retained list. For the exhaustive arm the retained list is
`(0, 1, ..., H-1)`, so the arm always selected candidate 0.

The measured `success_rate` of exactly 0.2500 with seed spread 0.0000 at
every H and every seed is that prediction confirmed: for this
`validity="multiple"` population with two marked positions the acceptable
fraction is exactly 1/4, and `P(candidate 0 is acceptable) = 0.2500` was
confirmed directly over 40 sampled tasks. A constant-output reference arm is
a coin flip, not a capability reference.

This is why "exact-indexed is above its own reference arm" is not a good
result: the reference measures nothing.

### 2. The exact-indexed arm's success was guaranteed by construction

The exact equality index buckets candidates by their marked-position bits, so
the bucket for a query contains exactly the candidates matching the
reference on those positions — which *is* the acceptable set. This was
confirmed in 4/4 sampled tasks: `bucket ⊆ acceptable_actions`, with the
bucket holding the same candidates as `acceptable_actions` up to ordering.

Consequence: any selection rule whatsoever succeeds when applied inside the
bucket, including a constant-output argmax. The arm's 1.0000 is a property of
the index and the benchmark, not of CASM-S. It is the retention ceiling, and
it would have read 1.0000 even if CASM-S had returned random outputs.

### 3. Verification rate 0.0000 confirms the output carried no usable evidence

`RelationConstraintVerifier.verify_task` compares
`float(computation.trace[0]) >= 0.5` — the CASM-S output — against the public
relation. It failed in all 75 cells. Combined with failure 1, the consistent
reading is that the frozen bridge model's output did not cross the decision
threshold in a way that matched the relation on this graph family. The bridge
training target was the circuit's Boolean output, but CASM-S's ops are
`alpha`-scaled soft Boolean functions whose products do not reach 1 unless
the whole `alpha` chain is 1, so a perfect-match candidate can still score
below 0.5. That is a representability failure of the *instrument*, not of
CASM-S as a substrate.

### 4. Calibration accuracy was the wrong criterion for a thresholded decision

The bridge reported 0.84375 validation accuracy, computed as
`(output >= 0.5) == target` on held-out circuits. That number is a regression
accuracy over the output distribution. The decision that matters downstream
is a *within-query argmax* followed by a *thresholded verification* against a
public relation — a different and harder requirement, because it depends on
the output separating candidates inside one query, not on each candidate's
output being near its target in isolation. A model can be 84% accurate in
absolute terms and still rank candidates arbitrarily within a query. C5-001
had no gate that would have detected this, which is precisely why the run
completed and produced a plausible-looking table.

## What survives: an instrument-level observation, not a capability result

The work accounting does not depend on CASM-S's output being semantically
correct, because the work is counted from the serialized graph submitted to
the model, not from the output. The reduction is therefore a real measurement
of what the pipeline *submitted*:

```
W_exhaustive(256) = 256 executed structures/query = 256 x 48 = 12288 edges/query
W_indexed(256, K=2) =   2 executed structures/query =   2 x 48 =    96 edges/query
ratio = 12288 / 96 = 128x
```

| H | K | structures/query | edges/query | reduction vs exhaustive |
|---:|---:|---:|---:|---:|
| 8 | 2 | 2 | 96 | 4x |
| 64 | 2 | 2 | 96 | 32x |
| 256 | 2 | 2 | 96 | **128x** |

This establishes that the current pipeline can make CASM-S execution work
proportional to the retained set rather than to the full candidate
population. It does **not** establish that the retained structures are the
right structures, nor that CASM-S successfully computes the target relation.
It is recorded here as an instrument-level observation so it is citable
without being promotable: it is a property of the accounting and the
pipeline, not of selective computation as a capability. It does not move C5.

The one arm whose numbers are not degenerate is `representation-addressed`,
whose 0.3500–0.6000 tracks a real property: fixed-cosine retention is not
exact on this population (measured top-2 retention accuracy ≈ 0.75 at H=8,
vs the exact index's 1.0), so the arm's success falls between the base rate
and the ceiling. That is a valid *addressing-accuracy* measurement. It is not
an execution measurement, and the arm was registered from the outset as not a
learned-retrieval result — so it is recorded as addressing evidence and not
as C5 evidence.

## What this void does not license

- It is **not** a finding that CASM-S cannot execute the relevance circuit.
  The evidence establishes only that *this measurement instrument* could not
  distinguish CASM-S execution capability from base-rate behaviour. The
  candidate causes — representational mismatch in the soft-op `alpha`
  scaling, unlearned gate routing, the 0.5 threshold, or the bridge training
  target — remain unresolved and are not separated by this run.
- It is **not** a negative result about selective execution. The work
  accounting is positive evidence that the pipeline can scale submitted work
  with `|R|`; what is missing is the capability half, which was never
  measured.
- It does **not** amend `TACOSM-C5-001`. A pre-registration revised to fit a
  result is not a pre-registration, and the failure here is in the design,
  not in a definition that could be reworded. The contract stands as
  registered, and this document is its outcome.

## Why C5-002 is a new experiment rather than a repair

The three quantities C5-001 collapsed into one number are genuinely distinct:

| quantity | what it answers | C5-001's status |
|---|---|---|
| retention / coverage | did addressing retain an acceptable candidate? | confounded with selection |
| execution correctness | did execution compute the intended relation? | never measured (verification 0) |
| work scaling | how much structural computation was performed? | measured, and valid |

C5-002 separates them, and — the part that makes it a new experiment rather
than a patched one — it adds a gate that terminates the run as
instrument-invalid before any capability number is produced, rather than
after. See `docs/TACOSM-C5-002.md`.

## Reproduction

The artifact retains the frozen checkpoint and the full 75-cell record. The
C5-001 measurement itself is reproducible from the registered contract:

```
python scripts/measure_c5_casm.py --casm-root /path/to/cdl-attention-experiment \
  --train-bridge --checkpoint-out /tmp/casm-s-c5.pt
```

The successor is not a re-run of this script; it is
`scripts/measure_c5_casm_002.py` against `contracts/TACOSM-C5-002.json`.
