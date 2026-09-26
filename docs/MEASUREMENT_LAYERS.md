# TAC-OSM — Three Layers of Measurement

**Status:** the measurement contract for this repository. Every number the
repository emits belongs to exactly one of the three layers below, and the
layers are **ordered**: a Layer 2 or Layer 3 number is uninterpretable unless
the Layer 1 gate for the model that produced it has passed.

This document exists because the repository's most expensive errors were all
Layer 1 failures that read as Layer 2 findings. Naming the layers is what
makes the next one visible instead of publishable.

---

## The contract

| | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| Question | is the measurement about the model at all? | does the mechanism behave as claimed? | does the system solve the task, and at what cost? |
| Failure mode | silent — produces a plausible wrong number | loud — produces a real number about the wrong thing | the only layer that can over-claim |
| On failure | every downstream number is meaningless | one mechanism claim is withdrawn | one capability claim is withdrawn |
| Reversibility | none — rerun | rerun | rerun |

**A Layer 1 failure is invisible from Layer 2 and Layer 3.** That is the whole
point. A zeroed parameter vector produces a smooth, bounded, sensible-looking
softmax; an ambiguous task produces a clean, reproducible accuracy. Neither
error is detectable in the numbers it emits. They are only detectable by
checking the preconditions, which is why Layer 1 is a gate and not a metric.

---

## Layer 1 — Model validity

> Before anything is measured, establish that the thing being measured is the
> model, on the task, with the intended parameters.

Layer 1 is not about the model's quality. It is about whether a measurement
*refers to* anything. It answers yes/no questions, and a **no** voids the run
rather than degrading it.

### The checks

| Check | What it catches | Where it lives |
|---|---|---|
| weights trained, not initialisation | the all-zero vector → uniform softmax → `routing@1 ≈ 1/H` read as a finding | `src/tac_osm/integrity.py`, `assert_trained` |
| checkpoint corresponds to the arm under test | a stale or mismatched archive loaded into the wrong arm | `snapshot_router` + `parameter_hash` in the run manifest |
| feature basis represents the relation | an oracle that passes while the learned map provably cannot express the relation | `src/tac_osm/representability.py` (§34), `builder.check_representability` |
| gold is the unique satisfier | an ambiguous task, where accuracy is a property of the generator | generator integrity tests |
| gold not exposed to the router | an accidental oracle — the model sees the answer | `src/tac_osm/leakage.py`, structural |
| oracle arm = 1.0000 | task ambiguity; also catches a broken harness | `measure_*.py`, printed *first* |
| frozen baseline unchanged | drift in the reference between comparisons | `TACOSM-BASELINE-001`, commit `91597ab` |
| seeds reproduce the task | a "result" that is a specific RNG path | `RunManifest` / `freeze` |
| benchmark version pinned | comparing numbers from different generators | schema hash in the manifest |

### The two failures this layer exists for

**TACOSM-HS-001.** The history-scaling sweep reported `routing@1 ≈ 1/40` at
H=32 and concluded the router's decision was near-meaningless at moderate
history sizes. The router had no weights loaded. `w = [0]*n` scores every
candidate identically, the softmax is uniform, entropy is `ln(H)`, and `1/H`
reads as a smooth, plausible degradation curve. The *interpretation* was a
finding about the harness published as a finding about the router. With
weights loaded, `routing@1` at H=32 is 0.4250. Recorded as claim **C4**, and
the gate that catches it is now unconditional in every measurement script.

**`dd8f63c`, `cdl-attention-experiment`.** The learned feature map never read
`state.key`. The intended relation was identically unrepresentable — the
analytically ideal weight vector scored exactly 0 for every candidate — while
the `true_key` *control arm* reported 1.0000 for four consecutive commits.
Oracle success and representability are independent: an oracle arm never
passes through the learned map, so an oracle passing says nothing about the
basis. Recorded as claim **C3**, and the §34 gate is now a compulsory
pre-training failure.

Both are Layer 1 failures. Both produced publishable-looking numbers. Both
were caught by checks that are now cheap and automatic, and neither was
detectable from the number itself.

### Why the gate raises instead of returning a flag

`assert_trained` raises `IntegrityError`. A Layer 1 check that returns a flag
gets logged, then ignored, then shipped. A build failure gets fixed. The same
reasoning makes a failed representability gate a build failure rather than a
warning.

---

## Layer 2 — Mechanism measurement

> Given a valid model on a valid task, does a specific mechanism behave as
> claimed — and *by how much*?

Layer 2 is where the repository's actual results live. Each measurement is
attached to one named mechanism in the loop and to one claim in
`docs/CLAIMS.md`. A Layer 2 number is real on a bad day and a good day alike;
what it must not be is *attributed to the wrong mechanism*.

### The three mechanism families

**Routing (`R_t`)** — is the relevant item identifiable?

`routing@1` (argmax accuracy), `recall@K` (`P(gold ∈ top-K)`), `gold_rank`,
`entropy` against `ln H`, and the **score margins**. The margins are the part
that matters and the part that was missing until TACOSM-MATCHED-001:

- `delta_1 = s_gold − s_best_distractor`, the gap argmax has to cross;
- `delta_K` against the K-th competitor, for the retrieval budgets;
- `prob_margin`, the same gap in softmax units.

A `recall@16 = 0.55` is ambiguous on its own: gold may sit at rank 2 behind
one strong distractor, or be buried among a hundred near-tied candidates. Both
give the same recall and need different fixes. `delta_1` separates them
directly, which is what made the matched-H result legible.

Note the units, because the wrong unit produces a false degradation. The
`RoutingDecision.scores` are softmax probabilities, normalised over H, so a
probability-unit margin shrinks with H *even for a perfect scorer*. Margins
are computed from `router.score()` — the raw dot products — and compared across
H. Both are reported: the raw margin is what the scoring function produces,
the probability margin is what the decision sees.

**Persistence (`S_t → S_{t+1}`)** — does written state change later decisions?

Read/write survival across steps, the intervention ladder
(`persistent | shuffled | reset | corrupted | random | wrong_key`), and
`state_lookup` / `replay` family accuracy, which cannot be solved without a
written vector on the marked positions. The interventions are prefrozen into
the ablation matrix, so a dose–response curve is a matrix cell, not a new
experiment.

**Execution (`C_t → A_t`)** — given the right candidate, is the right
structure run?

`nodes_executed`, relevant nodes executed, and structural correctness
(`exec|route`: conditional accuracy given the correct candidate was routed).
The last is what attributes a failure to routing rather than execution — at
H=2 `exec|route` is 0.9620 where accuracy is 0.9220, so the loss is routing,
not execution.

### The attribution rule

A Layer 2 number must name its mechanism, and a degradation must be
attributed by a control, not inferred. `oracle = 1.0000` attributes a
degradation to the model rather than the environment; `exec|route` attributes
it to routing rather than execution; the analytic vector attributes a trained
failure to the *learning rule* rather than the *hypothesis class*.

TACOSM-MATCHED-001 is the example of what this rule is for. Two readings of
`recall@16 = 0.550` at H=256 survived the data — the representation is
inadequate, or the training population was wrong — and both were refuted by
attribution controls: the analytic vector reaches `routing@1 = 1.0000` at
every H up to 512 (representation adequate), and a router trained at H=256 is
*worse at H=256* than one trained at H=8 (population not the issue). The
degradation was real; its mechanism was neither of the two candidates, and
only attribution separated that from a representation redesign.

---

## Layer 3 — System capability

> Does the integrated system solve the task, and at what computational cost?

This is the layer that gets funded and the layer that must not be reached
early. Its central measurement is the **capability-vs-computation curve**:

```
C_total  =  C_address  +  C(|R|)  +  C(|A|)  +  C_verify
```

where `|R|` is the relevant subset and `|A|` the acted-on structure, against
the conventional baseline where cost grows with total history `|H|`:

```
C_total  =  C_address  +  C(|H|)  +  C(|A|)  +  C_verify
```

The claim is **not that computation is small. It is that computation depends on
the relevant subset rather than on everything that happened.**

### The honest v0.1 cost model

```
C_total(H) = O(H) routing + O(10) execution + O(verification)
```

The executor is capped at `active_count = max_nodes = 10`. `C_executed`
therefore reads 10 at every H in TACOSM-HS-001 — a flat column that means the
architecture has **no room to scale**, not that scaling has been demonstrated.
`C_router` is the only term that genuinely varies, and it grows linearly with
H. The desired model — `C_total ≈ C_address + C(|R|) + C(|A|) + C_verify` with
`|R| ≪ H` — is **not demonstrated**, and claim **C5** is `UNTESTED` for exactly
this reason.

This is the gap the roadmap exists to close, and it is the reason a retrieval
boundary (Stage F1) matters more than a better router: the boundary is what
makes `|R|` a measurable quantity at all.

### What Layer 3 requires

A capability-vs-computation curve needs, at minimum:

- a retrieval boundary in the loop, so `|R|` is a defined quantity;
- total compute and relevant compute measured **separately** along the same
  axis, not one derived from the other;
- frozen history levels, with the capability parity margin declared **before**
  the confirmatory run;
- the full ablation matrix (Stage D), because a cost claim about an integrated
  system is not separable from which mechanisms are switched on.

None of these exist in v0.1. Until they do, no figure in this repository may
present `C_executed` as though it had been measured against `|R|`.

---

## How the layers interact

The dependency is one-directional and it is strict:

```
Layer 1 (gate)  →  Layer 2 (mechanism)  →  Layer 3 (system)
   voids              withdraws             withdraws
```

- **A Layer 1 failure voids everything downstream.** It does not degrade the
  numbers; it makes them refer to nothing. There is no partial credit.
- **A Layer 2 failure withdraws one mechanism claim** and may void the Layer 3
  claim built on it. C5 is `UNTESTED` because its Layer 2 precondition — a
  measured `|R|` — does not exist yet.
- **A Layer 3 failure withdraws the system claim** and leaves Layer 2 intact.
  TACOSM-MATCHED-001 withdrew nothing from Layer 2: the routing degradation is
  real, `SUPPORTED`, and unchanged. What it changed was the *mechanism* the
  claim pointed at.

### The pattern this document is designed to break

Every expensive error in this repository followed the same path: a Layer 1
failure produced a plausible Layer 2 number, which was interpreted as a
Layer 2 finding, which was then generalised toward a Layer 3 conclusion about
the architecture. HS-001's "the router is meaningless at H=32" was a harness
bug read as a routing result. F0's "`recall@16` degrades, so the
representation is inadequate" was a training artefact read as a
representation limit, and would have been the justification for rebuilding the
basis.

Both were caught by Layer 1 controls *after* the interpretation was written.
The layers exist so that the check runs first and the interpretation is
written second.
