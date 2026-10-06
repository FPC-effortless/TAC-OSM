# TAC-OSM — Layers of Measurement

**Status:** the measurement contract for this repository. Every number the
repository emits belongs to exactly one of the layers below, and the layers
are **ordered**: a Layer 2 or Layer 3 number is uninterpretable unless the
Layer 1 gate for the model that produced it has passed, and no layer is
reached before the question has been asked whether the answer is already
inherited — Layer 0.

This document exists because the repository's most expensive errors were all
Layer 1 failures that read as Layer 2 findings. Naming the layers is what
makes the next one visible instead of publishable. Layer 0 was added later,
for the opposite failure: spending experimental capital re-establishing
results the repository already held.

---

## The contract

| | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| Question | is the answer already established, and may it be inherited? | is the measurement about the model at all? | does the mechanism behave as claimed? | does the system solve the task, and at what cost? |
| Failure mode | invisible in the other direction — a redundant run is not detectable from its own output | silent — produces a plausible wrong number | loud — produces a real number about the wrong thing | the only layer that can over-claim |
| On failure | nothing is withdrawn; capital is spent | every downstream number is meaningless | one mechanism claim is withdrawn | one capability claim is withdrawn |
| Reversibility | none — the run was unnecessary | none — rerun | rerun | rerun |

**A Layer 1 failure is invisible from Layer 2 and Layer 3.** That is the whole
point. A zeroed parameter vector produces a smooth, bounded, sensible-looking
softmax; an ambiguous task produces a clean, reproducible accuracy. Neither
error is detectable in the numbers it emits. They are only detectable by
checking the preconditions, which is why Layer 1 is a gate and not a metric.

---

## Measurement Integrity — the M0 protocol

> Before a number is assigned a layer, the run itself must be the registered
> run. This is milestone **M0**, and it is a protocol rather than a layer.

The layers below classify *evidence*. M0 is what makes a run eligible to carry
evidence at all, and it sits before every layer rather than in the table: the
table classifies what a number *means*, and M0 asks whether the number came
from the run that was committed. It is named M0, not L−1, because it is
milestone work rather than a measurement layer — the difference between a
harness that *can* check a run and one that *does*.

Three things are frozen, and only these:

* **the contract check** (`src/tac_osm/contract.py`, `contracts/*.json`) — the
  run matches the registered levels, seeds, schedule length and arms, or the
  script stops and reports nothing. `--smoke` is the one declared way to run
  off the registered design; it weakens no check, and prints its deviation in
  full so the output cannot be mistaken for a measurement;
* **the model-state integrity gate** (`src/tac_osm/integrity.py`) — the
  Layer 1 checks, which raise `IntegrityError` rather than return a flag, for
  the reason in §"Why the gate raises instead of returning a flag";
* **the machine-readable record** (`src/tac_osm/measurement/`) — the outcome
  is written as data alongside the terminal report, with the run's provenance,
  its design, its deviations and its per-seed values, so M0 is auditable after
  the fact rather than only by whoever happened to be reading the screen.

Two invariants hold, and the second is the one that gets forgotten:

```
1.  no contract-compliant run  ⇒  no publishable measurement
2.  contract-compliant run     ⇏  hypothesis supported
```

**1.** A run that deviated from the registered design — a shortened schedule,
a dropped level, an added arm — is not the intervention the decision rule was
written about, so its numbers are not a result of that experiment. They may
still be informative, which is why `--smoke` exists rather than a lock; the
deviation is printed *and* recorded in the record's `contract_deviations`, so
"this was not the registered design" is a field rather than a matter of trust.
This is also why a smoke run's record is still written: it is the evidence
that the run was not registered, and deleting it would delete the proof.

**2.** The converse is the trap the contract system opens once it closes the
first one. A check that the run matched the specification is a statement about
*provenance*, not about the world: the registered hypothesis may simply be
false, and a compliant run of a false hypothesis is a well-specified null
result. A contract deliberately does not pin outcomes — encoding the expected
result would make the pre-registration a foregone conclusion rather than one.
So no row in `docs/CLAIMS.md` moves status because a contract check passed; a
claim moves when a measurement moves. This is the `NOT INHERITED` field of the
contract row in `docs/EVIDENCE_REGISTER.md`, and it is the reason no record
carries a `status` field: the instrument measures, it does not arbitrate the
decision rule.

**M0 is not a column in the table above, and not a layer.** The table
classifies evidence; M0 is the precondition for evidence. A run that fails M0
has no layer to be assigned, and a run that passes M0 has not yet earned one —
Layer 1 is still the first gate that can void a measurement, and Layer 0 is
still the first question about whether the measurement was necessary.

### The M0 freeze — which scripts the three things apply to

The three freezes above do not apply to every script in `scripts/`, and
asserting that they did would be false. What the freeze asserts instead is a
**partition**: every `measure_*.py` is either contracted or deliberately
exempt, and both halves are enumerated in `tests/test_contract.py` group 6, so
a script added with neither a contract nor an exemption fails the suite rather
than emitting unregistered numbers.

| script | contract | integrity gate | record | why |
|---|---|---|---|---|
| `measure_surrogate.py` | yes | yes | yes | contracted — the F3 arm comparison |
| `measure_learn.py` | yes | yes | yes | contracted — the F2 arm comparison |
| `measure_matched_h.py` | yes | yes | yes | contracted — the matched-H matrix |
| `measure_retrieval.py` | yes | yes | yes | contracted — the F1 arms, amended A1 |
| `measure_history_scaling.py` | exempt | yes | no | publishes the frozen `TACOSM-HS-001` reference table |
| `measure_retrieval_ceiling.py` | exempt | yes | no | publishes the frozen F0 reference table |
| `measure_baseline.py` | exempt | no | no | the untrained-by-design baseline |

The two exempt measurement scripts still run the integrity gate, and the
reason is the same as C4's: a frozen reference produced from untrained weights
is a frozen *wrong* number, which is precisely the failure that made a
published HS-001 table wrong once and had to be re-verified (see
`docs/CLAIMS.md` §C4).

`measure_baseline.py` is the exception to that exception, and it is correct
rather than a gap. Its arms are `oracle`, `random`, `static`, `full_context`
and `learned`; the first four need no weights at all, and the fifth trains
*inside* the run and is measured at the end of the schedule. An
`assert_trained` call there would be checking a router that is untrained by
design, and its protection is the arm set rather than the gate — the baseline
is a comparison against arms whose behaviour does not depend on training.
`test_baseline_has_no_integrity_gate_by_design` pins both absences so the
exemption is not found later as an oversight.

**The record's absence on the exempt side is a boundary, not a hole.** The
machine-readable record arrived with the contract system in `M1.0`, so it
covers exactly the contracted scripts. The audit trail for the frozen
reference is the published tables in `docs/CLAIMS.md` and
`docs/EVIDENCE_REGISTER.md` at the frozen commits — which are committed,
whereas `results/*.json` is gitignored. The committed table is therefore the
auditable form of those numbers. What the test pins is that any *future*
contracted script must reach the record, because a contract without an outcome
to compare against is a pre-registration nothing can check after the fact.

---

## Layer 0 — Inherited evidence

> **External or prior evidence that can be inherited rather than
> experimentally re-established.**

Layer 0 exists because the repository kept re-measuring things it had already
settled. Three experiments in a row — MATCHED-001, F2, F3 — each re-printed
the oracle control, the representability gate and the analytic-vector margins
as part of their own runs, because those numbers were load-bearing for their
interpretations. That is correct and will continue: an experiment that
*consumes* an inherited row verifies it as a gate. What is not correct is
*re-deriving* an inherited row as though it were open, which is how a
pre-registered design drifts into a post-hoc one — the diagnostic becomes a
finding, the finding becomes a claim, and the claim is weaker than the
evidence it replaced because it was measured under a protocol designed for
something else.

The layer answers five questions, and a row that cannot answer all five is
not Layer 0:

| Question | What an answer looks like |
|---|---|
| What is already established? | A named mechanism or property, not a number. Numbers are tied to a protocol; the mechanism is what transfers. |
| By whom, or by what source? | A commit, an experiment ID, or a provenance entry in `provenance/COMPONENTS.md`. |
| What exactly transfers? | The mechanism, the protocol, or the *lesson* — the three are different imports and the ledger records which one. |
| What does **not** transfer? | The exclusions. Every Layer 0 row carries its boundary, because an import without its exclusions is an over-claim. |
| What local validation remains necessary? | The gate the inheriting experiment must still run. Layer 0 is inheritable *evidence*, not an exemption from Layer 1. |

### The scale

`L0` inherits cleanly, with its boundary stated. `L1` is inherited but
required a local adaptation, so the adaptation is the contribution and is
re-verified locally. `L2` is a TAC-OSM mechanism result — established here,
by a named experiment, and inheritable by the next experiment but not from
outside. `L3` is an integration result about this specific system under test.
`L4` is a core claim of the program, which no single experiment establishes
and which the register exists to keep honest.

The register that assigns these levels is `docs/EVIDENCE_REGISTER.md`. This
document defines what a level *means*; that one records which row carries
which one.

### The rule Layer 0 imposes

**Before an experiment measures something, it must state which of its own
components are already settled.** The statement is not a literature review;
it is a table of the rows the design consumes, with the layer and the source
for each. An experiment that cannot fill in the table does not know which of
its own results would be new.

The corollary, and the part that bites: **an experiment does not get to
re-measure a Layer 0 row and report it as a finding.** It verifies it as a
gate, or it cites it. F3's reproduction gate — nine cells, `diff = 0.0000`
against the published MATCHED-001 baseline — is what an inherited row looks
like when it is being consumed honestly: the baseline arm exists to prove the
frozen reference did not move, and the deltas are then against that reference
rather than against a re-measured one.

### Where Layer 0 sits in the dependency

```
Layer 0 (inherited)  ──┐
                      ├──►  Layer 1 (gate)  ──►  Layer 2 (mechanism)  ──►  Layer 3 (system)
Layer 0 does not void  │       voids                withdraws                withdraws
the run; it is the     │
run's permission to    │
skip re-derivation     │
```

Layer 0 is not a gate and cannot void a run, because an inherited row can be
wrong without the current experiment being able to detect it — that is what
makes it inherited. What it can do is make a run *redundant*: an experiment
whose every component is L0 is not an experiment, it is a re-run, and the
register is what makes that visible before the compute is spent.

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
- `delta_K = s_gold − s_(K)`, the gap to the K-th *competing* score, for the
  retrieval budgets;
- `prob_margin`, the same gap in softmax units.

A `recall@16 = 0.55` is ambiguous on its own: gold may sit at rank 2 behind
one strong distractor, or be buried among a hundred near-tied candidates. Both
give the same recall and need different fixes. `delta_1` separates them
directly, which is what made the matched-H result legible.

**`delta_K`'s reference matters, and it had the wrong one until Audit 8.** The
metric was defined as the gap to the K-th competitor but implemented as the
gap to the K-th *overall* score, gold included — so `delta@1` was zero by
construction for any scorer that ranked gold first, and the analytic vector
reported `+0.0000` where its true margin is `+3.0`. The failure was invisible
because the key was never printed. See Audit 8 in
`docs/TACOSM-MATCHED-001.md`: a metric that is written but never read is not
verified by anything downstream, and its definition cannot be checked against
its computation. The corrected form is now printed, which is what makes it
checkable.

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
boundary (roadmap `M2.1`, historically `Stage F1`) matters more than a better
router: the boundary is what makes `|R|` a measurable quantity at all.

### What Layer 3 requires

A capability-vs-computation curve needs, at minimum:

- a retrieval boundary in the loop, so `|R|` is a defined quantity;
- total compute and relevant compute measured **separately** along the same
  axis, not one derived from the other;
- frozen history levels, with the capability parity margin declared **before**
  the confirmatory run;
- the full ablation matrix (`M1.3`), because a cost claim about an integrated
  system is not separable from which mechanisms are switched on.

Those conditions did not exist in v0.1, but later bounded synthetic controls
now measure retained-set size and execution work. They do not yet satisfy the
full L3/L4 burden for C5 because learned semantic admission, realistic
execution, and total end-to-end accounting are not jointly established under
the current universal protocol.

---

## How the layers interact

The dependency is one-directional and it is strict:

```
Layer 0 (inherited)  ──►  Layer 1 (gate)  ──►  Layer 2 (mechanism)  ──►  Layer 3 (system)
   does not void            voids                withdraws                withdraws
```

- **A Layer 0 failure voids nothing.** It costs capital. An experiment that
  re-derives an inherited row produces a correct number and a weaker claim,
  because the number was measured under a protocol designed for a different
  question. The register exists to make that visible before the run rather
  than after it.
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

### The pattern Layer 0 is designed to break

The opposite error, and the one the repository was making without noticing:
three consecutive experiments re-measured the analytic vector and the oracle
control as part of their own runs, in each case correctly, because the numbers
were load-bearing. Each run was longer than it needed to be, and in each the
*diagnostic* was at risk of being read as a *finding* — F3's own
pre-registration records the near-miss explicitly, in the section that says
re-running diagnostics that already have answers "is how a pre-registered
design drifts into a post-hoc one."

Layer 0 is the name for the discipline that was being applied by hand. The
difference between consuming an inherited row and re-establishing it is the
difference between a gate and an experiment, and the register is what keeps
them apart.


## Retrospective Audit 001 amendment

The layer definitions remain standing, but historical "empty gap" prose must
be read against the corrected portfolio state. Explicit temporal persistence,
retained-subset execution, and synthetic end-to-end selective controls now
exist as bounded L2/L3 evidence. The universal audit therefore treats this
file as the measurement-layer framework, while `docs/CLAIMS.md` and
`docs/RETROSPECTIVE_SCIENCE_AUDIT_001.md` are authoritative for current
scientific disposition.
