# TAC-OSM — Evidence Register

**Binding companion to `docs/MEASUREMENT_LAYERS.md`.** That document defines
what a layer *means*. This one records which row carries which one.

The register exists for one purpose: to stop the repository spending
experimental capital re-establishing results it already holds. Every row is
classified once, with its source and its boundary. A row's level does not
change unless the measurement behind it changes — not because a later
experiment would like a stronger premise.

---

## How to read a row

| Field | Meaning |
|---|---|
| `LAYER` | `L0` inherited cleanly · `L1` inherited, adapted locally · `L2` TAC-OSM mechanism result · `L3` TAC-OSM integration result · `L4` core program claim. See `MEASUREMENT_LAYERS.md` §"Layer 0". |
| `TYPE` | `mechanism` · `protocol` · `lesson` · `negative` · `gate`. A *mechanism* transfers as working machinery, a *protocol* as a way of measuring, a *lesson* as a constraint on interpretation only. |
| `PRIOR ART` | The external source, if any. Blank means no external antecedent — this is TAC-OSM's own evidence. |
| `NOT INHERITED` | The exclusions. The part of the source's evidence that does **not** transfer. This is the binding field: an import without its exclusions is an over-claim. |
| `REQUIRED EVIDENCE` | What the row needs to move up a layer, or what a claimant must produce to cite it. |
| `BLOCKER` | The specific thing standing between this row and the next layer. Blank means the row is at its terminal level or is already closed. |

**A row with `LAYER = L0` and a blank `REQUIRED EVIDENCE` is settled.** It is
cited, not re-measured. A row with `LAYER = L2` or `L3` and a `BLOCKER` is
live: the blocker names the next experiment.

---

## Layer 0 — inherited cleanly

Rows established outside TAC-OSM, or settled by a TAC-OSM experiment that is
now closed. The mechanism transfers; the numbers do not.

### The research infrastructure

Rows no experiment is asked to re-derive. They are inherited as *infrastructure*
— a way of measuring — which is why their `NOT INHERITED` field is the one that
matters most here: **infrastructure is not evidence.** A protocol row in this
subsection says a measurement is *checkable*, and checkable is a narrower claim
than *sound*.

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| A pre-registration the run itself checks — the registered design committed as data, and the measurement script refusing to complete a run that does not match it | L0 | protocol | pre-registration practice (the `L1` row below); the machine-readable and self-enforcing form is local | **contract validity is not scientific validity.** The check proves the run matched the registered specification; it does not prove the registered hypothesis is true. A contract-compliant run is a *well-specified* run, not a supported one. **An amendment is not a revision:** a contract may be amended before the run, with the replaced definition kept as a field, and that is a pre-registration being honest rather than one being edited | none — `src/tac_osm/contract.py`, `contracts/*.json`, enforced by `measure_surrogate.py`, `measure_learn.py`, `measure_matched_h.py`, `measure_retrieval.py` (`1e36d2e`, `d58fd68`, `eed6250`) | |

The distinction this row exists to hold is the one the repository's two
expensive errors blurred, in the opposite direction from the usual one.
TACOSM-HS-001 and `dd8f63c` were runs whose numbers did not refer to what the
docs said they did. Making the specification executable closes that hole — and
it opens a new one if the check is read as more than it is:

```
contract validity  ≠  scientific validity
a compliant run     ≠  a supported hypothesis
```

A contract pins what is held constant, what is being tested, and what counts.
It deliberately does **not** pin outcomes, because a contract that encoded the
expected result would be a foregone conclusion rather than a pre-registration.
So the strongest statement the infrastructure licenses is *"this number came
from the design that was committed before the run"* — which is a statement
about provenance, not about the world. No row in `CLAIMS.md` moves status
because a contract check passed; a claim moves when a measurement moves.

### External mechanisms

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| A persistent state container whose read/write is causally load-bearing | L0 | mechanism | TAC-transformer `IdentityState`, E3 (TAC-235/236, `6cce2ce`) | long-horizon memory (source evidence is bounded); the **write** across a temporal boundary — no source result exists | none — adopted as an interface | |
| The verify → localize → select → patch → re-verify loop shape | L0 | mechanism | TAC-transformer `ProceduralMemoryStore`, E3 (`b079ca5`) | the REAL017 lineage that used it is **do-not-cite until audited** (verifier received corruption labels, repair path received gold slots); only the *shape* transfers | none — adopted as an interface | |
| A structural executor over a DAG with copy-mask preflight | L0 | mechanism | CASM `phase1_dag`, E3 (`c315544`) | *learned* structural routing — Phase 1.5A failed 0/18 at loss 0.0012; the substrate transfers, the learned map does not | none | |
| State addressing by key, with a key-blind feature map as a named failure mode | L0 | lesson | PNDS Stage 3b-r2, E4; `dd8f63c` → `f989430` | nothing beyond the lesson | none — adopted as gate §34 | |
| A binary outcome reward `1[a_t = gold]` as the environment's own signal | L0 | mechanism | PNDS-GATE-001 Stage 2c, E2 (`42f9814`) | causality (Stage 2c is a bandit, not a decision loop); generality beyond `dim=8` bit descriptors | none | |
| A margin-based ranking objective `s_gold − max_{j≠gold} s_j` | L0 | mechanism | standard ranking-loss practice | nothing — the objective is standard; the adaptation is what is new (see L1) | none | |
| A state-independent reward baseline subtracted for variance reduction | L0 | mechanism | standard policy-gradient practice | nothing — the technique is standard; *where* the baseline is taken is the adaptation | none | |

### TAC-OSM results that are now closed

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| The relevance relation is representable in the hypothesis class, at every H tested | L0 | gate | — | `analytic_weights` is a hand-designed reference, **not** an outcome-trained router; routing with it is not a TAC-OSM result | none — settled by the §34 gate and the analytic vector (`routing@1 = 1.0000` at H ∈ {8,64,256,512}, `delta_1 ≈ +3.07`) | |
| The task environment is unambiguous | L0 | gate | — | oracle accuracy is a property of the generator, not of any router | none — `oracle = 1.0000` at every H, printed first in every run | |
| Gold is the unique satisfier and is not exposed to the router | L0 | gate | — | nothing | none — `leakage.py`, structural | |
| The environment's success metric is `action == target_action`, and the baseline reward is therefore `1[a_t = gold]` | L0 | mechanism | — | nothing — this is the fact that makes F3's leakage boundary statable in terms of density rather than gold-absence | none | |

---

## Layer 1 — inherited, adapted locally

The antecedent is external. What TAC-OSM contributes is the *adaptation*, and
the adaptation is what is re-verified locally rather than cited.

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| A key-gated feature map over a linear scorer, as the router's basis | L1 | mechanism | `RelationalBanditRouter` (PNDS Stage 2c, E2, `42f9814`) | the source's causality claim (bandit, not loop); that the learned arm reaches the oracle ceiling (source: 0.8853 vs 1.0 at 8 candidates) | local verification that the basis represents the relation — the §34 gate, run on the real basis | none — the gate passes |
| `delta_1` and `delta@K` in raw dot-product units, not softmax units | L1 | protocol | F0 introduced the margins; the unit correction is local | a probability-unit margin shrinks with H even for a perfect scorer, so a cross-H comparison in those units measures the normalisation | none — corrected and printed (Audit 8, MATCHED-001) | |
| The policy-gradient baseline taken over the surrogate's own running mean | L1 | mechanism | standard variance reduction (L0 above) | the specific quantity being averaged: `mean(s_{<t})` of the surrogate, not `1/H` of the candidate count and not `mean(scores)` | local verification that centring is not a constant offset — the probe that found `mean(scores)` gave a positive reward on 180/180 steps | none — replaced with the running mean, verified |
| A pre-registered decision rule with its consequences attached, committed before the run | L1 | protocol | PNDS-URP v0.4 pre-registration practice | nothing | none — used by MATCHED-001, F2 and F3 | |

---

## Layer 2 — TAC-OSM mechanism results

Established here, by a named experiment. Inheritable by the next TAC-OSM
experiment, **not** from outside.

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| Temporal persistence survives enforced decision boundaries in the synthetic state control (C1) | L2 | mechanism | TAC-transformer IdentityState provides the state-container vocabulary | semantic/learned long-horizon memory is not inherited | TACOSM-TEMPORAL-001 run 36509506303: causal gate plus carry/reset/corrupt comparison through k=32 | **bounded to the synthetic vector-store control** |
| A cheap linear router learns the relation from outcomes alone (C2) | L2 | mechanism | — | large-H behaviour — this result is at the small population where outcome supervision is dense | `TACOSM-BASELINE-001`: 0.4396 vs `random` 0.1296, `static` 0.0376 | |
| Accuracy falls with history because routing degrades, not because the task hardens (C6) | L2 | mechanism | — | the *mechanism* — the original "linear-scorer capacity" reading is **refuted**; what degrades is the trained router under a sparse reward | `TACOSM-HS-001`, oracle 1.0000 at every H | |
| The top-K signal survives top-1 collapse at `H ≤ 64` (C7) | L2 | mechanism | — | `H ≥ 128`, where it does **not** survive on a transfer-trained router | `TACOSM-RETRIEVAL-001` F0 | |
| **Selection is not the binding constraint** — exploration that finds more successes still does not lift the endpoint | L2 | negative | — | any reading stronger than "these two named interventions do not fix the trained router at H=256" (C10) | `TACOSM-LEARN-001` (F2): `epsilon_greedy` explored 15.2% of steps, 3.8 vs 2.4 successes per 500, endpoint flat | |
| **The large-H top-K signal is responsive to training** — it improves materially under a training-only intervention | L2 | mechanism | — | `K = 1`, which did not move; and any reading that the *mechanism* now scales | `TACOSM-SURROGATE-001` (F3): `Δ(recall@16)` = +0.0860 / +0.2460 at H=256 | |
| Scheduler and family-stream integrity (C8) | L2 | gate | — | nothing | `test_relational_and_state_lookup_do_not_collide_on_a_pinned_run` | |

---

## Layer 3 — integration results about this system

A property of TAC-OSM v0.1 specifically, under a named protocol. These are
the rows most likely to be over-claimed, and the register's job is to keep
their bounds attached.

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| Matched-H training is the **worst** row at every evaluation population | L3 | negative | — | any generalisation beyond the registered protocol; the matched-H design is one population schedule | `TACOSM-MATCHED-001`: `routing@1` at H=256 is 0.0740 vs 0.0640 for the H=8-trained row | |
| The learning rule is the failure, not the population and not the basis | L3 | mechanism | — | this localises the failure; it does not identify the fix | `TACOSM-MATCHED-001`: 5 successes in 500 steps at H=256, weight norm ×12, margin *worsens* | |
| **Dense supervision improves top-K retrieval but not top-1 routing** | L3 | mechanism | — | "the router learned relevance" — the surrogate is gold-anchored exactly as the outcome reward is | `TACOSM-SURROGATE-001` (F3): primary endpoint +0.0400/+0.0240 within a 0.0600 spread; `recall@16` material; `gold_rank` 74.70 → 25.83 | |
| `C_total` in v0.1 is `O(H) routing + O(10) execution + O(verification)` | L3 | negative | — | the *desired* cost model `C(\|R\|)` — a flat `C_executed` column means no room to scale, not that scaling was shown | `TACOSM-HS-001` | the confirmatory F1 run — the boundary is built, the curve is not measured |
| **`TACOSM-C5-001` is VOID for capability inference** — a confirmatory run that could not distinguish execution capability from base-rate behaviour | L3 | negative | the C5-001 design itself | **any** capability reading, positive or negative; and the 128× work reduction as a C5 result (see the row below) | run 36512989760 with the pinned CASM-S: exhaustive collapsed to the population base rate (0.2500, spread 0.0000) and indexed 1.0000 is guaranteed by the acceptable-action bucket; verification 0.0000 in all 75 cells | **instrument, not mechanism** — the soft-op `alpha` scaling, gate routing, the 0.5 threshold and the bridge target are unresolved. `TACOSM-C5-002` is required |
| **`TACOSM-C5-002`'s representability gate fired** — the instrument is invalid for measuring CASM-S execution capability on this graph family, and the gate terminated the run before a capability number could be produced | L3 | gate | the C5-002 pre-registration (`docs/TACOSM-C5-002.md`) | **any** capability reading, positive or negative; and any change to the pre-registered `GATE_MIN_ACCURACY = 0.5` after the fact, which would convert the gate from a termination condition into a tuning knob | run 36536249559 with the pinned CASM-S at `c315544`: the gate separated **59 of 256** held-out pairs (0.2305) against a minimum of 0.5, so the task stream never ran (`"cells": []`); the same bridge reports **0.84375** absolute output accuracy, the quantity C5-001 mistook for capability | **bridge training target** — the bridge is trained on single candidates against their Boolean output, and nothing in that objective rewards within-query ranking, so 0.84375 absolute and 0.2305 separation are both "as trained". The soft-op `alpha` scaling, the 0.5 threshold and unlearned gate routing remain unresolved and unseparated by any run. The successor needs a pair-trained bridge, a new checkpoint and a new contract |
| **`TACOSM-C5-003` is pre-registered, not run** — the pair-trained bridge and the eight-criterion gate registered as the successor to C5-002's fired gate | L3 | gate | the C5-003 pre-registration (`docs/TACOSM-C5-003.md`) | **any** capability reading, positive or negative; and reading the registration as progress on C5 — a registered design is a well-specified intention, not a measurement | **no measurement has been made.** The registration is a contract (`contracts/TACOSM-C5-003.json`, status `pre-registered`, layer L4) and a pre-registration document, both committed before any run. Nothing has executed against them. The gate's eight criteria — frozen thresholds, held-out pair accuracy, output spread, verifier acceptance, no `true_edge_set` dependence, no oracle leakage, checkpoint determinism, pair-trained objective — are registered as constants, and none has been observed to hold | **the measurement has not been run** — the script now exists (`scripts/measure_c5_casm_003.py`, registered in `_WITH_CONTRACT`) with 30 torch-free design tests and a `--smoke` contract check, but the measurement itself requires `torch` and the pinned CASM-S checkout, so it is runner-only. The compute lane is `.github/workflows/tacosm-c5-003.yml` (branch `compute/c5-003`), and no run has been dispatched. C5's blocker stays the instrument until a gate-confirmed run exists |
| CASM-S execution work submitted by the pipeline scales with the retained set, not with the population — **an instrument-level observation** | L3 | mechanism | — | that the retained structures are the *right* structures, or that CASM-S computed the intended relation; and any reading as C5 progress | `TACOSM-C5-001` run 36512989760: 256 → 2 structures/query and 12288 → 96 edges/query at H=256, K=2 (128×), counted from the serialized graph rather than the model output | voided as capability evidence by the row above. The two rows stay separate because the work accounting does not depend on the output being semantically correct |

---

## Hardened experiment prerequisites

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| The write/read delay is enforced in the executable benchmark | L1 | gate | persistent-state temporal vocabulary | implementation is not capability evidence | TACOSM-TEMPORAL-001 boundary gate | run the registered temporal measurement |
| The retrieval boundary is part of the runtime call graph and its one-time build cost is separately accounted | L2 | protocol | deterministic lookup as a sparsity control | exact equality index is not semantic retrieval | TACOSM-SELECTIVE-001 with static candidate-universe reuse | run the registered selective measurement |
| Structured verifier and bounded executable repair can re-execute without hidden gold access | L2 | mechanism | verifier-guided repair loop shape | no capability claim transfers from this control | dedicated repair evaluation | capability evaluation not run |

## Layer 4 — core program claims

No single experiment establishes these. They are the claims the register
exists to keep honest: each one is supported only by the *accumulation* of
Layer 2 and Layer 3 rows, and each one has a standing blocker.

| Row | LAYER | TYPE | PRIOR ART | NOT INHERITED | REQUIRED EVIDENCE | BLOCKER |
|---|---|---|---|---|---|---|
| Runtime retrieval boundary can preserve capability while bounding router input in an exact synthetic control | L3 | integration | deterministic lookup is an L0 control mechanism | semantic retrieval, learned addressing, and end-to-end compute scaling are not inherited | TACOSM-SELECTIVE-001 run 36509506303: indexed success=1.0 across H={8,64,256}, K={2,4}; actual router input bounded by K | **does not establish C5** |
| Persistent state makes useful computation reusable across time | L4 | — | — | no temporal capability evidence yet | the hardened write-at-t/read-at-t+k benchmark plus carry/reset/shuffle/corrupt controls | **temporal persistence measurement not run** |
| The integrated system is more economical than full-context at capability parity | L4 | — | — | everything — this is the program's thesis, not a result | the curve in `ROADMAP.md` §"M3.1", measured | all of the above |

---

## What this register forbids

**A Layer 2 or Layer 3 row may not be cited as Layer 0.** F3's top-K result
is L2 within TAC-OSM and is *not* evidence about any other system. The
register records the level so the boundary travels with the citation.

**A negative row may not be promoted by accumulation.** F2's negative on two
exploration interventions and F3's negative on reward density together
constrain *where* the failure is. They do not together become "the mechanism
does not scale" — claim C10 exists to hold that line, and the hypothesis class
is adequate at every H tested, which is what makes the distinction load-
bearing rather than rhetorical.

**A row's level does not change when a document is rewritten.** An entry moves
when the measurement behind it moves: a re-run, a new experiment, or a
discovered boundary. Prose is not evidence, and a claim found in a document
with no row here is unsupported by definition — the same discipline
`CLAIMS.md` already imposes.

---

## The difference between this register and the other ledgers

Three documents now record provenance, and they are not redundant:

* **`provenance/COMPONENTS.md`** — what was *imported*, from which repository,
  at which commit, with `reuse` and `do_not_claim`. It is about code and
  external lineage. It does not classify TAC-OSM's own results.
* **`docs/CLAIMS.md`** — every claim the repository *makes*, with a status and
  a blocker. It is the scientific ledger, and it answers "is this supported?"
* **`docs/EVIDENCE_REGISTER.md`** (this document) — the *level* of each piece
  of evidence, answering "may this be inherited, or must it be measured
  again?" It is the pre-experiment gate that the other two cannot serve as:
  CLAIMS records whether a claim stands, not whether the next experiment is
  allowed to take it as given.

The three agree by construction: a CLAIMS entry cites its experiment, the
experiment's rows appear here, and the imported components appear in the
provenance ledger. A disagreement between them is a bug.

A fourth, narrower ledger now exists alongside them, and it is the one that is
checked by running rather than by reading:

* **`contracts/*.json`** — the *specification* of each published measurement,
  committed before the run and enforced by the measurement script itself. It
  is not evidence about anything and carries no layer of its own; it is the
  instrument that makes the rows above verifiable. Its limit is stated in its
  row above and bears repeating: it certifies that a run matched its design.
  It says nothing about whether the design was a good idea, or whether the
  hypothesis the design was built to test is true.
