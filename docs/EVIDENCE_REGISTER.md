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
| **`TACOSM-C5-003` is pre-registered, not run** — the pair-trained bridge and the eight-criterion gate registered as the successor to C5-002's fired gate | L3 | gate | the C5-003 pre-registration (`docs/TACOSM-C5-003.md`) | **any** capability reading, positive or negative; and reading the registration as progress on C5 — a registered design is a well-specified intention, not a measurement | **no measurement has been made.** The registration is a contract (`contracts/TACOSM-C5-003.json`, status `pre-registered`) and a pre-registration document, both committed before any run. Nothing has executed against them. The gate's eight criteria — frozen thresholds, held-out pair accuracy, output spread, verifier acceptance, no `true_edge_set` dependence, no oracle leakage, checkpoint determinism, pair-trained objective — are registered as constants, and none has been observed to hold | **the measurement has not been run** — the script exists (`scripts/measure_c5_casm_003.py`, registered in `_WITH_CONTRACT`) with torch-free design tests and a `--smoke` contract check, but the measurement itself requires `torch` and the pinned CASM-S checkout, so it is runner-only. The compute workflow is `.github/workflows/tacosm-c5-003.yml`; its `compute/c5-003` push branch does not currently exist, while `workflow_dispatch` remains available. C5's blocker stays the instrument until a gate-confirmed run exists. **The contract carries four amendments, all made before any run:** A1 records the C5-002 validation/gate seed collision and gives C5-003 a distinct validation seed; A2 registers criterion 4 on the satisfying half because the verifier rejects violating descriptors unconditionally; A3 carries the model output into criterion 4's `Outcome.value` because the verifier's `1e-9` consistency check would otherwise require exact `1.0`; A4 carries that same output convention into the real task-stream verification path, separates `execution_accuracy_rate` from hidden action-selection success, adds `selection_success_rate`, and strengthens criterion 7 to repeat the held-out execution and compare the returned output/gate/node records. No confirmatory C5-003 result exists under any of these definitions.
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
| Persistent state makes useful computation reusable across time | L4 | — | — | no general temporal-persistence claim is inherited from the current E2E-008 result | E2E-008 plus a task where post-action information is necessary, with carry/reset/shuffle/corrupt controls | **E2E-008 reset/recompute found no temporal-carry advantage; longer-horizon/post-action-necessary measurement remains required** |
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


---

## E2E-006 fresh confirmatory integration evidence

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| E2E-006 mean q2 = 0.8665 on a fresh held-out composition set | L3 | integration | the selected hidden_dim=64 integrated multimodal chain crosses the preregistered 0.80 synthetic capability threshold | real-world multimodal understanding, semantic memory, learned addressing, learned operator discovery, scaling or efficiency | fixed public operator dispatch and deterministic entity addressing remain scaffolds; independent validation must pass |
| No-memory/control drops | L3 | localization | persistent state, image alignment, and all three modalities contribute under the registered intervention protocol | causal generality outside this synthetic benchmark | controls reuse identical evaluation episode objects; diagnostic endpoints are not primary |
| Fresh held-out split excludes sealed E2E-005 and development compositions | L3 | protocol | the E2E-006 result was not selected on the same test compositions | universal generalization | E2E-006 tests only the registered fresh held-out compositions |
| Classifier-decision consistency and zero train/eval semantic overlap | L3 | gate | the measured number is not explained by the previously discovered probability-to-logit evaluator defect or obvious episode-key leakage | absence of all possible implementation leakage | the integrity gate constrains known attack surfaces; it is not a proof of arbitrary absence of leakage |

The C15 result is therefore a bounded Layer 3 integration result. It moves the
immediate blocker from “can the integrated chain function at all?” to “can the
same chain function when explicit addressing and operator scaffolds are
relaxed?” The next lane should attack those scaffolds without reusing the
E2E-006 held-out outcomes for selection.

---



## E2E-007 residual-state integration evidence

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| E2E-007 mean q2 = 0.9350 on a new held-out composition set | L3 | integration | the development-selected residual-linear state-write configuration works end to end on a second fresh synthetic benchmark | real-world multimodal understanding, semantic memory, learned addressing, learned operator discovery, scaling or efficiency | operator dispatch and entity addressing remain explicit scaffolds |
| Mean memory drop = 0.3135 with all modal controls below normal | L3 | localization | persistent state and multimodal observations materially contribute under the registered interventions | causal generality outside the synthetic benchmark | controls are diagnostic, not independent primary endpoints |
| XOR reaches 1.00 across every seed while AND remains 0.71–0.77 | L3 | mechanism diagnostic | the prior XOR state-content bottleneck was substantially repaired; the current residual failure moved toward AND | general Boolean-computation superiority | per-operator endpoints are secondary and not separately preregistered primaries |
| Residual-linear development selection: mean q2 0.9080 vs linear 0.8470 | L2 | mechanism | state-write transformation is a measurable optimization lever in this synthetic implementation | universal superiority | selection occurred on the registered development split only |

The evidence frontier has therefore moved from “make the integrated chain work”
to “remove its remaining explicit scaffolds and diagnose operator-specific
failure without reusing confirmatory outcomes.”

---

## E2E-001 post-measurement evidence disposition

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| Integrated multimodal E2E-001: mean q2 = 0.5200, seed-bootstrap 95% CI [0.5020, 0.5385] | L3 | negative | the registered synthetic integrated implementation failed its 0.80 primary criterion | no general PLM impossibility; no real-world multimodal claim; no scaling claim | all five seeds retained; artifact 11251182171; workflow 37062142549 |
| Memory and image-alignment controls exactly equal the integrated arm | L3 | localization | no measurable persistence or image-alignment contribution under this protocol | no claim that persistence/fusion is useless generally | paired evaluation episodes and zero bootstrap drops |
| Target memory attention ≈ 0.06094 with 16 slots | L3 | diagnostic | supports the hypothesis that target-specific state addressing did not emerge | not a formal hypothesis test of uniform attention | uniform reference is 0.0625; attention diagnostic was secondary |
| Operator selection = 1.0000 | L3 | diagnostic | query-label echoing is working | autonomous operator discovery/routing | operator ID was explicit and supervised |

The evidence therefore moves the immediate blocker from "can the integrated chain
be trained?" to "can a trainable persistent-state address space be made coherent
between write and read?" The next experiment must test that interface directly.


---

## Integrated E2E generator audit — invalidation

| Evidence | Layer | Type | Disposition | Reason |
|---|---|---|---|---|
| E2E-001 measured q2=0.5200 and controls | L3 | negative/localization | **VOID** | shared generator ignored the query entity argument and made q1 and q2 target the same entity, violating the registered temporal protocol |
| E2E-002 address diagnosis run 37067955696 | L3 | pre-confirmatory | **VOID / no scientific result** | inherited the same malformed generator; runner failed before artifact creation |
| E2E-001 artifact 11251182171 | provenance only | historical artifact | retained but non-evidentiary | historical reproducibility record only; no number may be used for tuning or baseline selection |

The corrected experiment must independently generate q1 and q2 and assert
that their entity identities differ before any measurement is reported.


---

## E2E-003 corrected temporal/address diagnosis

| Evidence | Layer | Disposition | What it supports | Boundary |
|---|---|---|---|---|
| explicit-write, explicit-read, explicit-both arms | L3 | VALID | correct structural addressing did not improve held-out q2 capability | no learned semantic addressing claim |
| explicit-both paired controls | L3 | VALID | no measurable persistence, image-alignment, or unimodal contribution under this protocol | no general multimodal/memory impossibility claim |
| combined seed results | L3 | VALID | all three arms produced the same five q2 values | bounded to corrected synthetic benchmark |

Primary explicit-both q2 = **0.5200**; seed-bootstrap 95% interval
**[0.5020, 0.5385]**; minimum seed **0.4925**.

Next diagnostic: explicitly align state content representation with CASM rather
than changing addressing again.


---

## E2E-008 clean collision-free integrated evidence

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| E2E-008 q2 = 1.0000 across five seeds and 12 held-out compositions | L3 | integration | the integrated PLM executes the registered synthetic multimodal task end to end | real-world multimodal understanding; learned memory; general scaling | benchmark is synthetic and scaffolded; score is saturated |
| All four fixed operators reach 1.0000 | L3 | mechanism diagnostic | the integrated execution path handles the registered AND/OR/XOR/XNOR compositions on this benchmark | learned operator discovery or open-ended program synthesis | operator identity is public/query-conditioned and Boolean execution is closed-form CASM |
| Benchmark identity and leakage gates pass | L3 | validity gate | the E2E-008 result is tied to the collision-free generator and clean held-out split | arbitrary absence of all possible leakage | known image/entity side-channel collision was removed; train/eval semantic overlap was zero |
| Five exact evaluation fingerprints and benchmark SHA-256 are later reproduced by the reset/recompute run | L3 | provenance | attribution uses the same evaluation objects as E2E-008 | new generalization evidence | reset/recompute is an attribution follow-up, not a fresh capability test |

---

## Matched feed-forward architectural controls

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| FF-002 mean q2 = 0.6287 versus PLM 1.0000 with disjoint composition-bootstrap intervals | L3 | architectural attribution | a bounded integrated-PLM advantage over the registered feed-forward control | temporal persistence as the unique cause | FF-002 removes persistent state/CASM/verifier/state update machinery as well as temporal carry |
| FF-002 receives detached q1 environment outcome | L3 | confound control | giving the baseline the same cross-boundary outcome information does not close the gap | equality of mechanisms | q1 outcome is detached; the control remains feed-forward |
| Parameter count 74,124 vs 74,126 | L3 | matched-control gate | the comparison is effectively parameter matched | compute/runtime equivalence | parameter match is not architecture equivalence |
| FF-001 mean q2 = 0.6317 and FF-002 = 0.6287 | L3 | replication/diagnostic | direct q1 feedback does not rescue the feed-forward family | causal attribution of persistence | FF-001 is diagnostic; FF-002 is the stronger registered control |

---

## Reset/recompute temporal-carry attribution

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| Normal and reset/recompute q2 = 1.0000 across all five seeds | L3 | **uninformative design null** | the specific post-q1 outcome-gated update is not required to reach the saturated E2E-008 ceiling | temporal carry is not load-bearing in an unsaturated task; persistence is useless generally | the reset arm retains all original observations and q2 is at ceiling |
| Normal-minus-reset = 0.0000; paired composition-bootstrap 95% CI = [0.0000, 0.0000] | L3 | **uninformative endpoint** | no measurable gap at a saturated ceiling | evidence that temporal carry is unnecessary under a task with headroom | threshold was 0.05, but the ceiling makes the comparison non-discriminating |
| q1 prediction/action/outcome equality and recomputed-state identity both pass | L3 | intervention gate | the reset intervention removes the post-q1 outcome-gated update without changing the q1 path or pre-q1 state | a clean test of whether q1-time information itself must be carried | the reset arm still reconstructs from all original observations |
| Independent validator passes; artifact 11469064681; run 37593010244 | L3 | provenance/validity | the measured intervention was executed and reproduced cleanly | a definitive memory-versus-compute conclusion | the run is retained as an instrument-valid but non-discriminating ceiling result |



| Learned-address v1 full confirmatory attempt run 37623028125 | L3 | **instrument failure** | the lane reached smoke successfully but the full measurement aborted on the post-shuffle target-slot balance assertion | any learned-address capability conclusion | the run uploaded only a smoke artifact after the measurement failed; the smoke artifact must not be treated as a measured result |
---

## Latent operator induction evidence

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| Hidden operator removed from model input; q2 mean = 0.9293 | L3 | scaffold-removal | the integrated state/CASM path can operate without explicit operator-ID dispatch on the registered synthetic task | robust learned operator discovery or an intrinsic XNOR-difficulty claim | seed 3 q2 = 0.7117 and operator selection = 0.75; only five seeds were preregistered |
| Operator-selection mean = 0.9500; composition-bootstrap 95% CI [0.9000, 1.0000] | L3 | mechanism | the support truth-table context is sufficient for strong operator inference in most seeds | uniform all-seed robustness | preregistered minimum was 0.80; seed 3 failed |
| Shuffle-support q2 mean = 0.6793 versus normal = 0.9293 | L3 | context-use diagnostic | performance depends materially on the supplied support context | causal generality of the support mechanism | shuffle is a diagnostic intervention, not a new capability benchmark |
| Oracle-operator q2 mean = 0.9740 | L3 | upper-bound/localization | about 2.6 percentage points of q2 accuracy are lost even with operator selection supplied; seed-3 weakness is therefore substantially selection-related | learned operator discovery | oracle supplies the hidden operator only for diagnostic measurement |
| No-memory q2 mean = 0.6250 | L3 | mechanism diagnostic | the multimodal state representation remains functionally important | temporal persistence necessity | temporal carry was separately ruled out as necessary on E2E-008 |


## Random-key addressing baseline 001

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| `TACOSM-ADDRESS-BASELINE-001`, run 37625321076, artifact 11484519604 | L3 | valid mechanism baseline | raw-dot retrieval reference across d, M and isotropic Gaussian query noise | learned addressing or semantic/content addressing | raw dot is the Bayes/ML ranking rule under the registered isotropic Gaussian model |
| d=16 capacity/noise grid, five seeds × 100,000 trials/condition | L3 | quantitative baseline | retrieval degradation as M and noise increase | PLM efficiency or learned routing claims | random-key associative retrieval only |
| d=16, sigma=0.8 drops from 0.6976 at M=3 to 0.0544 at M=256 | L3 | capacity diagnostic | the registered random-key channel has a sharp memory/noise interaction | universal memory limit or semantic retrieval scaling | specific to the registered Gaussian key model |
| Commit 3ced0b2646340f35383fb0eb3ab9ad79f69a692b; contract SHA 0822928b35a6855bd43f7b693e79208b1d88544b02b3ba978d88835bf5098002 | L3 | provenance | exact rerun lineage is recorded | immunity to future implementation changes | branch/commit must remain pinned for reproduction |

## Pure temporal-persistence follow-up — valid measurement

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| `TACOSM-PLM-TEMPORAL-PERSISTENCE-002`, run **37637018951**, artifact **11490971609**, measured commit `a5129dc4f8e241de44902fba9e1edcb817856f57` | L3 | valid measurement | PLM-specific load-bearing persistence on the registered synthetic multimodal task | general memory, semantic memory, MTSK, addressing, operator discovery, scaling, real-world multimodal competence | explicit entity addressing and fixed XOR remain in force |
| Carry q2 mean **1.0000**; fresh q2 mean **0.5000**; delta **+0.5000** | L3 | primary mechanism result | retaining the q1-time q2 entity materially changes later q2 correctness | universality of persistence | all five seeds show the same arm-level result |
| Hierarchical paired bootstrap 95% CI **[0.48167, 0.51800]** | L3 | uncertainty | the paired effect remains positive under seed-and-episode resampling | population-wide effect size | five preregistered seeds; synthetic episodes |
| All information-flow and leakage gates passed | L3 | integrity | q1 cannot read the q2 secret, the secret is masked after the boundary, carry/fresh receive the same q2 query and post observation, and train/evaluation semantic overlap is zero | absence of every conceivable implementation flaw | artifact gates independently passed |
| State mechanism was `ExplicitEntityState` with residual-linear writes | L3 | architectural boundary | persistence evidence belongs to the current single-scale state path | MTSK validation | MTSK remains a separate unmeasured mechanism |

## MTSK status — explicitly unmeasured

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| Current PLM E2E state implementation is `ExplicitEntityState`; no MTSK module or MTSK experiment exists in the measured commit | L3 | architectural audit | MTSK should not be credited to the current persistence result | that MTSK is unnecessary or ineffective | MTSK needs its own preregistered comparison against a single-timescale control |

## C19 20-seed robustness extension — valid measurement

| Evidence | Layer | Type | What it supports | What it does not support | Required caveat |
|---|---|---|---|---|---|
| 20-seed extension, latest run 37633578443, artifact 11489055422 | L3 | valid robustness | recurrent latent-operator behavior across a larger pre-specified seed panel | operator synthesis; semantic addressing; real-world multimodal competence | C19's original decision remains unchanged |
| Operator-selection mean 0.9375; min 0.75 | L3 | robustness | C19 seed instability is recurrent rather than isolated to one seed | intrinsic operator difficulty | all failures are XOR/XNOR pair confusion |
| Aggregate selection: AND 1.00, OR 1.00, XNOR 0.90, XOR 0.85 | L3 | mechanism diagnostic | operator selection failures are concentrated in the XOR/XNOR pair | generalized XNOR difficulty | 3,000 evaluation episodes per operator across 20 seeds |
| Q2 mean 0.93275; min 0.71167 | L3 | primary robustness descriptor | latent-operator task remains strong on average but has meaningful seed variance | robust all-seed success criterion | descriptive extension, not a new capability threshold |
| Mean no-memory 0.62692; oracle q2 0.98233; oracle gap 0.04958 | L3 | localization | state/content computation remains useful and operator selection is not the only source of error | causal separation of all model components | diagnostics only |
| Training operator counts sum to exactly 28,800 per seed | L3 | integrity gate | training exposure accounting is now directly verified | absence of every possible bias | counts cover sampled combo2 stream only |


## Temporal-dependency pilot 001 — instrument failure

| Evidence | Layer | Disposition | Reason | Scientific consequence |
|---|---|---|---|---|
| `TACOSM-PLM-TEMPORAL-DEPENDENCY-001`, run **37633578402** | L3 | **INSTRUMENT FAILURE** | The measurement aborted on the preregistered train/evaluation semantic-overlap gate before producing an artifact | No temporal-dependency result may be inferred from this run; the stricter persistence experiment remains the admissible test |
| Earlier v1 runs with syntax/variable-bound failures | L3 | **INSTRUMENT FAILURE** | Static or runtime harness defects prevented a valid measurement | No scientific interpretation |
