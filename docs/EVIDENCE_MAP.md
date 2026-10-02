# Evidence Map — the pre-experiment gate

**Binding companion to `docs/EVIDENCE_REGISTER.md` and
`docs/MEASUREMENT_LAYERS.md`.** This document is the gate an experiment passes
*before* it runs. The register records the level of each row; this document is
the procedure that makes a new experiment confront those levels before it
spends capital on them.

The map has two jobs that it did not have before. It still traces every
interface to the evidence that licenses it and to the boundary of that
evidence. It is now also the thing that answers: **is an experiment actually
necessary?**

---

## The gate

Run this before designing an experiment, and record the answers in the
experiment's pre-registration:

```
Proposed experiment
       │
       ▼
What claim does it support?
       │
       ▼
What evidence already exists?          ──► EVIDENCE_REGISTER.md
       │
       ▼
What can be inherited?                 ──► rows at LAYER L0 or L1
       │
       ▼
What remains unknown?                  ──► the row's BLOCKER, if any
       │
       ▼
L0 / L1 / L2 / L3 / L4                 ──► the layer the *new* row would carry
       │
       ▼
Is an experiment actually necessary?
       │
       ├─ no  ─► every component is L0/L1 and the blocker is elsewhere:
       │          the run is a re-run. Cite the register instead.
       │
       └─ yes ─► the L2/L3 rows this experiment would create are named now,
                 in the pre-registration, before any arm runs.
```

**The last box is the one that bites.** An experiment whose every component is
`L0` or `L1` is not an experiment — it is a re-run, and it will produce
correct numbers attached to weaker claims, because the numbers were measured
under a protocol designed for a different question. The register is what
makes that visible before the compute is spent rather than after.

**The discipline runs the other way too.** An experiment that *would* create a
new `L2` or `L3` row must name that row in its pre-registration, with the
layer and the blocker attached. An experiment that cannot name the row it
would create does not know what it is measuring, and a row created after the
run and fitted to the outcome is not evidence — it is a description.

---

## Map — capability to evidence

The evidence scale is the portfolio's: `E0` idea → `E1` implemented → `E2`
smoke → `E3` controlled → `E4` reproduced → `E5` cross-condition → `E6`
cross-domain, per `docs/pnds/RESULTS.md` in `cdl-attention-experiment`. The
`LAYER` column is this repository's scale and the two are not the same: `E`
grades a source experiment's maturity, `L` grades whether a TAC-OSM experiment
may inherit it.

| Capability | Existing implementation | Evidence | LAYER | Integration status |
|---|---|---|---|---|
| Persistent state | TAC `IdentityState` | E3 — carry/reset/shuffle probes, TAC-235/236 | L0 | **reusable** |
| State addressing | CDL / PNDS Stage 3b-r2 | E4 — 10/10 seeds, `p=0.001`, key-reset → 0.1250 | L0 | **reusable** |
| Relevance routing (teacher) | CDL `app.py` | E5 — Top-1 0.875 vs 0.633 gzip, 6/12/24 candidates | L0 | **teacher only** |
| Cheap routing | CDL Stage B `QKRouter` | E4 — distilled 73.75% vs direct 77.92%, `p≈0.275` | L0 | **unresolved** |
| Structural execution | CASM `phase1_dag` | E3 — Gates 0–6, copy-mask preflight | L0 | **reusable substrate** |
| Learned structural routing | CASM Phase 1.5A L2 | **failed 0/18** — loss 0.0012, wrong basin 0.93–0.97 | L0 | **failed** |
| Learned conditional routing | PNDS Stage 2c | E2 — 0.8853 vs static 0.0000, t=258 | L1 | **strongest router evidence** |
| Verification | TAC `VerifierGuidedRepairController` | E3 — TAC-267→274 | L0 | **provisional** |
| Repair | TAC `ProceduralMemoryStore` | E3 — bounded sandbox repair | L0 | **candidate** |
| Persistent write | `src/tac_osm/state.py`, `src/tac_osm/unified.py` | implementation-level | L1 | **implemented; not yet independently causal** |
| Temporal state transition | `TACOSM-TEMPORAL-001` | E3-equivalent TAC-OSM controlled result | L2 | **validated bounded explicit write/read persistence through k=32** |
| Verified state commit | `src/tac_osm/unified.py::KnowledgeStore` | implementation-level | L1 | **implemented; causal contribution still unmeasured** |
| Path-level verification | `src/tac_osm/verifier.py` | implementation-level | L1 | **implemented; comparative measurement not run** |
| Full `I→D→Z→M→P→R→Pi→C→A→O→V→W→L` loop | unified native model v1 | — | — | **new research target** |

---

## Map — TAC-OSM's own results, and who may inherit them

These rows did not exist when this map was first written. Each one was created
by a named experiment, and each is listed here because the next experiment
must decide whether to inherit it or re-measure it.

| Result | Experiment | LAYER | Inheritable by | NOT inheritable as |
|---|---|---|---|---|
| The relation is representable at every H tested | §34 gate + `analytic_weights` | L0 | any experiment, as a gate | evidence that a *trained* router works — the vector is hand-designed |
| The task environment is unambiguous | every run, `oracle = 1.0000` | L0 | any experiment, printed first | evidence about any router's quality |
| Written state changes later decisions (C1) | `TACOSM-BASELINE-001` | L2 | a state-formation experiment | a claim beyond the three v0.1 families |
| A cheap router learns the relation from outcomes (C2) | `TACOSM-BASELINE-001` | L2 | `M2.1`, as the quality half of its trade | large-H behaviour — the result is at a small population |
| Routing degrades with history, not the task (C6) | `TACOSM-HS-001` | L2 | any large-H experiment | the refuted "linear-scorer capacity" mechanism |
| The top-K signal survives at `H ≤ 64` (C7) | `TACOSM-RETRIEVAL-001` F0 | L2 | `M2.1` | `H ≥ 128`, where it fails on a transfer-trained router |
| Matched-H training is the worst row | `TACOSM-MATCHED-001` | L3 | nothing — this closed a design option | a population schedule recommendation |
| The learning rule is the failure | `TACOSM-MATCHED-001` | L3 | the next learning-dynamics experiment | the identity of the fix |
| Selection is not the binding constraint | `TACOSM-LEARN-001` F2 | L2 | any exploration intervention | a verdict on the mechanism (C10) |
| **The large-H top-K signal is responsive to training** | `TACOSM-SURROGATE-001` F3 | L2 | **`M2.1` — this is its reason to exist** | `K = 1`, which did not move |
| Dense supervision improves top-K, not top-1 (C11) | `TACOSM-SURROGATE-001` F3 | L3 | `M2.1`, as the registered evidence for its re-queue | "the router learned relevance" |

The last two rows are `M2.1`'s inheritance, and they are the rows that changed
since this map was last written. Before F3, `M2.1`'s re-queue condition was
"the top-K signal is shown intact", and the only evidence for it was F0's
transfer-trained measurement, which is the measurement `M2.1` was designed not
to trust. After F3, the re-queue condition is met by a row that is
*responsive to training* — a different and stronger thing.

---

## What the map forbids

Three imports are tempting and wrong.

**CDL is not the runtime router.** Stage A evidence supports ranking
quality, not serving cost. The teacher scores every candidate with a full
LM pass — `O(N_candidates · C_LM)` — which is structurally incompatible with
the claim that executed computation scale with the relevant subset. Status
`NOT ESTABLISHED`. It enters as a teacher and a ceiling.

**The CASM L2 router is not a mechanism.** It is evidence. Phase 1.5A
reached 0.0012 training loss while holding out the intended relation across
all 18 trials. The successor does not inherit it; it inherits the *lesson*:
exact generated solutions are the primary metric, not optimization loss.

**The repair controller does not carry its own strongest claim.** The
REAL017 lineage is do-not-cite until audited. What transfers is the loop
shape — verify → localize → select → patch → re-verify — which is
architecture-neutral by construction.

And a fourth, added since: **F3's top-K result is not "the router learned
relevance."** The surrogate is gold-anchored exactly as the outcome reward is.
What transfers is the measured responsiveness of the top-K signal to the
training signal, and the bound — `K ≥ 2`, `K = 1` flat — travels with it.

---

## The gap the map exposes

The map now separates two different unresolved questions.

First, **verified persistence in the learned loop** remains open. The temporal
benchmark has validated an explicit write/read mechanism: a vector written at
t remains causally readable through k=32 under carry while reset destroys the
effect. What remains unmeasured is whether a learned write policy and
verification gate select useful experience, preserve it, and improve future
decisions better than always-write/no-write controls.

Second, **learned selective computation** remains open at the asymptotic level.
The product-key frontier demonstrates real capability/computation tradeoffs,
but fixed-factor construction still scores a growing fraction of persistent
state as M increases. Learned semantic proposal and end-to-end CASM-connected
selection therefore remain live blockers.

The unified-native-model benchmark adds a third research boundary: whether a
single predictive structural substrate can learn language, image, and audio
tasks without moving hidden truth across the pre-action information boundary.
That benchmark is preregistered separately and cannot inherit modality claims
from any individual prior experiment.
