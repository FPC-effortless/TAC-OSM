# TAC-OSM v0.1

> **A controlled experimental environment for studying persistent-state-
> conditioned relevance routing and selective computation.**

That is the honest scope, and it is narrower than the architecture's ambition.
v0.1 does not demonstrate a capability. It demonstrates a *instrument*: a loop
in which a specific mechanism can be switched off, measured, and attributed —
and in which a measurement that refers to nothing is caught before it is
reported.

## The central claim under test

```
C_total  =  C_address  +  C(|R|)  +  C(|A|)  +  C_verify      intended
C_total  =  C_address  +  C(|H|)  +  C(|A|)  +  C_verify      conventional
```

where `|R|` is the relevant subset of history and `|H|` is all of it.

**This is not the claim that computation is small. It is the claim that
computation depends on the relevant subset rather than on everything that
happened.** It is the reason the architecture exists, it is what separates it
from running the whole context through a model, and it is the claim every
downstream piece of work exists to test.

**Its status is `UNTESTED`**, and the honest v0.1 cost model is:

```
C_total(H) = O(H) routing + O(10) execution + O(verification)
```

The executor is capped at `active_count = max_nodes = 10`, so `C_executed`
reads 10 at every history size — a flat column that means the architecture has
**no room to scale**, not that scaling has been demonstrated. `C_router` is the
only cost term that genuinely varies, and it grows linearly with `H`. The
desired decomposition is *not demonstrated*, and no figure in this repository
may present `C_executed` as though it had been measured against `|R|` (claim
**C5**). See `docs/MEASUREMENT_LAYERS.md` for the full accounting.

The claim becomes testable when a retrieval boundary exists in the loop, which
is what makes `|R|` a measurable quantity at all.

---

## What this repository exists for

The loop the architecture implements:

```
S_t → R_t → C_t → A_t → O_t → V_t → S_{t+1}
```

state is read → the relevant subset is routed to → structure is executed → an
action is taken → the outcome is verified → state is written. The central
claim above is a statement about the *cost* of the second and third arrows:
they should scale with what `R_t` selected, not with everything in `S_t`.

The three source laboratories established the primitives. They have not yet
been asked the question that matters.

| Repo | Owns | Status |
|---|---|---|
| `TAC-transformer` | `S_t` — persistence, `V_t`/repair loop shape | E3, reusable |
| `cdl-attention-experiment` | `R_t` — relevance routing, `C_t` — structural execution | E2/E3 |
| `TAC-Prime` | architecture exploration | separate TAC-PSM line |

The prior methodology asked *can each primitive exist* and answered it
through sequential gated experiments. That produced excellent controls. It
also left the actual question open:

> Can these primitives form a useful integrated computational system, and
> **which mechanisms are causally responsible**?

That is a question about the whole, and it is answered by **ablation against
an integrated model**, not by further component proof.

---

## The integration boundary

Five protocols, in `src/tac_osm/__init__.py`. Nothing crosses this boundary
without an adapter, and every adapter carries the exclusions recorded in
`provenance/COMPONENTS.md`.

```
                    TAC-OSM
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        State        Router      Executor
          │            │            │
          ▼            ▼            ▼
         TAC           CDL          CASM
```

The interfaces are narrower than their sources on purpose. An ablation
surface is only interpretable if each mechanism can be replaced without
rewriting the model.

---

## What is imported, and what that import may claim

`provenance/COMPONENTS.md` is the binding ledger. Three constraints from it
shape the design:

**CDL is a teacher, never the runtime router.** It scores every candidate
with a full LM pass — `O(N · C_LM)` — which is structurally incompatible with
the thesis that executed computation scale with the relevant subset rather
than total history. Status `NOT ESTABLISHED`.

**The CASM L2 router is evidence, not mechanism.** Phase 1.5A reached 0.0012
training loss while holding out the intended relation across all 18 trials
(wrong basin 0.93–0.97). Its successor inherits the lesson, not the code.

**The repair controller carries its loop shape, not its claim.** The REAL017
lineage is do-not-cite until audited: the verifier received corruption
labels and the repair path received gold slots.

---

## The three pre-model gates

These run before any model is trained, because a failure here invalidates
every learned-arm measurement that follows. `tests/test_tac_osm.py` enforces
all three.

**G1 — interface validity.** The router provably cannot see target, answer,
gold structure id, oracle mask, or correct action.

**G2 — representability.** The intended relation is expressible *through the
actual learned feature map*, not through an oracle arm. This is PNDS-URP
v0.4 §34, added because a key-blind feature map at `dd8f63c` scored
analytically ideal weights at exactly 0 for every candidate while the
`true_key` arm reported 1.0000 for four consecutive commits.

```
oracle success  =/=>  learned-map representability
```

**G3 — benchmark validity.** Gold is the unique satisfier, the relation is
decidable from router-visible inputs, the fixed baseline provably cannot
express it, and exact generated solutions — not teacher-forced loss — are
the primary metric.

---

## The ablation harness

One architecture, controlled switches. Every variant is one config with
fields flipped; there is deliberately no per-experiment code path.

```yaml
state:      {enabled: true, write: true, intervention: persistent|shuffled|reset|corrupted|random|wrong_key}
router:     {type: learned|static|random|oracle|full_context}
structure:  {type: learned|random|oracle|flattened}
verifier:   {type: none|final|path, repair: true|false}
```

The plan is in `docs/ABLATION_PLAN.md`. The primary metric is not router
accuracy or LM loss — those are diagnostics. It is the joint
capability/computation curve:

```
C_executed  ≈  f(|R|)          intended
C_executed  ≈  f(|H|)          conventional baseline
```

where `R` is relevant structure and `H` is accumulated history. History
levels are frozen and the capability parity margin declared **before** the
confirmatory run.

---

## The measurement contract

Every number this repository emits belongs to one of three ordered layers.
The ordering is strict and it is the reason the gates exist.

| | Layer 1 — model validity | Layer 2 — mechanism | Layer 3 — system |
|---|---|---|---|
| asks | is the measurement about the model at all? | does the mechanism behave as claimed? | does the system solve the task, at what cost? |
| on failure | **voids** every downstream number | withdraws one mechanism claim | withdraws the system claim |

**A Layer 1 failure is invisible from the layers above.** A zeroed parameter
vector produces a smooth, bounded, sensible-looking softmax. An ambiguous task
produces a clean, reproducible accuracy. Neither is detectable in the number it
emits — only in the preconditions, which is why Layer 1 is a *gate* and not a
metric.

That is not hypothetical. Both of this repository's expensive errors were
Layer 1 failures published as Layer 2 findings:

- **TACOSM-HS-001** reported `routing@1 ≈ 1/40` at H=32 as a routing result.
  The router had no weights loaded: `w = [0]*n` scores every candidate
  identically, the softmax is uniform, and `1/H` reads as a smooth degradation
  curve. With weights loaded the number is 0.4250. Now caught by
  `src/tac_osm/integrity.py`, which raises before any measurement runs.
- **`dd8f63c`** shipped a learned feature map that never read `state.key`, so
  the intended relation scored exactly 0 for every candidate while the oracle
  *arm* reported 1.0000 for four consecutive commits. Oracle success implies
  nothing about representability. Now a compulsory pre-training failure
  (`representability.py`, §34).

Both were caught *after* the interpretation was written. The layers exist so
the check runs first. Full contract: `docs/MEASUREMENT_LAYERS.md`.

---

## What has been measured

Results live in `docs/` with a pre-registered decision rule, and every claim
in them is mirrored in `docs/CLAIMS.md` with a status and a blocker. A claim
cannot drift from `measured` to `supported` without an entry changing there.

| Experiment | Question | Result |
|---|---|---|
| `TACOSM-BASELINE-001` | does a cheap linear router learn the relevance relation from outcomes alone? | **yes** — 0.4396 vs `random` 0.1296, `static` 0.0376 |
| `TACOSM-HS-001` | does accuracy fall with history because routing degrades? | **yes** — oracle 1.0000 at every H, so the environment is not the cause |
| `TACOSM-RETRIEVAL-001` (F0) | does the top-K signal survive top-1 collapse? | **at `H ≤ 64`, yes**; **at `H ≥ 128`, no** (rec@16 → 0.550 at H=256) |
| `TACOSM-MATCHED-001` | is the large-H loss a representation problem or a training problem? | **neither** — the learning rule is the failure |

`TACOSM-MATCHED-001` is the result that redirected the roadmap. Its
pre-registered rule committed to two outcomes and produced neither: matched-H
training is the *worst* row at every population, while the analytic vector —
one shared untrained weight vector — reaches `routing@1 = 1.0000` at
H ∈ {8, 64, 256, 512}. The hypothesis class is adequate and H-invariant. What
fails is REINFORCE's update: 5 successes in 500 steps at H=256, and 5000 steps
leave `routing@1` at 0.0700 while the weight norm grows 12-fold and the margin
*worsens*.

> **The experiment was designed to choose an architecture, and it answered
> that the architecture is not the problem.** Stage F1 (the retrieval index)
is therefore **suspended, not abandoned** — an index measured against a
baseline whose weakness is a training artefact is unsound in both directions.
The learning dynamics are next (Stage F2).

---

## Repository layout

```
src/tac_osm/__init__.py         the five interfaces + the Environment protocol
src/tac_osm/representability.py PNDS-URP v0.4 §34, as a compulsory pre-training test
src/tac_osm/leakage.py          the anti-leakage boundary, checked structurally
src/tac_osm/integrity.py        model-state gate: catches untrained weights before a run
src/tac_osm/ablation.py         one architecture, controlled switches
tests/                          the three pre-model gates + the integrity gate
docs/ARCHITECTURE.md            interfaces before implementations
docs/MEASUREMENT_LAYERS.md      the three-layer measurement contract
docs/ABLATION_PLAN.md           the primary scientific instrument
docs/CLAIMS.md                  every claim, its status, and its blocker
docs/ROADMAP.md                 stage order, and why F1 is behind F2
docs/EVIDENCE_MAP.md            prior research -> interface, and the boundary
provenance/COMPONENTS.md        binding ledger: reuse vs do_not_claim
scripts/                        the exact commands that produced every number
```

---

## What this repository does not claim

Building an integrated model licenses no new capability claims.

| Level | Statement |
|---|---|
| Architecture | Integrated prototype: demonstrated |
| Persistence | supported |
| Relevance routing | bounded supported |
| Structural execution | bounded / partial |
| Verification | provisional in the integrated setting |
| Persistent update | under test |
| Efficient routing | unresolved |
| **System-level claim** | **not yet established** |

The source evidence remains bounded by its own documents. Nothing here
upgrades a tier.

---

## Running the gates

```bash
python -m pytest tests/ -q
```

No dependencies beyond the standard library and `pytest`, by design.

## Provenance

Every result must identify
`repository → branch → commit → benchmark version → experiment id →
configuration → seed → artifact → metric`, per `PNDS_RESEARCH_PROTOCOL.md`.
`src/tac_osm/ablation.py::freeze` attaches that record before a run.

## License

Apache-2.0.
