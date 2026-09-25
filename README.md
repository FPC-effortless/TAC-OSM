# TAC-OSM v0

The integrated prototype for **persistent structural computation**.

This is the integration repository for three research lines that each own one
stage of a single causal loop. It is deliberately **not** another research
substrate repository: it does not reconstruct the source work, it adapts it
behind a small interface boundary and then asks a question the parts cannot
answer alone.

```
S_t → R_t → C_t → A_t → O_t → V_t → S_{t+1}
```

> **Persistent Structural Computation:** useful computation should be
> reusable across time through persistent state, selectively addressable
> through learned relevance routing, executable through structural pathways,
> and subject to verification and repair.

---

## Why this repository exists

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

## Repository layout

```
src/tac_osm/__init__.py         the five interfaces + the Environment protocol
src/tac_osm/representability.py PNDS-URP v0.4 §34, as a compulsory pre-training test
src/tac_osm/leakage.py          the anti-leakage boundary, checked structurally
src/tac_osm/ablation.py         one architecture, controlled switches
tests/test_tac_osm.py           the three pre-model gates
docs/ARCHITECTURE.md            interfaces before implementations
docs/ABLATION_PLAN.md           the primary scientific instrument
docs/EVIDENCE_MAP.md            prior research -> interface, and the boundary
provenance/COMPONENTS.md        binding ledger: reuse vs do_not_claim
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
