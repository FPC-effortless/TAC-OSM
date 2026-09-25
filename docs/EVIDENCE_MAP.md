# Evidence Map — from prior research to TAC-OSM v0

**Binding companion to `provenance/COMPONENTS.md`.** Every interface in the
new architecture is traced to the evidence that licenses it, and to the
boundary of that evidence.

This document exists to prevent a specific failure mode: accidentally
treating a historical experiment as if it established more than it did.

---

## Evidence scale

`E0` idea → `E1` implemented → `E2` smoke → `E3` controlled → `E4`
reproduced → `E5` cross-condition → `E6` cross-domain.

Per `docs/pnds/RESULTS.md` in `cdl-attention-experiment`.

---

## Map

| Capability | Existing implementation | Evidence | Integration status |
|---|---|---|---|
| Persistent state | TAC `IdentityState` | E3 — carry/reset/shuffle probes, TAC-235/236 | **reusable** |
| State addressing | CDL / PNDS Stage 3b-r2 | E4 — 10/10 seeds, `p=0.001`, key-reset → 0.1250 | **reusable** |
| Relevance routing (teacher) | CDL `app.py` | E5 — Top-1 0.875 vs 0.633 gzip, 6/12/24 candidates | **teacher only** |
| Cheap routing | CDL Stage B `QKRouter` | E4 — distilled 73.75% vs direct 77.92%, `p≈0.275` | **unresolved** |
| Structural execution | CASM `phase1_dag` | E3 — Gates 0–6, copy-mask preflight | **reusable substrate** |
| Learned structural routing | CASM Phase 1.5A L2 | **failed 0/18** — loss 0.0012, wrong basin 0.93–0.97 | **failed** |
| Learned conditional routing | PNDS Stage 2c | E2 — 0.8853 vs static 0.0000, t=258 | **strongest router evidence** |
| Verification | TAC `VerifierGuidedRepairController` | E3 — TAC-267→274 | **provisional** |
| Repair | TAC `ProceduralMemoryStore` | E3 — bounded sandbox repair | **candidate** |
| Persistent write | — | **not implemented** | **implement** |
| Temporal state transition | — | Stage 4W pre-registered only | **implement** |
| Verified state commit | — | **not implemented** | **implement** |
| Path-level verification | — | **not run** | **implement** |
| Full `S→R→C→A→O→V→S` loop | — | — | **new** |

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

---

## The gap the map exposes

The map is mostly filled for the loop's first five terms. The last two are
empty, and they are the ones the thesis actually needs:

> **`V_t` → `S_{t+1}`: whether verification gates a state update.**

No repository has run path-level verification, verified-only commit, or a
persistent write across a temporal boundary. Stage 4W is pre-registered but
unexecuted. This is where the program's claim actually lives, and it is
where the integrated model earns its keep: those transitions become testable
in situ rather than in isolation.
