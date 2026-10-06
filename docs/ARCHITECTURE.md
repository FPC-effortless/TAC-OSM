# TAC-OSM v0 — Architecture

**Status:** design record for the integrated prototype.
**Provenance:** `provenance/COMPONENTS.md` is the binding source for what each
mechanism may claim. This document does not upgrade any claim.

---

## 1. Purpose

A single executable artifact for the unified hypothesis:

> **Persistent Structural Computation.** Useful computation should be
> reusable across time through persistent state, selectively addressable
> through learned relevance routing, executable through structural
> pathways, and subject to verification and repair.

The three source laboratories each own part of this loop. None can execute
the whole. TAC-OSM exists to ask a question the parts cannot answer:

> Can an integrated system produce the claimed capability, and **which
> mechanisms are causally responsible for it?**

That question is answered by ablation, not by component proof.

---

## 2. Design constraint: interfaces before implementations

The integration boundary is a set of five interfaces. Nothing crosses that
boundary without an adapter, and every adapter carries the exclusions
recorded in `provenance/COMPONENTS.md`.

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

This has one purpose: **any mechanism can be replaced without rewriting the
model.** That is exactly what ablation requires.

---

## 3. The canonical loop

```
S_t → R_t → C_t → A_t → O_t → V_t → S_{t+1}
```

| Symbol | Interface | Current status |
|---|---|---|
| `S_t` | `PersistentState` | imported — read-only evidence |
| `R_t` | `RelevanceRouter` | imported — conditional router, `E2` |
| `C_t` | `StructuralExecutor` | imported — substrate, learned routing failed |
| `A_t` | `StructuralExecutor.execute` | imported |
| `O_t` | `Environment` | new — written here |
| `V_t` | `Verifier` | imported — loop shape only, claims downgraded |
| `S_{t+1}` | `PersistentState.write` | implemented in the hardened runtime; verified commit semantics are bounded synthetic infrastructure |

---

## 4. The interfaces

### 4.1 `PersistentState`

```python
class PersistentState(Protocol):
    def read(self, query: Query) -> StateRead: ...
    def write(self, update: StateUpdate) -> StateWrite: ...
```

Adapted from `tac_sie/types.py::IdentityState` — the deliberately minimal
`memory_keys` / `memory_values` / `slot_used` shape — rather than the
~30-field `tac_transformer/model.py` version. Rationale: the minimal shape
forces role separation, and an ablation surface with 60 config knobs is
uninterpretable.

`write` is TAC-OSM runtime machinery rather than imported source behavior. The temporal experiment establishes only bounded explicit state persistence; learned state formation and semantic write policy remain research targets.

### 4.2 `RelevanceRouter`

```python
class RelevanceRouter(Protocol):
    def route(self, query: Query, state: PersistentState,
              candidates: Sequence[Candidate]) -> RoutingDecision: ...
```

Two implementations, deliberately both present:

- **`CDLTeacher`** — the `O(N · C_LM)` relevance oracle. Used to *generate
  supervision* and as a quality ceiling. Never used as the runtime router:
  its cost is structurally incompatible with the thesis.
- **`LearnedConditionalRouter`** — adapted from
  `stage2c_relational.py::RelationalBanditRouter`, the corrected key-gated
  form at `f989430`. This is the strongest routing evidence in the
  portfolio (`E2`, 25/25 per-seed deltas positive).

### 4.3 `StructuralExecutor`

```python
class StructuralExecutor(Protocol):
    def execute(self, structure: Structure, inputs: Inputs) -> ExecutionResult: ...
```

Adapted from `casm_v01/phase1_dag/model.py`. The executor contract is
preserved verbatim, including its sharpest constraint:

> **Routing depends only on structural state; runtime values never enter
> the router.**

This boundary is what makes structural execution an *addressable* resource
rather than a correlation. It also survives the failure analysis: the L2
router failed at learned routing, not at execution.

### 4.4 `Verifier` / `RepairController`

```python
class Verifier(Protocol):
    def verify(self, computation: Computation, outcome: Outcome) -> VerificationResult: ...

class RepairController(Protocol):
    def repair(self, computation: Computation, verification: VerificationResult) -> RepairResult: ...
```

`VerifierGuidedRepairController` from `tac_transformer/repair_controller.py`
is already architecture-neutral: `run()` takes injected `verifier` and
`repair` callables and does not depend on the TAC LM. **No adapter needed.**

Its exclusions travel with it: the REAL017 lineage that used it is
**do-not-cite until audited** (verifier received corruption labels, repair
received gold slots). TAC-OSM imports the *loop shape* and nothing from
that claim.

---

## 5. What is learned, what is not

| Component | Status | Rationale |
|---|---|---|
| persistent state representation | direct import | `E3`, well-established |
| state addressing | direct import | `E3` |
| structural execution substrate | direct import | `E3` substrate; learned routing failed |
| verify/repair loop shape | direct import | `E3` shape, downgraded claims |
| **cheap CDL-derived routing** | experimental | Stage B `INCONCLUSIVE`, `p ≈ 0.275` |
| **learned structural routing** | experimental | L2 0/18; `O`-masked successor |
| **persistent write** | implemented/runtime-tested | verified commit path exists; learned semantic write policy remains unresolved |
| **temporal state transition** | bounded measured mechanism | TACOSM-TEMPORAL-001 supports explicit state reuse through enforced boundaries; not semantic memory |
| **verified state commit** | implemented infrastructure | verifier-gated commit path is present; capability contribution not independently established |

The model is not pretending everything is validated. It is a machine with
three working imported mechanisms, three explicitly unresolved ones, and
two that are open research targets.

---

## 6. Why the failure modes transfer

The two most valuable failures in the portfolio both become design
constraints here.

### Low loss ≠ competence

CASM Phase 1.5A reached ~0.0012 training loss while holding out the intended
structural relation (0/18 successes, wrong basin 0.93–0.97). The same
divergence appeared in the CASM v0.1 LM, where teacher-forced loss fell to
0.22–0.24 while exact generated solutions sat at 16%/6%.

**Consequence:** TAC-OSM's primary metrics are **exact generated solutions
and class-balanced task metrics**. LM loss and router accuracy are
diagnostics, never primary endpoints.

### Oracle success ≠ representability

At `dd8f63c` the learned feature map never read `state.key`; the intended
relation was identically unrepresentable, scoring exactly 0 for every
candidate, while the `true_key` *arm* reported 1.0000 for four consecutive
commits.

**Consequence:** `src/tac_osm/representability.py` is a compulsory
pre-training unit test on every learned component. It is not a retrospective
audit. An oracle arm never passes through the learned map, so oracle
success implies nothing about the basis.

---

## 7. Ablation, not component proof

The methodology change is structural. Prior work asked:

> Can each primitive exist?

TAC-OSM asks:

> Does this mechanism contribute to useful behavior in the integrated
> system?

The primary scientific instrument is therefore the ablation harness, where
one architecture with controlled switches produces every variant:

```
state:      enabled | disabled | shuffled | corrupted | wrong-key
router:     learned | static   | random   | oracle    | full-context
structure:  learned | random   | oracle   | flattened
verifier:   none    | final    | path     | path+repair
write:      enabled | disabled
```

Every row of `docs/ABLATION_PLAN.md` is produced by one code path with
switches flipped, never by a separate experiment codebase. That is what
makes the comparisons interpretable.

**Component existence is addressed by interface validity,
representability, and benchmark-validity gates** — cheap, analytical, run
before training. Everything else is answered against the integrated model.

---

## 8. Scope discipline

Building the model licenses no new claims. The honest accounting:

| Level | Statement |
|---|---|
| Architecture | Integrated prototype: demonstrated |
| Persistence | supported |
| Relevance routing | bounded supported |
| Structural execution | bounded / partial |
| Verification | provisional in the integrated setting |
| Persistent update | implemented for bounded synthetic state path; learned update semantics under test |
| Efficient routing | unresolved |
| **System-level claim** | **not yet established** |

`C_executed ≈ f(|R|)` rather than `f(|H|)` remains a hypothesis until
measured under matched conditions, with a pre-declared capability parity
margin and frozen history levels.
