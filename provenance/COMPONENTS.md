# Component Provenance Ledger

Every mechanism imported into TAC-OSM is recorded here with its source of
truth, the evidence that supports it, and — critically — what that evidence
does **not** license.

Protocol reference: `docs/pnds/PNDS_RESEARCH_PROTOCOL.md` (PNDS-URP v0.4)
in both `TAC-transformer` and `cdl-attention-experiment`. Evidence scale
E0 (idea) → E6 (cross-domain), per `docs/pnds/RESULTS.md`.

## Legend

`reuse` — what the TAC-OSM interface may legitimately import.
`do_not_claim` — what the source evidence does not establish. This is the
binding part of the ledger: an import carries these exclusions forward.

---

## Persistent state — from TAC-transformer

| Field | Value |
|---|---|
| source_repo | `FPC-effortless/TAC-transformer` |
| branch / commit | `main` @ `6cce2ce0fc027ce81fef7321d1b8e97c2fd3b66f` |
| modules | `tac_transformer/model.py`, `tac_transformer/core/__init__.py` |
| symbol | `IdentityState` (dataclass) |
| last touched | `tac_sie/types.py` @ `face2d9` (the minimal `SIE` view) |

`IdentityState` is a batched dataclass of tensors: `stability`,
`program_memory`, plus optional tiers (`stable_program_memory`,
`archival_program_memory`, `program_age`, `program_write_frequency`),
engram and content-addressed fields, and `content_*` cue/value/mask triples.

There are **two** `IdentityState` definitions in the source tree:

- `tac_transformer/model.py` — the mature, research-featured version.
- `tac_sie/types.py` — a deliberately minimal `memory_keys` /
  `memory_values` / `slot_used` view for the engine-decomposition lane.

TAC-OSM uses the minimal `tac_sie` shape as its **interface** and adapts
either implementation behind it, because the minimal shape is the one that
actually forces role separation (`docs/structure_centric_intelligence_research_program.md`).

| Field | Value |
|---|---|
| evidence | TAC core carry/reset/shuffle probes; TAC-235/236 |
| level | E3 (controlled) |
| status | supported, bounded and synthetic |

**reuse**

- the persistent-state container and its batched tensor shape
- carry / reset / shuffle / corrupt intervention vocabulary
- the "state is causally load-bearing, not correlational" measurement style

**do_not_claim**

- long-horizon memory — the source evidence is bounded
- persistent **write** across a temporal boundary — no result exists yet;
  `S_{t+1}` is unestablished (Stage 4W is pre-registered, not run)
- state composition, continual learning, unseen state structures
- that persistence yields any wall-clock or context-scaling advantage

---

## Repair controller — from TAC-transformer

| Field | Value |
|---|---|
| source_repo | `FPC-effortless/TAC-transformer` |
| branch / commit | `main` @ `6cce2ce0fc027ce81fef7321d1b8e97c2fd3b66f` |
| modules | `tac_transformer/repair_controller.py` @ `b079ca5`, `tac_transformer/procedural_memory.py` @ `b079ca5` |
| symbols | `VerificationResult`, `RepairAttempt`, `RepairControllerDecision`, `RepairControllerResult`, `VerifierGuidedRepairController`, `ProceduralMemoryStore` |

Importantly, `VerifierGuidedRepairController` is **already** an
architecture-neutral loop:

```python
def run(
    self,
    *,
    task_key: str,
    initial_output: str,
    verifier: Callable[[str], VerificationResult],
    repair: Callable[[str, str], str],
) -> RepairControllerResult:
```

It takes injected `verifier` and `repair` callables and does not depend on
the TAC LM. This is the cleanest single import in the portfolio and needs no
adapter.

| Field | Value |
|---|---|
| evidence | TAC-267 through TAC-274 repair-control benchmarks |
| level | E3 (controlled, synthetic, sandboxed) |
| status | supported, bounded |

**reuse**

- the verify → localize → select-procedure → patch → re-verify loop shape
- `ProceduralMemoryStore` for repair-procedure retention and reuse
- bounded multi-attempt control, `max_attempts` and retry-termination

**do_not_claim**

- live-repository repair — evidence uses copied sandbox files
- that the loop is causally load-bearing in the integrated setting
- **strong claims**: the REAL017 lineage that used this controller was
  downgraded. `feature/tac-scm-real003` is **do-not-cite until audited**
  (`docs/REAL017_AUDIT.md`) because the verifier received corruption labels
  and the repair path received gold slots. TAC-OSM may not inherit that
  claim; only the loop shape.
- unrestricted multi-bug repair chains — TAC-273 exposed interacting
  repair-chain completion as a hard frontier; TAC-274 improved bounded
  planning but did not solve it

---

## CDL relevance teacher — from cdl-attention-experiment

| Field | Value |
|---|---|
| source_repo | `FPC-effortless/cdl-attention-experiment` |
| branch / commit | `main` @ `c31554413301e3c9d3e6b3f8c8c6be572a74a748` |
| module | `app.py` (standalone module, not a package) |
| symbols | `Case`, `make_case`, `make_benchmark`, `continuation_nll_batch`, `rank_metrics`, `selected_answer_nll`, `gzip_scores` |

**Interface note — the most consequential finding in this audit.**

`app.py` is a Gradio application, not an importable library. The CDL
computation lives at `continuation_nll_batch`, which

1. is module-level stateful (it caches the SmolLM2 model and tokenizer in
   globals),
2. depends on `transformers==5.15.1`, `gradio`, `pandas`, `safetensors`,
3. requires a full LM forward pass **per candidate**.

The repository has **no `pyproject.toml` or `setup.py`** — the repo is not
installable as a dependency. Any integration therefore requires either a
vendored extraction or a re-declaration of the protocol.

| Field | Value |
|---|---|
| evidence | Stage A paired (120 cases × 6/12/24 candidates); joint-attention control |
| runs | Actions `33231291796`, `33231710117`; artifacts `9708661555`, `9708717039` |
| model | `HuggingFaceTB/SmolLM2-135M` |
| level | E5 (cross-condition: candidate scaling) |
| status | supported, bounded to synthetic relational language |

**reuse**

- the conditional-description-length *definition*, `CDL(M,Q) = -NLL(Q | M)`
- the hard-distractor benchmark design (same-entity-wrong-relation,
  same-relation-wrong-entity, correct-value-in-wrong-fact, random plausible)
- `rank_metrics` / `selected_answer_nll` measurement definitions
- CDL as a **teacher signal** only

**do_not_claim**

- **sublinear routing cost** — the teacher scores every candidate with a full
  LM pass: `O(N_candidates · C_LM)`. This is *structurally incompatible* with
  the PNDS thesis that executed computation scale with the relevant subset
  rather than total history. Status `NOT ESTABLISHED`; this is the program's
  central unresolved efficiency gap.
- that CDL replaces attention — the joint-attention control measured raw
  attention mass at Top-1 0.2667 / MRR 0.5422 vs CDL 0.8750 / 0.9361, but
  the correct inference is that **attention weight mass is a weak explicit
  relevance score**, not that CDL substitutes for the attention mechanism
- natural-language generality — 15/120 failures cluster on same-entity
  wrong-relation distractors (especially `head of government`)

---

## CDL → cheap router distillation — from cdl-attention-experiment

| Field | Value |
|---|---|
| source_repo | `FPC-effortless/cdl-attention-experiment` |
| branch / commit | `main` @ `c31554413301e3c9d3e6b3f8c8c6be572a74a748` |
| module | `run_stage_b.py` @ `f43dfcf` |
| symbols | `QKRouter` (mean-pooled bilinear Q/K student), `train` |

| Field | Value |
|---|---|
| evidence | Stage B, Actions run `33231945184`, artifact `9708841751` |
| level | E4 (reproduced) |
| status | **INCONCLUSIVE** — the critical transition |

Result: direct-label Q/K Top-1 **77.92%** vs CDL-distilled **73.75%**
(paired McNemar on 68 discordant pairs, `p ≈ 0.275`). Distillation
transferred teacher *geometry* (teacher-agreement 72.92% vs 67.08% for
direct) but produced **worse** task performance. The implemented target
normalizes teacher scores within each candidate set and applies a unit
softmax, discarding absolute scale.

**reuse**

- `QKRouter` as the cheap-router architecture
- the distillation **failure as a constraint**: any integrated model that
  uses a learned router must be evaluated on downstream execution and cost,
  not teacher-agreement. The Stage B result is the standing proof that
  teacher imitation is not capability.

**do_not_claim**

- that CDL supervision has been distilled into a useful cheap router — it
  has not. Status `INCONCLUSIVE`.
- that the two students differ significantly — they do not (`p ≈ 0.275`).
  The defensible statement is "no demonstrated benefit from this
  formulation," not "CDL supervision is worse."

---

## CASM structural executor — from cdl-attention-experiment

| Field | Value |
|---|---|
| source_repo | `FPC-effortless/cdl-attention-experiment` |
| branch / commit | `main` @ `c31554413301e3c9d3e6b3f8c8c6be572a74a748` |
| module | `casm_v01/phase1_dag/model.py` |
| symbols | `_Base`, `CASMS`, `StaticMask`, `structural_encode`, `gate`, `forward` |
| grammar | `casm_v01/phase1_dag/grammar.py` (`Op`, `ARITY`, `Node`, `Edge`) |

The executor contract is frozen in `casm_v01/phase1_dag/README.md`:

- single-pass topological execution, no recurrent spectral diagnostic
- fixed upper-triangular candidate substrate `A` shared across episodes
- hard existence mask for variable-size programs
- true wiring is one of many admissible wirings, so `m_i m_j A_ij` is not the
  oracle
- **routing depends only on structural state; runtime values never enter the
  router**
- `alpha_eta.shape == (2)`: one learned computation strength per syntactic
  argument port
- raw operand order retained; commutative canonicalization is bookkeeping
- copy-mask is a mandatory **oracle preflight**, never a learned baseline
- first learned comparison is Static Mask vs CASM-S

Executor: `alpha = c * softplus(eta)`; `NOT = 1 - alpha[0]*x`;
`AND = a*b`; `OR = a+b-a*b`; `XOR = a+b-2ab`.

| Field | Value |
|---|---|
| evidence | CASM Phase-1 Gates 0–6 harness; CASM v0.1 paired ablations |
| level | E3 |
| status | **PARTIALLY_SUPPORTED** — the substrate is reusable, learned routing is not |

**reuse**

- the executor semantics, the `Op`/`Node`/`Edge` grammar, and the
  structural encoder
- the seven-gate diagnostic set and the copy-mask falsification preflight
- the frozen routing-vs-execution information boundary as an architectural
  constraint
- `environment_success` / gold-index isolation from the PNDS harness (below)

**do_not_claim**

- that learned structural routing works. Two separate failures:

  1. **Phase 1.5A L2 factorized router: 0/18.** Pre-registered
     reconstruction; `d ∈ {1,2,3} × 6 seeds`. Training loss reached ~0.0012
     while exact held-out accuracy was 0.7857 and the *wrong* gate basin
     was 0.93–0.97. Monotonically decreasing loss, consistent wrong basin.
     **Low optimization loss is not evidence of executable routing.**
  2. **CASM v0.1 multi-seed promotion withdrawn.** A post-run audit found
     the `graph_reachability` generator always emitted `answer yes`, so the
     reported −0.04311-nat advantage was essentially all of that defective
     task. Excluding it, the five remaining tasks are **+0.00118 nats** — a
     tie or slight regression.
- that the CASM v0.1 model learned anything real about graph reachability:
  the capacity-floor diagnostic showed 0/50 on `yes` (total class collapse
  to `no`), and state tracking at 16%/6% exact solves.
- that any of this is evidence about language modelling generally.

---

## Learned relational router — from cdl-attention-experiment (`pnds` branch)

| Field | Value |
|---|---|
| source_repo | `FPC-effortless/cdl-attention-experiment` |
| branch / commit | `pnds` @ `bd2baea4499914672bbfead2cb1378c1cfaa5539` |
| module | `casm_v01/pnds_gate_001/stage2c_relational.py` @ `42f9814` |
| symbols | `Candidate`, `Episode`, `make_episode`, `environment_success`, `static_select`, `random_select`, `RelationalBanditRouter` |

This is the **strongest routing evidence in the portfolio** and the one
TAC-OSM should build its learned router on. In the corrected Stage 2c
environment, relevance is conditional — the context marks positions; the
gold candidate matches the query on marked positions and deliberately
anti-matches on unmarked ones, so total query-agreement is actively
anti-correlated with relevance.

| Candidates | learned | static | random | Δ(learned−static) |
|---:|---:|---:|---:|---:|
| 8 | 0.8853 | 0.0000 | 0.1167 | +0.8853 ± 0.0077 (t = 258) |
| 16 | 0.7687 | 0.0000 | 0.0900 | +0.7687 ± 0.0126 (t = 136) |
| 32 | 0.5027 | 0.0000 | 0.0267 | +0.5027 ± 0.0401 (t = 28) |

25/25 per-seed deltas positive, 5 training seeds, 300 held-out episodes,
no gold supervision. The learned weight signature is the intended rule:
gated-agreement weights positive (1.29–1.47), total-agreement weights
**negative** (−1.25 to −1.72), bias +4.00.

| Field | Value |
|---|---|
| evidence | PNDS-GATE-001 Stage 2c; CI run `36043770779` (green) |
| level | E2 (controlled, multi-seed, held-out, synthetic, non-causal) |
| status | **SUPPORTED** |

**reuse**

- `RelationalBanditRouter`'s **key-gated feature map** — the corrected form
  after `f989430`. This is the one learned router that provably represents
  its target relation.
- `environment_success` / `gold_index` isolation: gold is evaluator-only
- the four-arm protocol (learned / static / random / oracle) on identical
  episodes
- the unit-tested invariants: `test_gold_is_the_unique_satisfier`,
  `test_relation_is_observable_from_router_inputs`,
  `test_static_rule_cannot_express_the_relation`

**do_not_claim**

- causality — Stage 2c is a bandit, not a decision loop; no causal
  intervention component exists. **Do not promote to E3.**
- persistence, recurrence, or verification — none present
- that the learned arm reaches the oracle ceiling (0.8853 vs 1.0 at 8
  candidates; the shortfall is the honest measure of what outcome training
  leaves on the table)
- any generality beyond `dim=8` bit descriptors

---

## The representability gate — protocol §34

| Field | Value |
|---|---|
| source | `docs/pnds/PNDS_RESEARCH_PROTOCOL.md` (PNDS-URP v0.4), §34 |
| provenance | added after `dd8f63c` was invalidated by `f989430` |

This is the single most important methodological asset in the portfolio and
it transfers wholesale:

> No learned-arm measurement may be interpreted until the mechanism under
> test has been shown to *represent* the relation it is being asked to
> learn.

The failure it prevents: at `dd8f63c`, `PersistentStateRouter._features`
never read `state.key`, so all 70 key blocks were computed for every
candidate and the analytically ideal weights scored **exactly 0** for every
candidate — a constant, in its own null space. The relation was
identically unrepresentable. Meanwhile `true_key`, an **arm**, read the
relation directly and reported 1.0000 for four consecutive commits.

```
oracle success  ≠>  learned-map representability
```

TAC-OSM adopts this as a **compulsory pre-training unit test** on every
learned component, not a retrospective audit. See
`src/tac_osm/representability.py`.

---

## Excluded — deliberately not imported

| Source | Why excluded |
|---|---|
| `tac_transformer/model.py::TACTransformerLM` | The full LM is ~4000 lines and carries 60+ `TACConfig` knobs with interdependent semantics. Importing it would make TAC-OSM's ablation surface uninterpretable. TAC-OSM adapts `IdentityState` only. |
| `tac_transformer/structure_*` (8 modules) | REAL004/005/006 evidence is synthetic and internal; the structure-slot / bridge / family-router surface is not the cleanest carrier of the loop. |
| `tac_osm/*` on `research/tac-osm-v05-*` | The v0.5 anti-lookup interpretation is **not established** — the split exercises unseen physical effect rows (`effect[c ^ a]`), so it tests extrapolation, not compositional generalization. Pre-existing CI failures on those branches. |
| `casm_v01/phase15a/*` | Phase 1.5A L2 reconstruction is a documented negative result (0/18). Preserved as evidence, not imported as mechanism. |
| CASM v0.1 byte-level LM (`model/`) | Multi-seed promotion withdrawn after the defective-task audit. |
| `TAC-Prime` (`tacm/`) | A separate TAC-PSM experiment line with its own progression. Not part of this integration. |

---

## Summary matrix

| Capability | Source | Commit | Evidence | Status |
|---|---|---|---|---|
| persistent state (read) | TAC-transformer | `6cce2ce` | E3 | reusable |
| verify/repair loop shape | TAC-transformer | `b079ca5` | E3 | reusable (shape only) |
| CDL relevance teacher | cdl-attention-experiment | `c315544` | E5 | teacher only |
| cheap CDL-derived router | cdl-attention-experiment | `f43dfcf` | E4 | **unresolved** |
| structural execution substrate | cdl-attention-experiment | `c315544` | E3 | reusable |
| learned structural routing | cdl-attention-experiment | `c315544` | — | **failed** (0/18) |
| learned conditional routing | cdl-attention-experiment | `42f9814` | E2 | **strongest router evidence** |
| persistent write across boundary | — | — | — | **not implemented** |
| `S_{t+1}` verified commit | — | — | — | **not implemented** |
| path-level verification | — | — | — | **not run** |
| context-scaling proof | — | — | — | **not run** |
