# TAC-OSM Claims Ledger

Every claim the repository makes, with its status and its blocker. The ledger
exists so that a claim cannot drift from "measured" to "supported" to
"established" without an entry changing here.

Conventions:

* **STATUS** is one of `SUPPORTED`, `PARTIALLY SUPPORTED`, `UNTESTED`,
  `REFUTED`, `NOT ESTABLISHED`.
* `SUPPORTED` requires a named experiment, a named commit, and a gate that was
  run. A plausible number is not a status.
* `UNTESTED` means the claim may well be true but the repository cannot
  currently distinguish it from false. That is a statement about the
  measurement, not about the world.
* `NOT ESTABLISHED` is the status for a measured-and-honest negative: the
  measurement was made correctly, nothing was found, and no explanation is
  offered.
* A `BLOCKER` names the specific thing that must change for the status to
  move. An entry without one is either done or unfixable.

### Provenance fields

Every entry also carries five fields, which are the reason a claim here cannot
be quietly upgraded from "measured once" to "established":

* **TYPE** — `mechanism` · `protocol` · `lesson` · `negative` · `gate` ·
  `core claim`. A *mechanism* is working machinery, a *protocol* is a way of
  measuring, a *lesson* is a constraint on interpretation only. The type
  determines what an entry can be used *for*.
* **LAYER** — `L0` inherited · `L1` inherited and adapted · `L2` TAC-OSM
  mechanism result · `L3` integration result · `L4` core program claim.
  Defined in `docs/MEASUREMENT_LAYERS.md` §"Layer 0"; assigned per row in
  `docs/EVIDENCE_REGISTER.md`. The layer answers "may this be taken as given
  by the next experiment, or must it be measured again?"
* **PRIOR ART** — the external antecedent, if any. Blank means the claim is
  TAC-OSM's own, which is itself information: C4 and C8 had no antecedent
  anywhere in the portfolio.
* **NOT INHERITED** — the exclusions. The binding field. An import without its
  exclusions is an over-claim, and this is where the sentence that must not
  travel with a claim is recorded.
* **REQUIRED EVIDENCE** — what a claimant must produce to cite the entry, and
  what the experiment needed to run. An entry whose required evidence cannot
  be named is not ready to be cited.

---

## C1 — Persistence

> Information written at step `t` can be used at step `t + k` to change a
> decision the observation alone does not determine.

**STATUS: SUPPORTED, bounded**

**TYPE:** mechanism · **LAYER:** L2 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** the persistent-state container and carry/reset/shuffle
intervention vocabulary are imported from TAC-transformer `IdentityState`
(E3, TAC-235/236, `6cce2ce`). The causal write/read claim is TAC-OSM's own.

**NOT INHERITED:** semantic or learned long-horizon memory. The measured system
uses an explicit vector store and deterministic state-conditioned router.

**REQUIRED EVIDENCE:** `TACOSM-TEMPORAL-001` with an enforced write-at-t,
read-at-t+k boundary and declared interventions.

`TACOSM-TEMPORAL-001`, run 36509506303, passed the causal boundary gate for
all 30 seed×delay cells. Across H=64, seeds 0–4, and k ∈ {1,2,4,8,16,32},
carry achieved `decision_success = 1.0000` at every delay. Reset achieved
0.0100 at k=1,2,4,8,16 and 0.0140 at k=32; corruption achieved 0.0000 at
every delay. The carry-minus-control difference therefore remained large at
every tested delay.

The result supports the bounded mechanism claim: an explicitly written state
value remained usable after up to 32 enforced intervening decision boundaries
and causally changed the later decision in this synthetic control.

It does **not** establish semantic persistent memory, learned persistent
representation, or the broader PLM intelligence claim.

The prior v0 C1 result is retired as temporal evidence because its task
construction did not enforce the write/read delay.
## C2 — Relational routing

> A cheap linear scorer over a hand-designed basis can learn the relevance
> relation from outcomes alone, at `O(1)` per candidate.

**STATUS: SUPPORTED**

**TYPE:** mechanism · **LAYER:** L2 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** `RelationalBanditRouter`'s key-gated feature map (PNDS Stage 2c,
E2, `42f9814`) — the strongest router evidence in the portfolio and the basis
TAC-OSM's learned router is built on. The map is L1: the mechanism is imported,
and the adaptation — running it inside a decision loop with persistence and
verification — is re-verified locally by the §34 gate.

**NOT INHERITED:** the source's causality (Stage 2c is a bandit, not a loop);
that the learned arm reaches the oracle ceiling (source: 0.8853 vs 1.0 at 8
candidates); generality beyond `dim=8` bit descriptors. The source's
large-population behaviour is also not inherited, because it was never
measured there.

**REQUIRED EVIDENCE:** the named experiment, with the representability gate
passing jointly — a learned success without it is an uninterpretable success.

`TACOSM-BASELINE-001`. Learned arm 0.4396 at 500 steps against `random`
0.1296 and `static` 0.0376. Cost is one dot product per candidate, no LM
forward pass — the property that keeps the `CDLTeacher` out of the runtime
path (`router.py::CDLTeacher.route` raises, deliberately).

Supported *jointly* with C3: the §34 representability gate
(`representability.py`) establishes that one shared analytic weight vector
separates gold on all three families with zero ties and zero violations, so a
learned-arm success is an optimisation result and a learned-arm failure is not
excusable as an expressiveness limit.

---

## C3 — Representability is a precondition, not a detail

> A perfect oracle conceals a broken feature basis indefinitely.

**STATUS: SUPPORTED**

**TYPE:** lesson · **LAYER:** L0 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** PNDS-URP v0.4 §34, added after `dd8f63c` was invalidated by
`f989430`. This is the single most transferable methodological asset in the
portfolio and it transfers wholesale: it is a *protocol*, not a mechanism.

**NOT INHERITED:** nothing — the lesson is architecture-neutral by
construction, which is exactly why it is L0.

**REQUIRED EVIDENCE:** the gate itself, run as a compulsory pre-training
failure rather than a retrospective audit.

Provenance: the `dd8f63c` failure at `cdl-attention-experiment`, where
`PersistentStateRouter._features` never read `state.key` and analytically
ideal weights scored exactly 0 for every candidate while `true_key`, a control
arm reading the relation directly, reported 1.0000 for four consecutive
commits.

The gate is now compulsory pre-model (`builder.check_representability`), runs
on the real basis against real episodes with one shared vector across all
episodes, and is a build failure rather than a warning. Its own regression
test is `test_the_gate_fails_without_the_gated_slot_block`, which reproduces
the failure the audit found: without the gated slot block, no shared weight
vector separates gold for `state_lookup` or `replay`.

---

## C4 — Model-state integrity is a first-class gate

> An evaluation against untrained weights produces plausible-looking numbers
> and is not detectable from its outputs.

**STATUS: SUPPORTED**

**TYPE:** gate · **LAYER:** L2 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** none. This failure mode was discovered inside TAC-OSM, in the
HS-001 sweep; it has no antecedent in the portfolio's prior work, which is
worth recording — the most expensive methodological lesson here was new here.

**NOT INHERITED:** the claim cannot be inherited by any external consumer
either, because it is a claim about *their* harness. It transfers only as the
gate itself, not as evidence about their system.

**REQUIRED EVIDENCE:** the gate running unconditionally in every measurement
script, and the regression test that reproduces the invisibility.

This is the claim the TACOSM-HS-001 measurement error established, and it is
now pinned as a gate and as tests.

The failure route, exactly: `learn=False` + weights never copied into the
evaluation router → `w = [0]*n` → a softmax over identical scores → uniform →
entropy `ln(H)` → `routing@1` read as ~1/40 and interpreted as a finding about
the router. It was a finding about the harness. A zeroed parameter vector
produces a *smooth, bounded, sensible-looking* distribution, so only the
parameters can catch it.

Gate: `integrity.py` — `parameter_hash`, `is_untrained`, `Checkpoint`,
`snapshot_router`, `assert_trained`. Runs unconditionally in
`measure_history_scaling.py` and `measure_retrieval_ceiling.py` before any
measurement. Failure raises `IntegrityError` rather than returning a flag, for
the same reason a failed representability gate is a build failure. The
checkpoint hash is recorded in the run manifest alongside the untrained hash
at the checkpoint's own width, so a reader verifies the comparison rather than
trusting that it happened.

`scripts/archive_trained_weights.py` snapshots a trained state per seed so an
evaluation has something to load. The archive is regenerated on demand from
deterministic seeds and `results/*.json` is gitignored, so the *gate* is the
committed part and the weights it verifies are reproduced rather than stored.

Tests: `tests/test_integrity.py`, including
`test_the_zero_weight_router_scores_uniformly` (the invisibility) and
`test_the_zero_weight_router_is_caught_before_evaluation` (the gate).

---

## C5 — Execution cost scales with the relevant subset, not total history

> `C_executed ≈ f(|R|)`, so executed computation depends on the relevant
> subset rather than on the size of the history.

**STATUS: UNTESTED**

**TYPE:** core claim · **LAYER:** L4 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** none. This is the program's thesis, not a finding, and it has
no antecedent in the portfolio.

**NOT INHERITED:** everything. In particular, the flat `C_executed` column in
HS-001 is **not** support for this claim in either direction: a constant 10 at
every H means the architecture has no room to scale, not that scaling has been
demonstrated.

**REQUIRED EVIDENCE:** a capability-vs-computation curve with total compute and
relevant compute measured **separately** along the same axis, not one derived
from the other, with the capability parity margin declared before the
confirmatory run.

**BLOCKER:** v0.1 executor has fixed `active_count = max_nodes = 10`. Until
M2.1 there was no retrieval or index boundary in the loop, so `|R|` was not a
defined quantity and the claim was not merely unsupported — it was *unstatable*
as a measurement. **M2.1 changed the first half of that and not the second.**
The boundary now exists (`src/tac_osm/retrieval.py`), `|R|` is now a defined
quantity and is reported per arm as `candidates_inspected`, and the index's
cost term is registered.

**TACOSM-C5-001 ran and is VOID — see below.** A confirmatory run *was* made
(run 36512989760, real pinned CASM-S at `c315544`) and it does not move this
claim, because its capability half was never measured: the reference arm
collapsed to the population base rate and the indexed arm's success was
guaranteed by the benchmark's own acceptable-action construction. The claim
therefore stays UNTESTED, and the blocker became the *instrument*, not the
mechanism's absence. `docs/TACOSM-C5-001-RESULT.md` records the four failure
modes and retains the artifact.

**TACOSM-C5-002 ran, and its gate fired — see below.** The successor was
registered to fix C5-001's specific defect: it separated retention, execution
correctness and work into three endpoints and added a representability gate
that runs *before* the task stream and terminates the run as instrument-invalid
if the frozen CASM-S cannot separate the relation's satisfier from a
one-bit-flipped violator. The gate fired. It separated **59 of 256** held-out
pairs (0.2305) against a pre-registered minimum of 0.5, so the run terminated
before any arm was measured and produced `"cells": []` — no capability table,
no capability number, nothing to misread.

The claim therefore still stays UNTESTED, and the blocker is still the
instrument. But the blocker is now *diagnosed*, and the diagnosis is a
measurement rather than a guess: the bridge model reports **0.84375** absolute
output accuracy on held-out circuits while separating satisfier from violator
in only **0.2305** of held-out pairs. One model, one graph family, one run. The
instrument's accuracy score is blind to its separation score — which is the
exact error that let C5-001 complete on an unrepresentative model, now caught
by a gate instead of by a retrospective void. See
`docs/TACOSM-C5-002-RESULT.md`.

**The successor needs a new bridge, not a new experiment ID.** The gate's
failure localizes to the bridge's training target: it is trained on each
candidate's Boolean output in isolation, and nothing in that objective rewards
within-query ranking, so a model reaching 0.84375 absolute accuracy is
*behaving as trained*. The registered fix is a bridge trained on **pairs** with
a loss on separation, not on single candidates with a loss on absolute level.
The four candidate causes named in C5-001's void (soft-op `alpha` scaling, the
bridge training target, the 0.5 threshold, unlearned gate routing) remain
unresolved — no run to date separates them — and the gate threshold stays at
0.5 because it was pre-registered before the run and is not tunable after
seeing the result.

**TACOSM-C5-003 is pre-registered and not run.** It is the first fresh
capability experiment after the C5-002 diagnosis, and it changes two things
only: the bridge is trained on **pairs** with a loss on separation
(`BRIDGE_OBJECTIVE = "pair_separation"`), so the objective contains the
quantity the downstream decision actually depends on; and the gate is a
representability gate with **eight registered criteria** evaluated before the
task population runs (`docs/TACOSM-C5-003.md`). The eight are: frozen
threshold constants; held-out pair accuracy at the registered minimum;
non-degenerate output spread; actual verifier acceptance on the satisfying
half of each held-out pair (amendment A2 — `RelationConstraintVerifier` is a
success verifier, so two-sided acceptance is capped at one half by
construction; and amendment A3 — the outcome's `value` field carries the
model's own output, because `verify()`'s `1e-9` output-consistency check would
otherwise require the alpha-scaled soft Booleans to emit exactly `1.0`, which
they reach only on a measure-zero parameter manifold); no dependence on
`true_edge_set`; no oracle information
entering routing or execution; deterministic reproduction from the frozen
checkpoint; and the pair-trained objective itself. Everything else is
deliberately unchanged from C5-002 — same population, same relation, same
arms, same work accounting, same separation contract, same CASM-S pin, same
threshold — except the bridge validation stream's seed, which C5-002 derived
as the gate's own seed and C5-003 names separately (amendment A1).

C5-003 is registered as an **integrated-boundary** experiment rather than a
better gate: it tests the four interfaces as one causal chain — `(S,Q) -> R`
addressing, `(R,Q) -> A` execution, `(A,O) -> V` verification, `R -> W(R)`
work — and the gate decides whether that chain is attached at all. This is the
design point C5-001's compound `success_rate` could not reach: a number that
merges the interfaces cannot distinguish an addressing failure from an
execution failure, and that distinction is what makes a negative result
interpretable.

**The claim still stays UNTESTED, and the blocker is still the instrument.**
A pre-registration is not a result, and nothing above moves C5. The successor
is now **written but not run**: `scripts/measure_c5_casm_003.py` exists and is
registered in `_WITH_CONTRACT` (`tests/test_contract.py`), its 30 torch-free
design tests pass, and its `--smoke` path enforces the contract on this
control plane — but the measurement itself requires `torch` and the pinned
CASM-S checkout, so it is runner-only by definition and no run has been
dispatched. The compute lane is
`.github/workflows/tacosm-c5-003.yml` (branch `compute/c5-003`). Until a
gate-confirmed run exists, this entry must not be read as progress on C5 — it
is a registered intention with a tested contract, recorded so that the
difference between "the experiment is designed" and "the claim is supported"
stays visible.

This is the architecture's central efficiency claim and it is *not* supported
by any measurement in the repository. The honest model of v0.1 cost is:

```
C_total(H) = O(H) routing + O(10) execution + O(verification)
```

not `C_total ≈ O(R)`. `C_executed` reads 10 at every H in TACOSM-HS-001
because the structural executor fixes `active_count = max_nodes`; a flat
column there means the architecture has *no room* to scale, not that scaling
has been demonstrated. `C_router` is the only cost term that genuinely varies,
and it grows linearly with H.

The claim becomes *measurable* at `M2.1` (`docs/ROADMAP.md`), where an index
inserts a retrieval boundary between routing and execution — and M2.1 has
built the boundary. What it has not done is run it: the capability-versus-
computation curve is the registered evidence this claim needs, and a
confirmatory run is what would produce it. Until that run exists the claim
must not appear in any report or figure as though measured, however clearly
`|R|` is now defined.

---

## C6 — Large-H routing failure is a routing problem

> Under a frozen router, task accuracy falls as history grows because
> relevance routing degrades, not because the environment becomes harder.

**STATUS: SUPPORTED**, and **narrowed by TACOSM-MATCHED-001** — see the note
below. The degradation is real and is attributable to the model, not the
environment. The *reason* it happens is not the one this claim originally
implied.

**TYPE:** mechanism · **LAYER:** L2 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** none for the degradation; the *narrowing* is TAC-OSM's own.

**NOT INHERITED:** the original mechanism. The claim as first written pointed
at "a fixed capacity of a linear scorer to separate gold from its best
distractor". That reading is **refuted** by the analytic vector, and the
sentence that must not travel with this claim is "capacity limitation of a
linear scorer".

**REQUIRED EVIDENCE:** the attribution controls that separate this claim from
its refuted mechanism: `oracle = 1.0000` (environment not the cause) and
`exec|route` (execution not the cause).

`TACOSM-HS-001`, commit `91597ab`, re-verified with the integrity gate in
place (see below). Oracle = 1.0000 at every H, so the environment is not
producing the degradation. `exec|route` stays high where routing succeeds
(0.9620 at H=2), so the loss is not execution.

| H | accuracy | routing@1 | gold_rank | recall@4 | exec\|route | C_router |
|---|---|---|---|---|---|---|
| 2 | 0.9220 | 0.9460 | 1.05 | 1.0000 | 0.9620 | 2 |
| 4 | 0.8000 | 0.8300 | 1.20 | 1.0000 | 0.9349 | 4 |
| 8 | 0.6120 | 0.6160 | 1.59 | 0.9860 | 0.8893 | 8 |
| 16 | 0.4360 | 0.4800 | 2.20 | 0.9060 | 0.7779 | 16 |
| 32 | 0.2840 | 0.3000 | 3.70 | 0.6960 | 0.7048 | 32 |
| 64 | 0.1260 | 0.1620 | 6.40 | 0.5300 | 0.5019 | 64 |

The router is trained once at 8 candidates and copied verbatim to every H, so
the sweep measures transfer, not retraining.

**Re-verification.** The table above was produced again with the model-state
integrity gate enforced and weights loaded through `load_weights`; every digit
reproduced. The frozen numbers are stable, and the corrected measurement that
produced them is confirmed rather than merely preserved.

**The HS-001 correction, recorded.** An earlier version of this sweep reported
`routing@1` ≈ 1/40 at H=32 and concluded the router's decision was
meaningless. That number came from a router with no weights loaded. With
weights copied, `routing@1` at H=32 is 0.4250 and the decision is clearly
informative. The *interpretation* was wrong; the corrected table is the
result. This is the failure that motivated C4.

**Narrowed by TACOSM-MATCHED-001** (`docs/TACOSM-MATCHED-001.md`, this
commit). The claim says routing degrades, which remains true. Its original
*mechanism* — that a linear scorer has "a fixed capacity to separate gold
from its best distractor, and that capacity is diluted as the distractor pool
grows" — is **refuted**. The analytic vector, one shared weight vector with no
training, reaches `routing@1 = 1.0000` at H ∈ {8, 64, 256, 512} with
`delta_1 ≈ +3.07`. The basis does not dilute. What degrades is the *trained*
router, because REINFORCE sees 5 successes in 500 steps at H=256 (a 1% rate
against 0.39% chance) and its `(1 − p_selected)` multiplier is squeezed by a
flat softmax over a large candidate set.

The measured degradation is unchanged. The sentence to strike from any
description of it is "capacity limitation of a linear scorer"; the sentence to
keep is "routing degrades, and the environment does not".

---

## C7 — The top-K signal survives

> The router is a reasonable relevance scorer but a poor large-scale
> retrieval mechanism: it retains substantial information about where the
> relevant item is even after its top-1 precision has collapsed.

**STATUS: SUPPORTED**, at `H ≤ 64`. **PARTIALLY SUPPORTED** overall. The
qualification's *cause* is now identified — it is a training artefact, not a
representation limit (see below).

**TYPE:** mechanism · **LAYER:** L2 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** none.

**NOT INHERITED:** `H ≥ 128`, where the signal does **not** survive on a
transfer-trained router. The qualification is part of the claim, not a caveat
to it.

**REQUIRED EVIDENCE:** `TACOSM-RETRIEVAL-001` (F0) establishes that the signal
survives; `TACOSM-SURROGATE-001` (F3) establishes the row that matters for
F1 — that the large-H top-K signal is *responsive to training*, improving
materially under a training-only intervention. Neither alone supports the
claim F1 inherits.

**BLOCKER:** the claim becomes a capability claim only when a retrieval
boundary exists, at which point it is F1's to make. Until then it is a
mechanism claim about a scorer with no index in front of it.

`TACOSM-RETRIEVAL-001` F0, this commit. The existing router scores all H
candidates; retaining the top-K by score:

| H | rec@1 | rec@2 | rec@4 | rec@8 | rec@16 |
|---|---|---|---|---|---|
| 8 | 0.616 | 0.854 | 0.986 | — | — |
| 32 | 0.300 | 0.490 | 0.696 | 0.904 | 0.998 |
| 64 | 0.162 | 0.306 | 0.530 | 0.748 | 0.920 |
| 128 | 0.096 | 0.172 | 0.314 | 0.550 | 0.762 |
| 256 | 0.054 | 0.088 | 0.174 | 0.336 | 0.550 |

At H=64 the curve is exactly the shape that makes search the bottleneck:
0.162 → 0.530 → 0.748 → 0.920 as K goes 1 → 4 → 8 → 16, against an environment
whose oracle is 1.0000.

**The qualification is the point of the measurement.** At H=128 and H=256 the
curve *does not* saturate well: rec@16 falls to 0.762 and 0.550. The
hypothesis "the scoring representation is adequate and only search is
missing" is supported at `H ≤ 64` and is **not supported** at `H ≥ 128`.

**The follow-up is run, and both candidate causes are refuted.** The two
readings the claim left open were:

1. the representation degrades gracefully but genuinely as the candidate
   population grows;
2. the router was trained at 8 candidates, so the sweep measures transfer,
   and the degradation at large H is a training-horizon artefact.

`TACOSM-MATCHED-001` ran the matched-H design that separates them and
produced **neither**. Reading 1 is refuted by the analytic vector reaching
`routing@1 = 1.0000` at H ∈ {8, 64, 256, 512}. Reading 2 is refuted by the
matched-H matrix: a router trained at H=256 is *worse at H=256*
(`routing@1 = 0.0740`) than one trained at H=8 (`0.0640`), and the H=8 row is
the best row at every evaluation population.

The failure is in the **learning dynamics** — reward scarcity under REINFORCE
at large H — not in the basis and not in the training population. The claim's
qualification is therefore *not* a property of the scoring representation, and
must not be read as one in any description of F1's viability. See
`docs/TACOSM-MATCHED-001.md`.

---

## C8 — Scheduler integrity

> A per-family accuracy table must not be an exposure table, and two families
> must not emit the same task stream.

**STATUS: SUPPORTED**

**TYPE:** gate · **LAYER:** L2 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** none. Discovered inside TAC-OSM during baseline preparation.

**NOT INHERITED:** nothing — the gate is a property of this generator, pinned
by a test that fails 25/25 on the old code.

**REQUIRED EVIDENCE:** `test_relational_and_state_lookup_do_not_collide_on_a_pinned_run`.

Discovered during baseline preparation: `relational` and `state_lookup`
called `_build_candidates` with the same seed, marks, dim, `n_candidates` and
noise, so under a pinned single-family config they emitted *identical* tasks
(60/60). A pinned per-family table built that way measures one stream under
two labels.

Fixed by a family-distinct RNG offset in `build_lookup_task`, and pinned by
`test_relational_and_state_lookup_do_not_collide_on_a_pinned_run`, which fails
25/25 on the old generator. The correct decomposition is one mixed run split
by `step.query.provenance`, which holds the stream, exposure and schedule
identical across families — the method `measure_baseline.py` uses.

The underlying cause was coupled randomness (`seed + len(families)` deriving
the task seed), which is why `M1.1` of the roadmap makes independent RNG
streams a hard contract rather than a style preference.

---

## C9 — `state_lookup` non-monotonicity

> Learned accuracy on `state_lookup` is non-monotonic in training steps:
> 0.2600 → 0.3463 → 0.2743 at 60/200/500.

**STATUS: NOT ESTABLISHED**

**TYPE:** negative · **LAYER:** L2 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** none.

**NOT INHERITED:** an explanation. This is the point of the entry: the three
candidate causes were checked and ruled out, and no explanation is offered
because none was found.

**REQUIRED EVIDENCE:** none is planned. Rationalising the curve would be worse
than leaving it here.

Recorded rather than explained. The three things that could have caused it
were checked and ruled out: the family streams are now distinct (C8),
exposure is equal under the mixed schedule, and the representability gate
passes. No explanation is offered, because none was found. A non-monotonic
curve with a clean instrument is an observation, not a result, and
rationalising it would be worse than leaving it here.

---

## C10 — Falsified interventions do not falsify the mechanism

> A negative intervention experiment licenses a claim about the intervention,
> not about the mechanism it was trying to improve.

**STATUS: SUPPORTED** — as a methodological claim, by the argument below. It
is in the ledger because it is the constraint that keeps `TACOSM-LEARN-001`
(F2) honest, and because it is the kind of constraint that gets quietly
dropped when a result is inconvenient.

**TYPE:** lesson · **LAYER:** L1 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** the distinction is standard practice in experimental design;
the *enforcement* — pre-committing the interpretation order before the run and
recording the branch that occurred — is the PNDS-URP pre-registration protocol
adapted to this repository.

**NOT INHERITED:** any licence to pronounce on the mechanism. The reason is
not rhetorical: the hypothesis class provably contains a perfect H-invariant
solution, so a mechanism whose hypothesis class is adequate cannot be
pronounced inadequate by a failure to find the solution with one optimiser.

**REQUIRED EVIDENCE:** a pre-registered decision rule with its consequences
attached, committed before the run and unamended whatever the result.

F2 tests two named training-time interventions — epsilon-greedy with
`eps_0 = 0.30` linearly decayed, and a temperature schedule from `2.0` to
`0.5` — against the frozen MATCHED-001 baseline. If neither lifts
`routing@1` at `train-H = 256`, the claim F2 is entitled to make is:

> these two interventions do not fix the trained router at H=256 under this
> protocol

The claim it is *not* entitled to make is:

> persistent-state relevance routing does not scale

The reason is not rhetorical. It is a measurement-layer argument: the
hypothesis class provably contains a perfect, H-invariant solution
(`analytic_weights`, `routing@1 = 1.0000` at H ∈ {8, 64, 256, 512},
`delta_1 ≈ +3.07`), so the relation is expressible and the environment is
unambiguous (`oracle = 1.0000`). A mechanism whose hypothesis class is
adequate cannot be pronounced inadequate by a failure to *find* the solution
with one optimiser. That is the difference between an optimisation result and
a representability result, and conflating them is precisely the error class
that the `dd8f63c` audit and the §34 gate exist to catch — in the other
direction, where an oracle passing was read as the basis being adequate.

The distinction the ledger holds:

- **"the current router doesn't scale"** — one trained linear scorer under one
  update rule. An intervention experiment can address this.
- **"persistent-state relevance routing doesn't scale"** — the mechanism.
  No single intervention experiment can address this.

F2's negative-result inspection order is pre-committed as
`representation → training dynamics → population discrimination → retrieval
architecture`, and F1 re-queues only on F2's primary endpoint firing or on the
top-K signal being shown intact. Both conditions are recorded in
`docs/TACOSM-LEARN-001.md`.

**F2 HAS RUN, AND THE CONDITIONAL FIRED.** Neither arm lifted the primary
endpoint: `routing@1` at `train-H = 256` moved -0.0220 (`epsilon_greedy`) and
-0.0140 (`temperature`) against a 0.0600 materiality threshold, i.e. within
seed noise, and `recall@16` at H=256 moved against both arms — materially so
for `epsilon_greedy` (-0.0760). The reproduction gate passed first (the
baseline arm reproduced all nine published MATCHED-001 numbers exactly), so
the comparison is against the published reference column.

So the claim F2 makes is the one above: **these two interventions do not fix
the trained router at H=256 under this protocol.** C10 is what keeps that
sentence from becoming the other one. The negative result *did* locate the
failure more narrowly than before — `epsilon_greedy` explored as registered
(15.2% of steps) and saw more successes per step at H=256 (3.8 vs 2.4, against
a 0.0039 chance rate), and the endpoint still did not improve, which rules out
*selection* as the binding constraint and leaves the learning rule, the
gradient signal, and the reward shaping as the candidates. None of that is a
verdict on the mechanism.

---

## C11 — Dense supervision improves top-K retrieval, not exact routing

> Increasing the density of the gold-anchored learning signal substantially
> improved top-K retrieval under the tested conditions, while failing to
> produce a material improvement in exact top-1 routing.

**STATUS: SUPPORTED, bounded** — `TACOSM-SURROGATE-001` (F3), commit
`5943166`.

**TYPE:** mechanism · **LAYER:** L3 — see `docs/EVIDENCE_REGISTER.md`

**PRIOR ART:** the margin objective and the state-independent baseline are both
standard (L0). What is TAC-OSM's own is applying them to *this* router at
*this* population — an integration hypothesis, not a technique.

**NOT INHERITED:** "the router learned relevance." The surrogate is anchored to
the gold index exactly as `float(outcome.success)` is — the environment scores
`success = action == target_action` — so both rewards are gold-anchored scalars
of the same logical kind and the intervention is the *density* of the signal,
with the anchor held fixed. A positive result licenses a statement about
learnability under dense supervision, not about outcome learning.

**REQUIRED EVIDENCE:** the registered run, with the reproduction gate passing
first — all nine published MATCHED-001 baseline cells reproduced with
`diff = 0.0000`, which is what makes every delta below a comparison against
the frozen reference.

At `train-H = 256`:

| endpoint | baseline | `analytic_margin` | `analytic_margin_clipped` |
|---|---|---|---|
| `routing@1` | 0.0740 | 0.1140 | 0.0980 |
| `recall@16` | 0.3020 | **0.3880** | **0.5480** |
| `gold_rank` | 74.6960 | 48.0900 | **25.8260** |
| successes / 500 | 2.4 | 3.8 | 2.0 |

`Δ(routing@1)` = +0.0400 and +0.0240 against a pre-registered materiality
threshold of 0.0600 — **within seed noise**. `Δ(recall@16)` = +0.0860 and
+0.2460 — **material**. The intervention landed as registered: the reward's
non-zero rate went from 0.0048 to 0.9980 at H=256.

**The asymmetry is the claim.** Gold moved closer to the top of the ranking
without separating from the single best distractor — `delta_1` at H=256 does
not move (−0.4383 → −0.4749) while `gold_rank` improves by a factor of three.
The gain is in the bulk of the score distribution, not at the top.

**BLOCKER:** the bound is `K ≥ 2`. At `K = 1` the endpoint did not fire, and
exact top-1 routing at H=256 remains at 0.0740–0.1140 against a 1.0000 oracle
and a 1.0000 analytic vector. The claim does not extend to C2, does not extend
C7's `H ≥ 128` qualification, and does not resolve the learning problem — it
moves the evidence frontier. Per the pre-registered decision rule, F1
re-queues on this top-K evidence alone, recorded as "top-K signal intact" and
*not* as "`routing@1` improved".


---

## C12 — Semantic state addressing through opaque addresses

> A learned state-addressing mechanism can identify the persistent item whose
> stored semantic content satisfies a query, even when the item's address is
> opaque and absent from the query.

**STATUS: SUPPORTED, bounded** — `TACOSM-STATE-REP-006`, clean registered run
`36664979180`, code head `b548642c397b4c9289761e65f11bd4327ccec86b`.

**TYPE:** mechanism · **LAYER:** L2

**PRIOR ART:** generic content-addressed retrieval is established prior art;
the specific TAC-OSM result is the tested integration of semantic state values,
opaque addresses, and the temporal state boundary.

**NOT INHERITED:** scalable addressing, long-horizon memory, learned indexing,
or arbitrary memory semantics.

**REQUIRED EVIDENCE:** the registered REP-006 protocol and its precondition
gate. The clean run passed all **841 tests** before execution.

At M=8 state items:

| seed | learned Top-1 | no-learning Top-1 |
|---:|---:|---:|
| 0 | 0.5117 | 0.0156 |
| 1 | 0.7422 | 0.2188 |
| 2 | 0.2695 | 0.0156 |
| 3 | 1.0000 | 0.4883 |
| 4 | 1.0000 | 0.1914 |
| pooled | **0.7047** | **0.1859** |

The learned arm exceeds the frozen control in every seed.

The analytic identity witness is 1.0000 at every seed. The reset intervention
fails closed on every seed, and the opaque address set survives the value-shuffle
control.

**BLOCKER:** M=8 is a small synthetic population. The implementation performs
an exhaustive O(M) scan. The result does not move C5.

A secondary end-to-end metric from the first clean run was quarantined because
its target-rank helper returned selected-argmax rank rather than gold rank. A
separate exact target-rank diagnostic was added; no claim above depends on the
quarantined endpoint.

---

## C13 — Semantic state addressing survives a bounded population scaling

> Under the registered synthetic task and fixed learner, learned semantic
> state-address recall remains above the no-learning control across
> M ∈ {2,4,8,16,32}, while addressing arithmetic grows linearly with M.

**STATUS: SUPPORTED, bounded** — `TACOSM-STATE-REP-007`, workflow
`36665305689`, artifact `11075258348`.

**TYPE:** mechanism · **LAYER:** L2

**PRIOR ART:** the general content-addressed retrieval pattern is inherited;
the population curve is TAC-OSM's measurement.

**NOT INHERITED:** sublinear lookup, an asymptotic complexity improvement,
long-horizon memory scaling, or capability parity at large H.

**REQUIRED EVIDENCE:** the registered five-level M curve with five seeds per
level, identical task/learner, analytic witness and no-learning control.

Learned pooled Top-1 recall by state population:

| M | learned | no-learning | chance |
|---:|---:|---:|---:|
| 2 | **0.9922** | 0.5273 | 0.5000 |
| 4 | **0.9828** | 0.2734 | 0.2500 |
| 8 | **0.7141** | 0.1977 | 0.1250 |
| 16 | **0.4125** | 0.1813 | 0.0625 |
| 32 | **0.3344** | 0.1563 | 0.0313 |

The learned arm exceeds the frozen control at every registered M, while the
analytic witness remains exactly 1.0000 throughout.

The key limitation is visible in the same table: absolute recall declines
substantially as M increases. The result therefore demonstrates bounded
robustness of the semantic addressing relation, not scale invariance.

Addressing arithmetic is:

`C_state = 40 + 48M` MACs.

Measured accounting matches the formula:
136, 232, 424, 808, and 1,576 MACs at M=2,4,8,16,32 respectively.

Because the implementation scans all M items, the addressing stage remains
**O(M)**.

**BLOCKER:** replace exhaustive scanning with a real learned or deterministic
index, then measure capability and compute separately. Until that experiment,
these results cannot support a selective-computation claim.


---


---

## C14 — The hardened selective runtime actually routes only the retained subset

> For the controlled static equality workload, the indexed runtime preserves
> capability while presenting only the retained subset R to the router.

**STATUS: SUPPORTED, bounded** — `TACOSM-SELECTIVE-001`, workflow
`36666455184`, artifact `11075954570`.

**TYPE:** protocol · **LAYER:** L3

**PRIOR ART:** retrieval-before-compute is established in systems such as
retrieval-augmented transformers and sparse routing; the TAC-OSM result is the
measured runtime boundary in this repository.

**NOT INHERITED:** semantic retrieval, learned addressing, execution-work
scaling with R, or the L4 C5 claim.

**REQUIRED EVIDENCE:** the registered H={8,64,256}, K={2,4} matrix with a
single static candidate universe reused across 100 queries per cell.

The indexed arm preserved exact task success at **1.0000** in every cell while
the router saw exactly K candidates per query:

| H | K | indexed success | router candidates/query | amortized candidate work |
|---:|---:|---:|---:|---:|
| 8 | 2 | 1.0000 | 2.00 | 4.08 |
| 8 | 4 | 1.0000 | 2.00 | 4.08 |
| 64 | 2 | 1.0000 | 2.00 | 4.64 |
| 64 | 4 | 1.0000 | 4.00 | 6.64 |
| 256 | 2 | 1.0000 | 2.00 | 6.56 |
| 256 | 4 | 1.0000 | 4.00 | 8.56 |

The exhaustive arm routes over H candidates. One-time index construction is
reported separately and amortized over exactly 100 queries.

This establishes the **runtime retention boundary** in the controlled equality
instrument.

**BLOCKER:** the executor performs one selected-candidate invocation, so this
experiment does not measure execution work proportional to |R|. A direct
batched-execution capability/compute experiment is required before C5 can move.



---

## C15 — Execution work can be bounded by a retained relevant subset in a controlled workload

> When the correct output requires all relevant candidate programs, an indexed
> runtime can preserve capability while executing only the retained relevant
> subset rather than the full candidate population.

**STATUS: SUPPORTED, bounded** — `TACOSM-C5-EXEC-001`, workflow
`36666872579`, code head `625ee3311b200ed7eae22879bd3c34f97f4a39b8`.

**TYPE:** mechanism · **LAYER:** L3

**PRIOR ART:** selective/conditional computation and retrieval-before-compute
are established systems patterns. The TAC-OSM result is the controlled
execution-work measurement below.

**NOT INHERITED:** semantic retrieval, learned indexing, hardware FLOPs, or the
full L4 C5 claim.

**REQUIRED EVIDENCE:** fixed-R capability/parity experiment over
H={64,128,256}, five seeds, 100 queries per cell, with one-time index
construction separated from per-query execution work.

The task contains exactly **R=4** relevant programs at every H. The exhaustive
arm executes all H candidate programs; the indexed arm executes exactly four.

| H | exhaustive success | selective success | exhaustive executor work | selective executor work | reduction |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.0000 | 1.0000 | 576 | 36 | 93.75% |
| 128 | 1.0000 | 1.0000 | 1,152 | 36 | 96.875% |
| 256 | 1.0000 | 1.0000 | 2,304 | 36 | 98.4375% |

The selective arm retained exactly four candidates at every H, so
`R/H = 0.0625, 0.03125, 0.015625` respectively.

The experiment therefore demonstrates the execution-side relation:

`C_execute = O(R)` with fixed `R=4`

while the exhaustive baseline scales as:

`C_execute = O(H)`.

The result is stronger than SELECTIVE-001's routing-only boundary because the
correct output requires aggregation across all retained relevant programs.

**BLOCKER:** the workload uses an exact public content-address index, a fixed
synthetic executor, and a deliberately controlled relevance relation. The
experiment does not establish semantic end-to-end computation or the broad L4
C5 claim.



---

## C16 — Persistent semantic selective computation can preserve capability while executing only R

> In the registered synthetic end-to-end loop, a noisy semantic query can cross
> a persistent state boundary, identify a relevant state value, retain exactly
> R=4 relevant programs from a population H, and execute only those R programs
> while matching the exhaustive H-program output.

**STATUS: SUPPORTED, bounded** — `TACOSM-C5-END-TO-END-001`, workflow
`36667352698`, clean run head `03bc42761a914051bc62d12588b842a1357a5618`.

**TYPE:** mechanism · **LAYER:** L3

**PRIOR ART:** retrieval-before-compute and selective/conditional execution are
established architectural patterns. The measured persistent-state + noisy-query
+ retained-program + aggregate-execution path is the TAC-OSM result.

**NOT INHERITED:** learned semantic indexing, realistic language workloads,
hardware FLOP/latency superiority, index-maintenance cost, or the broad L4 C5
program claim.

**REQUIRED EVIDENCE:** fixed M=64 persistent state, one-bit-noisy 10-bit query,
H={64,128,256}, R=4, five seeds, 100 queries per cell, separate exhaustive and
selective execution accounting.

All 15 registered seed/H cells matched the independent aggregate evaluator at
1.0000 in both arms.

| H | R | exhaustive executor work | selective executor work | reduction | R/H |
|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 768 | 48 | 93.75% | 0.0625 |
| 128 | 4 | 1,536 | 48 | 96.875% | 0.03125 |
| 256 | 4 | 3,072 | 48 | 98.4375% | 0.015625 |

The state index retained the target state at 1.000 and the candidate index
retained all four relevant programs at 1.000.

Thus the execution component satisfies the controlled relation
`C_execute = O(R)` while the exhaustive baseline is `O(H)`, with capability
parity in the registered workload.

**BLOCKER:** both indexes are hand-designed exact/controlled mechanisms and
the executor is synthetic fixed work. A realistic/learned semantic retrieval
experiment with full cost accounting is still required before the L4 C5 claim
can move.



---

## C16 — Persistent semantic selective computation can preserve capability while executing only R

> In the registered synthetic end-to-end loop, a one-bit-noisy query can cross a
> persistent state boundary, identify the relevant state value, retain exactly
> R=4 relevant programs from a population H, and execute only those R programs
> while matching the exhaustive H-program output.

**STATUS: SUPPORTED, bounded** — `TACOSM-C5-END-TO-END-001`, workflow
`36667352698`, clean run head
`03bc42761a914051bc62d12588b842a1357a5618`, artifact
`11077145266`.

**TYPE:** mechanism · **LAYER:** L3

**PRIOR ART:** retrieval-before-compute and selective/conditional execution
are established architectural patterns. The specific measured combination of
persistent state, noisy query recovery, candidate retention, and aggregate
execution is TAC-OSM's bounded result.

**NOT INHERITED:** learned semantic indexing, natural-language workloads,
hardware FLOP/latency superiority, dynamic index maintenance, or the broad L4
C5 program claim.

**REQUIRED EVIDENCE:** M=64 persistent state, one-bit-noisy 10-bit query,
H={64,128,256}, R=4, five seeds, 100 queries/cell, separate state addressing,
candidate retention, and execution accounting.

All 15 registered seed/H cells matched the independent aggregate evaluator at
1.0000 in both exhaustive and selective arms.

| H | R | exhaustive work | selective work | reduction | R/H |
|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 768 | 48 | 93.75% | 0.0625 |
| 128 | 4 | 1,536 | 48 | 96.875% | 0.03125 |
| 256 | 4 | 3,072 | 48 | 98.4375% | 0.015625 |

State target retention and all-four-program candidate retention were 1.000.

The selective execution workload therefore exhibits
`C_execute = O(R)` while the exhaustive control exhibits
`C_execute = O(H)`, under exact capability parity in the registered synthetic
task.

**BLOCKER:** state/candidate indexes are hand-designed exact/controlled and
the executor is synthetic fixed work. A learned semantic retrieval experiment
and a realistic end-to-end capability/compute curve are still required before
the broad C5 claim can move.

## Claims this repository does not make

Deliberately excluded, and recorded so their absence is a position rather
than an oversight:

* that CDL is runtime inference — it is a teacher, a ranking benchmark and a
  diagnostic upper bound, and its `O(N · C_LM)` cost is the thing the
  architecture exists to avoid;
* that the learned router approximates CDL's discrimination — measured
  teacher agreement is explicitly *not* the primary metric
  (`RoutingDecision.scores`);
* that a bigger model would fix large-H routing — untested, and after
  TACOSM-MATCHED-001 the evidence points the opposite way: the representation
  is adequate at every H tested, so expressiveness is not the limitation and
  capacity is not the fix;
* that `C_executed` has been shown to scale with `|R|` — see C5. In particular,
  the 128× reduction in submitted CASM-S work measured by `TACOSM-C5-001` is
  an **instrument-level** observation about the accounting and the pipeline,
  not a capability result, and it must not be cited as C5 progress. The
  experiment that produced it is void for capability inference;
* that the matched-H router is a better baseline than the H=8 one — it is
  worse at every evaluation population (see C7), and the H=8 row remains the
  F1 baseline;
* that the analytic vector is a solution — it is a hand-designed reference
  point proving expressibility, not an outcome-trained router, and routing
  with it is not a TAC-OSM result;
* that the `TACOSM-C5-001` void is a negative result about CASM-S — the
  evidence establishes only that *that measurement instrument* could not
  distinguish CASM-S execution capability from base-rate behaviour. The
  candidate causes (soft-op `alpha` scaling, gate routing, the 0.5 threshold,
  the bridge target) remain unresolved, and none is separated by the run.

---

## Ledger discipline

An entry changes when the measurement changes, not when the prose does. A
claim found in a doc with no entry here is unsupported by definition, and a
`SUPPORTED` entry whose experiment cannot be re-run is a `PARTIALLY
SUPPORTED` entry waiting to be discovered.
