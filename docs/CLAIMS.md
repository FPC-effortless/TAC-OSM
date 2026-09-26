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

---

## C1 — Persistence

> Information written at step `t` can be used at step `t + k` to change a
> decision the observation alone does not determine.

**STATUS: SUPPORTED**

`TACOSM-BASELINE-001`, commit `91597ab`. The `replay` family requires a
written vector on the marked positions *and* the public query's anti-match;
neither routing alone nor lookup alone solves it, and the learned arm reaches
it (0.5325 at 500 steps) while `static` and `full_context` score ~0.03–0.04.
The mixed schedule gives equal exposure across families, so the per-family
comparison is not an exposure artefact (test:
`test_mixed_schedule_gives_equal_family_exposure`).

The persistence interventions (`shuffled`, `reset`, `corrupted`, `random`,
`wrong_key`) are implemented in `state.py` and prefrozen into the ablation
matrix, so a dose–response ladder for this claim is a matrix cell rather than
a new experiment.

---

## C2 — Relational routing

> A cheap linear scorer over a hand-designed basis can learn the relevance
> relation from outcomes alone, at `O(1)` per candidate.

**STATUS: SUPPORTED**

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

**BLOCKER:** v0.1 executor has fixed `active_count = max_nodes = 10`; there is
no retrieval or index boundary in the loop.

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

The claim becomes testable at Stage F1, where an index inserts a retrieval
boundary between routing and execution. Until then it must not appear in any
report or figure as though measured.

---

## C6 — Large-H routing failure is a routing problem

> Under a frozen router, task accuracy falls as history grows because
> relevance routing degrades, not because the environment becomes harder.

**STATUS: SUPPORTED**, and **narrowed by TACOSM-MATCHED-001** — see the note
below. The degradation is real and is attributable to the model, not the
environment. The *reason* it happens is not the one this claim originally
implied.

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
the task seed), which is why Stage B of the roadmap makes independent RNG
streams a hard contract rather than a style preference.

---

## C9 — `state_lookup` non-monotonicity

> Learned accuracy on `state_lookup` is non-monotonic in training steps:
> 0.2600 → 0.3463 → 0.2743 at 60/200/500.

**STATUS: NOT ESTABLISHED**

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

---

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
* that `C_executed` has been shown to scale with `|R|` — see C5;
* that the matched-H router is a better baseline than the H=8 one — it is
  worse at every evaluation population (see C7), and the H=8 row remains the
  F1 baseline;
* that the analytic vector is a solution — it is a hand-designed reference
  point proving expressibility, not an outcome-trained router, and routing
  with it is not a TAC-OSM result.

---

## Ledger discipline

An entry changes when the measurement changes, not when the prose does. A
claim found in a doc with no entry here is unsupported by definition, and a
`SUPPORTED` entry whose experiment cannot be re-run is a `PARTIALLY
SUPPORTED` entry waiting to be discovered.
