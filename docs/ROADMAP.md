# TAC-OSM Roadmap

Derived from a review of public research and evaluation infrastructure
(OLMo/OLMES, lm-evaluation-harness, OpenAI Evals, Anthropic agent evaluation,
Harbor/Terminal-Bench, HELM, DeepMind research code, Karpathy's autoresearch),
against the current state of this repository.

The conclusion of that review: **the next improvement should not be a model
capability.** It should be making TAC-OSM a research-grade experimental system,
with the benchmark and evaluation machinery treated as part of the
architecture. Capability is then increased *inside* that controlled system.

## What the repository already has

- the `S_t → R_t → C_t → A_t → O_t → V_t → S_{t+1}` loop
- componentised ablations and a component matrix
- representability gates (§34), run on the real basis
- **a model-state integrity gate** — an evaluation cannot run against
  untrained weights and emit a plausible number (`src/tac_osm/integrity.py`,
  claim C4). It catches the failure class that produced a false conclusion in
  TACOSM-HS-001, and it runs unconditionally in the history-scaling and
  retrieval-ceiling scripts
- oracle / random / static / full_context / learned arms
- deterministic seeds and a verified mixed-family schedule
- generator integrity tests, including a family-stream-independence regression
- exact measurement commands under `scripts/`
- persistent state interventions, structural execution, verification and repair
- a frozen baseline (`TACOSM-BASELINE-001`) and a results ledger
- **a claims ledger** (`docs/CLAIMS.md`) — every claim with its status and its
  blocker, so a claim cannot drift from "measured" to "supported" without an
  entry changing
- **a retrieval ceiling** (`TACOSM-RETRIEVAL-001` F0) — the top-K signal
  survives top-1 collapse at `H ≤ 64` and does not at `H ≥ 128`
- **a closed intervention sequence** — MATCHED-001, F2 and F3 each closed one
  candidate location for the large-H routing failure, and the sequence is
  recorded in advance so it cannot be restated in an order that fits a later
  conclusion
- **an evidence spine** — a claims ledger (`docs/CLAIMS.md`) with five
  provenance fields per entry; a layer contract (`docs/MEASUREMENT_LAYERS.md`)
  defining `L0`–`L4`; an evidence register (`docs/EVIDENCE_REGISTER.md`)
  assigning the level of every row with its boundary; and a pre-experiment
  gate (`docs/EVIDENCE_MAP.md`) that answers *is an experiment actually
  necessary* before the compute is spent. Together these are what stops the
  repository re-measuring a result it already holds, or promoting a negative
  by accumulation.

## The research-integrity problem to fix first

Documentation and implementation have drifted apart. Statements equivalent to
"persistent write: not implemented" coexist with an implemented
verifier-gated commit and tests that exercise it, including the distinction
between an unconditional `state.write` and a verified commit.

Mature research infrastructure tries to eliminate exactly this. **The first
improvement is therefore a single machine-verifiable source of experimental
truth**, not another algorithm.

## The architectural correction

The system under test must not define its own score. Evaluation belongs
outside the model:

    Benchmark Spec        immutable, versioned
         │  hidden task truth
         ▼
      Runner              model + environment
         │  trajectory (raw, no headline score)
         ▼
      Grader              independent evaluator
         │
         ▼
      Analyzer            statistics + failure taxonomy
         │
         ▼
      Evidence ledger

## Milestones

Three milestones. They are the questions the thesis decomposes into, and each
is a *question*, not a deliverable:

| | M1 — Relevance | M2 — Selective computation | M3 — Economic scaling |
|---|---|---|---|
| asks | can useful, persistent-state-conditioned structure be *identified*? | does the system compute only over the relevant subset `R_t`? | does the capability remain economical as history grows? |
| answered by | the scorer and its learning dynamics | a retrieval boundary in the loop | a capability-vs-computation curve |
| the claim at stake | C2, C6, C7, C11 | C5 | the program thesis (L4) |
| fails if | nothing reliably identifies the relevant item | computation still scales with the whole history | parity is bought at a cost that does not beat full context |

The ordering is a dependency, not a preference. **You cannot measure whether
computation depends on the relevant subset until "the relevant subset" is a
defined quantity** — which is what M2's retrieval boundary supplies, and why
M1 precedes it: an index built over a scorer whose numbers mean what they say
is a sound measurement in neither direction. And M3 is a *curve*, which needs
both a scorer worth measuring (M1) and a boundary to measure against (M2).

The milestone structure replaced an earlier flat list of stages `A`–`I`. Those
IDs still appear in older commits and in the other documents, and the
mapping below is a contract, not a cleanup — a reference to `Stage D` in an
older document or a git log resolves to `M1.3`, and nothing is lost:

| Historical ID | Current ID | Status |
|---|---|---|
| Stage A — research kernel | `M1.0` | the measurement infrastructure |
| Stage B — benchmark hardening | `M1.1` | the measurement infrastructure |
| Stage C — freeze the baseline | `M1.2` | the measurement infrastructure |
| Stage D — ablation matrix | `M1.3` | the measurement infrastructure |
| Stage E — history scaling | `M1.5` | **done** — `TACOSM-HS-001` |
| Stage F, F0 — retrieval ceiling | `M1.6` | **done** — `TACOSM-RETRIEVAL-001` |
| Stage F2 — learning dynamics | `M1.7` | **done** — `TACOSM-LEARN-001` |
| Stage F3 — reward density | `M1.8` | **done** — `TACOSM-SURROGATE-001` |
| Stage F1 — efficient retrieval | `M2.1` | **re-queued** on M1.8's top-K evidence |
| Stage F4 — measurement layers | `M1.4` | the standing contract |
| Stage G — state formation | `M2.2` | not started |
| Stage H — repair | `M2.3` | not started |
| Stage I — language integration | `M3.2` | not started |
| — *(the curve, was nameless)* | `M3.1` | not started; this is the milestone the thesis needs |

**The measurement infrastructure is not a phase.** `M1.0`–`M1.4` are listed
first because everything else is uninterpretable without them, not because
they are finished. They are the standing contract, and they have no end date.

## Implementation order — M1: Relevance

M1 asks whether useful, persistent-state-conditioned structure can be
identified at all. It spans everything that makes the scorer and its training
trustworthy, because an M1 result is only as good as the measurement that
produced it.

### M1.0 — Research kernel *(was Stage A)*

`RunManifest`, `EpisodeSpec`, `Trajectory`, `EpisodeResult`,
`BenchmarkManifest`, `Evaluator`, `Statistics`. Every experiment becomes
reconstructable. Nothing here is a capability; all of it is what makes the
later numbers citable.

### M1.1 — Benchmark hardening *(was Stage B)*

`DEV` / `VAL` / `FINAL` splits, plus gates:

oracle succeeds · no-op fails · random is non-degenerate · gold is unique ·
gold not exposed to runtime · candidate permutation does not change truth ·
candidate position is randomised · family streams independent · nuisance
variables do not change truth · counterfactual variables do change truth ·
no train/val/test duplicates · evaluator cannot read runtime-private state ·
seed reproduces the task · schema hash matches the release · a known cheater
cannot obtain reward.

Plus adversarial arms: `cheat_gold_access`, `cheat_candidate_order`,
`cheat_seed_recovery`, `cheat_constant_action`, `cheat_all_compute`,
`cheat_unverified_write`, `cheat_reward_probe`.

And independent RNG streams — `generator`, `candidate`, `router`, `training`,
`environment`, `evaluation` — never derived indirectly. The pinned-family
collision was caused by coupled randomness (`seed + len(families)`), and is
the reason this is a hard contract rather than a style preference.

### M1.2 — Freeze the current baseline *(was Stage C)*

`TACOSM-BASELINE-001` at `91597ab` remains the reference. No undocumented
modifications. The freeze is what makes every later delta a comparison
against a published number rather than against a re-measured one — F3's
reproduction gate is this stage being used as intended.

### M1.3 — Proper ablation matrix *(was Stage D)*

state · router · executor · verifier · repair · write, under identical
episodes. Then *interactions*, not only one-component removal:
`router × state`, `router × verifier`, `state × verifier`, and the triple —
because mechanisms may only work jointly. Paired comparisons over identical
instances with a paired bootstrap, standard error and a predeclared minimum
detectable effect.

### M1.4 — The measurement contract *(was Stage F4)*

Every number this repository emits belongs to exactly one of five layers, and
the layers are ordered: a Layer 2 or Layer 3 number is uninterpretable unless
the Layer 1 gate for the model that produced it has passed, and no layer is
reached before the question has been asked whether the answer is already
inherited — Layer 0. The full contract is `docs/MEASUREMENT_LAYERS.md`, the
per-row assignment is `docs/EVIDENCE_REGISTER.md`, and the procedure that
makes a proposed experiment confront both before it runs is
`docs/EVIDENCE_MAP.md`.

| | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| asks | is the answer already established, and may it be inherited? | is the measurement about the model at all? | does the mechanism behave as claimed? | does the system solve the task, at what cost? |
| on failure | capital is spent | voids the run | withdraws one mechanism claim | withdraws the system claim |

Both of the repository's expensive errors — the TACOSM-HS-001 untrained-weight
sweep, and the `dd8f63c` key-blind feature map — were Layer 1 failures that
produced plausible Layer 2 numbers read as Layer 2 findings. This milestone is
a contract, not a deliverable: it has no end date because it applies to every
future milestone.

### M1.5 — History scaling *(was Stage E — done, `TACOSM-HS-001`)*

H = 2,4,8,16,32,64 with the relevant problem held fixed. Establishes that the
limitation is routing, and that total cost is `O(H)` in v0.1.

### M1.6 — The retrieval ceiling *(was Stage F, F0 — done, `TACOSM-RETRIEVAL-001`)*

F0 exists because the obvious next step is wrong. Building an index before
establishing whether the scoring representation carries the relevance signal
produces a cheap index around an inadequate representation, which retrieves
the wrong candidates faster and reads as a success.

**The existing router scores all H candidates; retain the top-K and measure
`P(gold ∈ top-K)` across H × K.** No new mechanism is built, so the result is
a property of the current representation.

The answer is mixed and is reported as mixed: the signal survives well at
`H ≤ 64` (at H=64, recall climbs 0.162 → 0.530 → 0.748 → 0.920 across
K = 1, 4, 8, 16) and does **not** saturate well at `H ≥ 128` (rec@16 falls to
0.550 at H=256). Two readings survived the data and F0 did not separate them:
the representation degrades with the population, or the router was trained at
8 candidates and the sweep measures transfer.

**F0's follow-up is run, and it closed the question.**
`TACOSM-MATCHED-001` ran the matched-H design that was supposed to separate
them, and produced **neither** branch of the pre-committed rule. Matched-H
training does not recover the large-H signal — it is the **worst row at every
evaluation population, including its own** — and the representation is *not*
inadequate, because the analytic vector reaches `routing@1 = 1.0000` at
H ∈ {8, 64, 256, 512} with `delta_1 ≈ +3.07`. `basis_size` never depends on
the candidate count, so the relation is H-invariant by construction.

The failure is in the **learning dynamics**: REINFORCE sees 5 successes in 500
steps at H=256 (1% against a 0.39% chance rate), its `(1 − p_selected)`
multiplier is squeezed by a flat softmax over a large candidate set, and
5000 steps leave `routing@1` at 0.0700 while the weight norm grows 12-fold and
the margin *worsens*. A wrong learning rule, not an under-trained model.

### M1.7 — Learning dynamics *(was Stage F2 — done, `TACOSM-LEARN-001`)*

> The architecture is not the problem. The hypothesis class contains a
> perfect, H-invariant solution; the environment provides enough information
> to find it (`oracle = 1.0000`); what fails is the *update*.

The interventions were pinned to named parameters and named schedules *before*
the run, because an improvement from an unspecified intervention is
unfalsifiable — any negative result could be attributed to having picked the
wrong knob. What was a two-item candidate list became a named three-arm
design:

| arm | knob | value | schedule | acts on |
|---|---|---|---|---|
| `baseline` | — | — | — | the MATCHED-001 protocol verbatim, reproduced |
| `epsilon_greedy` | `eps_0` | `0.30` | linear decay to 0 over `T=500` | action selection in `route` |
| `temperature` | `tau_0` → `tau_end` | `2.0` → `0.5` | linear over `T=500` | the training softmax in `_softmax` |

Both interventions are **training-only** — no arm changes the feature basis,
the environment, the candidate generator, the evaluator, or how the router
selects at evaluation (`epsilon = 0`, `temperature = 0.5` always). The
benchmark is frozen: the `baseline` arm must reproduce the published
MATCHED-001 numbers, or the run is invalid and no arm is reported.

The primary endpoint was `routing@1` at `train-H = 256` (the 0.0740 number),
with `recall@4`/`recall@8`/`recall@16`, `gold_rank`, `delta_1`, `entropy`,
`accuracy`, `C_router` and per-seed values as secondary endpoints. The
decision rule was committed in advance with its consequences attached —
including the one that matters most:

> **A negative F2 does not establish that persistent-state relevance routing
> does not scale.** It establishes that these two named interventions do not
> fix the trained router at H=256. The hypothesis class provably contains a
> perfect H-invariant solution, so a mechanism whose hypothesis class is
> adequate cannot be pronounced inadequate by a failure to *find* the
> solution with one optimiser. The distinction — "the current router doesn't
> scale" versus "the routing mechanism doesn't scale" — is exactly what the
> measurement-layer contract exists to enforce.

The surrogate-reward option MATCHED-001 recorded (a dense reward from the
analytic margin, available without gold labels) was **deliberately not part of
F2**. It changes the reward function rather than the exploration regime, so it
is a different experiment with its own leakage boundary to argue; it was
re-queued as M1.8, not merged.

**Result.** The reproduction gate passed — the baseline arm reproduced all
nine published MATCHED-001 numbers exactly, so the comparison is against the
published reference. The primary endpoint did not fire: `routing@1` at H=256
moved -0.0220 (`epsilon_greedy`) and -0.0140 (`temperature`) against a 0.0600
materiality threshold, i.e. within seed noise. The top-K signal was not
intact either — `recall@16` at H=256 fell to 0.2260 from 0.3020 under
`epsilon_greedy`, which is material *harm*.

Neither M2 condition is satisfied from this result, so M2 does not re-queue
from it. The commitment that matters is the one recorded above and in the
pre-registration's "what a negative result does not license" section: this is
*not* a representation verdict and does not scale the mechanism down. It
locates the failure outside the *selection* stage of training —
`epsilon_greedy` explored as registered (15.2% of steps) and saw more
successes per step at H=256 (3.8 vs 2.4, chance 0.0039), and the endpoint
still did not improve. The remaining candidates are the ones both arms
deliberately left alone: the learning rule, the gradient signal, and the
reward shaping. The surrogate-reward option below is where those live.

This milestone also decides C5's precondition. `C_executed ≈ f(|R|)` needs a
retrieval boundary to make `|R|` a measurable quantity, and the only retrieval
boundary worth building is one on a scorer whose numbers mean what they say.

### M1.8 — Reward density *(was Stage F3 — done, `TACOSM-SURROGATE-001`)*

M1.7 closed two named exploration interventions without moving the endpoint,
and it closed them in the way that matters: `epsilon_greedy` explored as
registered (15.2% of steps) and saw **more** successes per step at H=256
(3.8 vs 2.4, chance 0.0039) — the exact condition the scarcity hypothesis
predicts — and `routing@1` still did not improve. Selection is not the
binding constraint.

What both arms deliberately left alone is the reward itself, and that is the
one remaining component of the update that is still binary and terminal.
M1.8 intervenes on it, and only it:

| arm | reward | centring | clip | exploration |
|---|---|---|---|---|
| `baseline` | `float(outcome.success)` | `1/H` | — | none |
| `analytic_margin` | `s_gold − max_{j≠gold} s_j` | per-step mean across candidates | none | none |
| `analytic_margin_clipped` | as above | as above | `[0, 1]` | none |

**The load-bearing part of the pre-registration is the leakage-boundary
argument:** the surrogate is computed from the router's own parameters and its
own permitted inputs, and the gold index appears only to anchor the margin —
the training-time analogue of what the §34 representability gate already
licenses for measurement (`gold_fn` "is never an input to the mechanism"). A
positive result licenses "the hypothesis class is learnable under dense
supervision", **not** "the router learned relevance from outcomes", and is
reported as the weaker claim.

M1.8 is the experiment that separates the two hypotheses M1.7 left live,
because they are not separable by observation: both credit-assignment and
reward-sparsity predict "more useful interactions, no endpoint movement",
which is what M1.7 measured. A dense reward makes them differ on the 99% of
H=256 steps that currently carry no gradient signal at all.

**Result — the asymmetry is the finding.** The reproduction gate passed (nine
cells, `diff = 0.0000` against the published MATCHED-001 baseline). At
`train-H = 256`:

| endpoint | baseline | `analytic_margin` | `analytic_margin_clipped` |
|---|---|---|---|
| `routing@1` | 0.0740 | 0.1140 | 0.0980 |
| `recall@16` | 0.3020 | **0.3880** | **0.5480** |
| `gold_rank` | 74.6960 | 48.0900 | **25.8260** |
| successes / 500 | 2.4 | 3.8 | 2.0 |

The primary endpoint did **not** fire: `Δ(routing@1)` = +0.0400 and +0.0240
against the same 0.0600 materiality threshold, within seed noise. The
secondary endpoints did: `Δ(recall@16)` = +0.0860 and +0.2460, both material,
with `gold_rank` improving by a factor of three and the reward's non-zero rate
going from 0.0048 to 0.9980 at H=256. Gold moved closer to the top of the
ranking without separating from the single best distractor — `delta_1` at
H=256 does not move (−0.4383 → −0.4749). **The gain is in the bulk of the
score distribution, not at the top.**

**A negative M1.8 is a stronger result than M1.7's.** Two named interventions
on selection and one on reward, all failing to move the same top-1 number,
constrain the *location* of the failure rather than listing things that do not
work. That reopens the architecture question on firmer ground than "the current
router doesn't work" — the inspection order is committed to in advance and is
not reordered to fit the outcome.

**What this milestone licenses, and what it does not.** The row that fired is
recorded in the register as *the large-H top-K signal is responsive to
training* (L2) and as claim **C11** (L3, bounded at `K ≥ 2`). Neither is a
statement about top-1, neither resolves the learning problem, and neither says
the router learned relevance — the surrogate is gold-anchored exactly as the
outcome reward is. What the pair *does* do is give M2 a registered reason to
exist, which is the next milestone.

### M1.9 — Where M1 stands, and the one open question it leaves

M1 has produced a scorer that provably *can* represent the relevance relation
at every H tested, that provably *learns* it at small populations from outcome
supervision alone, and whose large-H failure has been localised by
elimination:

```
TACOSM-MATCHED-001  →  the representation is adequate
TACOSM-LEARN-001    →  selection is not the binding constraint
TACOSM-SURROGATE-001 →  reward density improves top-K, not top-1
        therefore   →  the learning signal matters for retrieval quality,
                        but exact top-1 ranking at large H remains unresolved
```

The remaining M1 question is the one the sequence leaves open by construction:
**is exact top-1 routing at large H a property this hypothesis class can be
trained into, or is it a property of the update?** The hypothesis class
provably contains a perfect H-invariant solution, so the question is not
whether it is expressible — the §34 gate settles that. It is whether a
policy-gradient update over a single scalar reaches it. Every intervention
tried closes a *location* without closing the *question*, which is the
signature of a problem that is about the optimiser rather than the space.

M1 is not blocked on this answer. The registered top-K evidence is sufficient
to proceed to M2, and proceeding is what the decision rule committed to. What
M1 does **not** hand to M2 is a scorer whose top-1 number means what it says
at large H, and M2's design must not assume one.

## Implementation order — M2: Selective computation

M2 asks whether the system computes only over the relevant subset. It is
entered on the registered top-K evidence M1.8 produced, and *only* on that
evidence — not on a top-1 improvement that did not happen.

### M2.1 — Efficient retrieval *(was Stage F1 — re-queued)*

    H → cheap index → K ≪ H → existing scorer → R

Changes `C_router` from `O(H)` to `O(K)` while reusing the existing scorer
unchanged. Measured against M1.6 at matched K, because the claim is a
cost-quality trade rather than "the index works". The four arms
(A exhaustive+top-1, B exhaustive+top-K, C indexed+top-K, D oracle-index+top-K)
and the failure taxonomy are pre-registered in `TACOSM-RETRIEVAL-001.md`.

This is the milestone that can test the efficiency hypothesis, because it
inserts a retrieval boundary the v0.1 loop does not have. It is also the
milestone that makes `|R|` a defined quantity, which is what C5 and all of M3
are waiting on.

**Why it was suspended, and why it re-queues now.** Its hypothesis is a
cost-quality trade: can routing computation be reduced while preserving the
relevant candidate? The matched-H result said the quality half of that trade
is not a property of the representation, so an index built then would have
been measured against a baseline whose weakness was a training artefact — a
bad baseline makes a mediocre index look good, and a good index look
unnecessary. That is not a sound measurement in either direction, and the
specification was kept intact and un-amended because it was pre-registered and
a pre-registration is not revised to fit a result. It is re-queued, not
withdrawn.

**The evidence that re-queues it is M1.8's, and it is registered as what it
is.** Before M1.8 the re-queue condition was "the top-K signal is shown
intact", and the only evidence for it was M1.6's transfer-trained measurement
— the measurement M2.1 was designed not to trust. After M1.8 the condition is
met by a row that is *responsive to training* rather than merely present on a
transfer-trained router: `Δ(recall@16)` = +0.0860 / +0.2460 at H=256 under a
training-only intervention. That is a different and stronger thing, and it is
why M2.1's baseline is the M1.8 `analytic_margin_clipped` arm rather than the
M1.6 transfer row.

**The bound travels with the evidence.** M1.8 moved `recall@16` and did not
move `routing@1`, so M2.1 is a cost-quality trade at `K ≥ 2`, not at `K = 1`.
An index that preserves top-K recall at `K ≪ H` is a positive result; an index
that preserves top-1 is not what the evidence supports, and a run that
reported top-1 preservation as its headline would be reporting a number M1.8
says is not there.

**What this milestone is not.** M2.1 is a cost experiment that assumes the
scorer, not a router experiment that happens to add an index. Three drifts
are tempting and all three are out of scope:

1. **It is not "try another router."** M1.8's asymmetry makes a stronger
   scorer look like the obvious next move, and M2.1 must not absorb it. A
   better scorer is an M1 result, measured under M1's protocol, and would
   change the baseline M2.1 is defined against — which is the exact
   ambiguity the re-queue condition exists to prevent. If a scorer change is
   worth running it gets its own pre-registration and its own milestone, and
   M2.1 waits for it rather than smuggling it in as an index component.
2. **It is not "fix the top-1 endpoint."** That question is M1.9's, and it is
   open. M2.1's registered evidence is top-K, and a run designed to close the
   top-1 gap would be measuring a number M1 has already said is not there.
3. **It is not a re-measurement of M1.6.** The exhaustive arms exist as the
   cost-quality reference, and every indexed-arm number is a delta against
   them at matched K — not a standalone retrieval-quality table, which the
   repository already has.

**The failure mode this guardrail exists for.** An experiment that re-enters
M1 territory under an M2 label produces a correct number attached to the wrong
claim, and the register cannot see it, because the rows it would create are
genuinely new — they are just new for the wrong milestone. The pre-experiment
gate in `docs/EVIDENCE_MAP.md` is what catches this: it asks *what claim does
this support*, and a claim about scorer quality is not a claim about
computation.

### M2.2 — State formation *(was Stage G)*

`StateProposal → verify → commit`, with `ADD / UPDATE / INVALIDATE / DELETE /
NOOP`. A proposal is not state until verification passes. Then compare
no-write, oracle-write, fixed-rule write, learned write, and learned write
under verification — measuring `future_task_gain`, `memory_reuse`,
`wrong_write_rate`, `retraction_rate`, `stale_state_rate`.

### M2.3 — Repair *(was Stage H)*

`P(recover | initial failure)`, cost per recovery, false-repair probability,
verification overhead.

## Implementation order — M3: Economic scaling

M3 asks whether the capability remains economical as history grows. It is the
last milestone because it is a *curve*, and a curve needs both axes present.

### M3.1 — The capability-vs-computation curve *(was Stage I's predecessor)*

The milestone is not "improve the learned model". It is:

> Make TAC-OSM capable of producing an independently reproducible,
> statistically defensible capability-vs-cost curve under a frozen benchmark.

The shape of that evidence:

| H | 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| full context | 96% | 95% | 94% | 91% | 86% |
| linear router | 96% | 94% | 91% | 82% | 70% |
| indexed router | 96% | 95% | 94% | 94% | 93% |

with total compute and relevant compute measured **separately** along the same
axis, not one derived from the other. These numbers are illustrative, not
results — they are the shape of evidence the system should be able to
generate. The measurement-layer contract forbids presenting `C_executed`
against `|R|` before a retrieval boundary exists, which is exactly why M2
precedes this.

### M3.2 — Language integration *(was Stage I)*

Only now a small LM backbone — after the curve exists, so the backbone enters
a system whose economical-scaling claim is already measured rather than
assumed.

### CDL's role

CDL is **not** runtime inference. It is a relevance teacher, a difficult
ranking benchmark, and a diagnostic upper bound. The question is whether a
cheap learned router recovers enough of its discrimination without inheriting
its per-candidate cost.

## Deferred

large Transformer rewrite · replacing attention · huge neural memory ·
complicated MoE routing · CDL as runtime inference · dozens of configuration
knobs · another isolated substrate repo · optimising only router accuracy ·
optimising only training loss.
