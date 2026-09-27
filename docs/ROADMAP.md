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

## Implementation order

### Stage A — Research kernel (no new capability)

`RunManifest`, `EpisodeSpec`, `Trajectory`, `EpisodeResult`,
`BenchmarkManifest`, `Evaluator`, `Statistics`. Every experiment becomes
reconstructable.

### Stage B — Benchmark hardening

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

### Stage C — Freeze the current baseline

`TACOSM-BASELINE-001` at `91597ab` remains the reference. No undocumented
modifications.

### Stage D — Proper ablation matrix

state · router · executor · verifier · repair · write, under identical
episodes. Then *interactions*, not only one-component removal:
`router × state`, `router × verifier`, `state × verifier`, and the triple —
because mechanisms may only work jointly. Paired comparisons over identical
instances with a paired bootstrap, standard error and a predeclared minimum
detectable effect.

### Stage E — History scaling  *(done — TACOSM-HS-001)*

H = 2,4,8,16,32,64 with the relevant problem held fixed. Establishes that the
limitation is routing, and that total cost is `O(H)` in v0.1.

### Stage F — Retrieval and indexing  *(F0 done; F1 **suspended**)*

F0 exists because the obvious next step is wrong. Building an index before
establishing whether the scoring representation carries the relevance signal
produces a cheap index around an inadequate representation, which retrieves
the wrong candidates faster and reads as a success.

**F0 — Retrieval ceiling** *(done — `TACOSM-RETRIEVAL-001`)*

The existing router scores all H candidates; retain the top-K and measure
`P(gold ∈ top-K)` across H × K. No new mechanism is built, so the result is a
property of the current representation.

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

**F1 is suspended, not abandoned.** F1's hypothesis is a *cost-quality trade*:
can routing computation be reduced while preserving the relevant candidate?
The matched-H result says the quality half of that trade is not a property of
the representation, so an index built now would be measured against a baseline
whose weakness is a training artefact. That is not a sound measurement of an
index in either direction — a bad baseline makes a mediocre index look good.
The specification below is kept intact and un-amended, because it was
pre-registered and a pre-registration is not revised to fit a result. It is
re-queued, not withdrawn.

### Stage F2 — Learning dynamics  *(pre-registered; RUN — result: neither arm improves the endpoint)*

> The architecture is not the problem. The hypothesis class contains a
> perfect, H-invariant solution; the environment provides enough information
> to find it (`oracle = 1.0000`); what fails is the *update*.

**Pre-registered in `docs/TACOSM-LEARN-001.md`.** The interventions are pinned
to named parameters and named schedules *before* the run, because an
improvement from an unspecified intervention is unfalsifiable — any negative
result could be attributed to having picked the wrong knob. What was a
two-item candidate list is now a named three-arm design:

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

The primary endpoint is `routing@1` at `train-H = 256` (the 0.0740 number),
with `recall@4`/`recall@8`/`recall@16`, `gold_rank`, `delta_1`, `entropy`,
`accuracy`, `C_router` and per-seed values as secondary endpoints. The
decision rule is committed in advance with its consequences attached —
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
analytic margin, available without gold labels) is **deliberately not part of
F2**. It changes the reward function rather than the exploration regime, so
it is a different experiment with its own leakage boundary to argue; it is
re-queued, not merged.

**F1 re-queues only on the primary endpoint firing, or on the top-K signal
being shown intact** — and those are different evidence, recorded as such.

**F2 result (recorded in `docs/TACOSM-LEARN-001.md` under Outcomes).** The
reproduction gate passed — the baseline arm reproduced all nine published
MATCHED-001 numbers exactly, so the comparison is against the published
reference. The primary endpoint did not fire: `routing@1` at H=256 moved
-0.0220 (`epsilon_greedy`) and -0.0140 (`temperature`) against a 0.0600
materiality threshold, i.e. within seed noise. The top-K signal was not
intact either — `recall@16` at H=256 fell to 0.2260 from 0.3020 under
`epsilon_greedy`, which is material *harm*.

Neither F1 condition is satisfied, so F1 does not re-queue from this result.
The commitment that matters is the one recorded above and in the
pre-registration's "what a negative result does not license" section: this is
*not* a representation verdict and does not scale the mechanism down. It
locates the failure outside the *selection* stage of training —
`epsilon_greedy` explored as registered (15.2% of steps) and saw more
successes per step at H=256 (3.8 vs 2.4, chance 0.0039), and the endpoint
still did not improve. The remaining candidates are the ones both arms
deliberately left alone: the learning rule, the gradient signal, and the
reward shaping. The surrogate-reward option below is where those live.

This stage also decides C5's precondition. `C_executed ≈ f(|R|)` needs a
retrieval boundary to make `|R|` a measurable quantity, and the only retrieval
boundary worth building is one on a scorer whose numbers mean what they say.

**F1 — Efficient retrieval** *(suspended; contingent on F2)*

    H → cheap index → K ≪ H → existing scorer → R

Changes `C_router` from `O(H)` to `O(K)` while reusing the existing scorer
unchanged. Measured against F0 at matched K, because the claim is a cost-quality
trade rather than "the index works". The four arms (A exhaustive+top-1,
B exhaustive+top-K, C indexed+top-K, D oracle-index+top-K) and the failure
taxonomy are pre-registered in `TACOSM-RETRIEVAL-001.md`.

This is the stage that can test the efficiency hypothesis, because it inserts
a retrieval boundary the v0.1 loop does not have.

### Stage F3 — Measurement layers  *(the standing contract)*

Every number this repository emits belongs to exactly one of three layers, and
the layers are ordered: a Layer 2 or Layer 3 number is uninterpretable unless
the Layer 1 gate for the model that produced it has passed.

| | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| asks | is this measurement about the model at all? | does the mechanism behave as claimed? | does the system solve the task, at what cost? |
| on failure | voids the run | withdraws one mechanism claim | withdraws the system claim |

Both of the repository's expensive errors — the TACOSM-HS-001 untrained-weight
sweep, and the `dd8f63c` key-blind feature map — were Layer 1 failures that
produced plausible Layer 2 numbers read as Layer 2 findings. See
`docs/MEASUREMENT_LAYERS.md`. This stage is a contract, not a deliverable: it
has no end date because it applies to every future stage.

### Stage G — State formation

`StateProposal → verify → commit`, with `ADD / UPDATE / INVALIDATE / DELETE /
NOOP`. A proposal is not state until verification passes. Then compare
no-write, oracle-write, fixed-rule write, learned write, and learned write
under verification — measuring `future_task_gain`, `memory_reuse`,
`wrong_write_rate`, `retraction_rate`, `stale_state_rate`.

### Stage H — Repair

`P(recover | initial failure)`, cost per recovery, false-repair probability,
verification overhead.

### Stage I — Language integration

Only now a small LM backbone.

### CDL's role

CDL is **not** runtime inference. It is a relevance teacher, a difficult
ranking benchmark, and a diagnostic upper bound. The question is whether a
cheap learned router recovers enough of its discrimination without inheriting
its per-candidate cost.

## The measurement that changes the claim

The milestone is not "improve the learned model". It is:

> Make TAC-OSM capable of producing an independently reproducible,
> statistically defensible capability-vs-cost curve under a frozen benchmark.

The shape of that evidence:

| H | 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| full context | 96% | 95% | 94% | 91% | 86% |
| linear router | 96% | 94% | 91% | 82% | 70% |
| indexed router | 96% | 95% | 94% | 94% | 93% |

with total compute and relevant compute measured separately along the same
axis. These numbers are illustrative, not results — they are the shape of
evidence the system should be able to generate.

## Deferred

large Transformer rewrite · replacing attention · huge neural memory ·
complicated MoE routing · CDL as runtime inference · dozens of configuration
knobs · another isolated substrate repo · optimising only router accuracy ·
optimising only training loss.
