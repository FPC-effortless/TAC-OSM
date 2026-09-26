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

### Stage F — Retrieval and indexing  *(F0 done; F1 not started)*

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
0.550 at H=256). Two readings survive the data and F0 does not separate them:
the representation degrades with the population, or the router was trained at
8 candidates and the sweep measures transfer. **The follow-up that separates
them — train at matched H, re-run F0 — is the next thing to run**, because it
decides whether F1 is worth building.

**F1 — Efficient retrieval** *(not started; contingent on F0)*

    H → cheap index → K ≪ H → existing scorer → R

Changes `C_router` from `O(H)` to `O(K)` while reusing the existing scorer
unchanged. Measured against F0 at matched K, because the claim is a cost-quality
trade rather than "the index works". The four arms (A exhaustive+top-1,
B exhaustive+top-K, C indexed+top-K, D oracle-index+top-K) and the failure
taxonomy are pre-registered in `TACOSM-RETRIEVAL-001.md`.

This is the stage that can test the efficiency hypothesis, because it inserts
a retrieval boundary the v0.1 loop does not have.

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
