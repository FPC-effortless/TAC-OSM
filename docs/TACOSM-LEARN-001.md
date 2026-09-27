# TACOSM-LEARN-001 (F2) — the learning rule, pre-registered

**Status: pre-registered.** This document is written *before* the experiment
runs. The decision rule, the interventions, the frozen protocol and the
failure taxonomy are all fixed here, and they are not amended whatever the
result turns out to be. A pre-registration is not revised to fit a result;
the row that occurs is recorded, not substituted (the convention
`TACOSM-MATCHED-001` established when its own two branches both failed).

It supersedes nothing. It follows `TACOSM-MATCHED-001`, which closed F0's open
question and left this one open:

> **C — the learning rule, not the representation or the training population.**
> The hypothesis class contains a perfect, H-invariant solution. The
> environment provides enough information to find it — `oracle = 1.0000`. What
> fails is the *update*, whose reward signal is too sparse to estimate a
> gradient from at large H.

This experiment tests that, and only that.

---

## The claim under test, stated as a question

> **Does a specific, named training-time intervention improve learned
> relevance discrimination at large candidate-population size — measured
> against baseline training under the same task distribution, the same
> evaluation protocol and the same frozen baseline?**

Not "can we make H=256 work better". The difference matters: an improvement
from an unspecified intervention is unfalsifiable, because any negative result
can be attributed to having picked the wrong knob. So every intervention below
is pinned to a concrete parameter, a concrete schedule and a concrete place in
the loop before the run.

---

## Why exploration is the hypothesis

REINFORCE's update, from `router.py::LearnedRelationalRouter.update`:

```
grad = lr * (reward - chance) * (1 - p_selected) * features
```

At `H = 256` the measured facts from MATCHED-001 are:

| quantity | H=8 | H=64 | H=256 |
|---|---|---|---|
| successes / 500 steps | 248 | 18 | **5** |
| success rate | 49.6% | 3.6% | **1.0%** |
| chance | 12.5% | 1.56% | 0.39% |
| success/chance | 3.97 | 2.30 | 2.56 |
| `routing@1` after 500 steps | 0.6240 | 0.1320 | **0.0640** |

The multiplier `(1 - p_selected)` is a *confidence* term: it vanishes as the
policy sharpens. At a large H with a near-uniform softmax, `p_selected ≈ 1/H`
for every candidate, so the multiplier is near 1 on every step — the update is
*large* when it fires. It just almost never fires, and when it does the single
rare success moves the weights by a step scaled to a 1/256 event. Two failure
modes follow, and both are exploration pathologies rather than representation
pathologies:

1. **The policy never sharpens, so successes stay at chance rate.** At 1%
   success against 0.39% chance the signal-to-noise on 500 steps is roughly
   5 successes versus ~2 expected by chance. That is not a gradient estimate;
   it is a handful of coin flips.
2. **The weights grow without learning the relation.** 5000 steps take
   `‖w‖` from 0.591 to 6.880 (12×) while `routing@1` moves 0.0600 → 0.0700 and
   `delta_1` *worsens* from −0.7758 to −6.3544. The norm is growing on noise.

Exploration addresses both: more trials per step means more successes per
step, and a temperature that *anneals* means the policy is allowed to sharpen
only once it has something to sharpen.

---

## The frozen protocol — what does not change

This is the "do not alter the benchmark" rule, made explicit. Every arm below
runs under all of these, identically:

| Frozen | Value | Source |
|---|---|---|
| task families | `relational`, `state_lookup`, `replay`, in that order, cycling | `WorldConfig.families` |
| `dim` | 8 | `WorldConfig.dim` |
| noise | 0.10 | `WorldConfig.noise` |
| train H | 8, 64, 256 | the MATCHED-001 levels, unchanged |
| eval H | 8, 64, 256 | matched to the training levels |
| training steps | 500 | MATCHED-001, so the baseline column reproduces its numbers |
| eval steps | 100 per cell | MATCHED-001 |
| seeds | 0–4, train and eval matched within a cell | MATCHED-001 |
| evaluation router temperature | 0.5 | `RouterConfig.temperature`, unchanged by any arm |
| evaluation protocol | `router.score` raw dot products; argmax for `routing@1` | MATCHED-001 |
| feature basis | the six blocks, `analytic_weights` unchanged | §34 gate |
| scoring at evaluation | sampling disabled — argmax over raw scores | F0 / MATCHED-001 |
| integrity gate | unconditional, before every training and evaluation cell | C4 |
| oracle control | printed first, per eval H; must be 1.0000 | C1 |

**The baseline is not re-measured, it is reproduced.** The `baseline` arm is
exactly the MATCHED-001 protocol. If its column does not reproduce the
published MATCHED-001 numbers, the run is invalid and no arm is reported — the
frozen baseline is the reference, and a run that cannot reach it has changed
something it did not declare.

**Evaluation is identical across arms.** Only training differs. No arm changes
`features`, `basis_size`, the candidate generator, the evaluator, the metric
definitions or the noise model. An arm that needs any of those to change is a
different experiment and is rejected here, not added.

### What exploration is *not*

It is not a change to the feature basis, the hypothesis class, the
environment, the candidate generator or the evaluation protocol. It is not a
bigger model. It does not add a term to the reward that depends on the gold
index. And it is **training-only**: no arm may alter how the router selects at
evaluation time. `epsilon = 0` and `temperature = 0.5` at evaluation, always.

---

## The interventions — pinned before the run

Both are defined as concrete parameters with concrete schedules. Neither is a
free variable.

### Arm 2: epsilon-greedy selection

**What constitutes exploration.** With probability `epsilon`, the selected
candidate is drawn uniformly at random instead of sampled from the policy's
softmax. With probability `1 - epsilon`, the policy's own sample is used
unchanged.

**Where it acts.** In `LearnedRelationalRouter.route`, between the softmax and
the categorical sample — i.e. it replaces the *action* that becomes
`decision.selected`. The REINFORCE update in `update()` is unchanged and still
receives `probs[selected]`, the policy's probability of the action actually
taken. It is a training-time action-selection change; nothing about the update
rule changes.

**Training only.** At evaluation `epsilon = 0`. `RoutingDecision.provenance`
records the arm, and the evaluator uses `router.score` argmax, which does not
sample at all.

**Schedule.** `epsilon_t = eps_0 * (1 - t / T)`, with `eps_0 = 0.30` and
`T = 500` (the training length), decaying linearly to 0. So exploration is
front-loaded where successes are rarest, and the final policy is greedy.

**`lambda`.** Not applicable — this arm has no loss term. Its single knob is
`eps_0 = 0.30`, fixed here.

### Arm 3: temperature schedule

**What constitutes exploration.** The softmax temperature *during training* is
raised above its evaluation value of 0.5, then annealed back down to it. A
higher temperature flattens the policy, which raises the probability of
selecting non-greedy candidates and therefore the number of successes seen per
step.

**Where it acts.** In `LearnedRelationalRouter._softmax`, on the training
sample only. The scale of the raw scores is not changed — the features and
weights are untouched — only the sampling distribution. The update is
unchanged.

**Training only.** Evaluation temperature stays at 0.5, and the evaluator
computes `routing@1` from `router.score` argmax, which never passes through
the softmax.

**Schedule.** `tau_t = tau_0 * (1 - t / T) + tau_end * (t / T)`, linear from
`tau_0 = 2.0` to `tau_end = 0.5`, `T = 500`. The endpoint is deliberately the
evaluation temperature, so the trained policy ends in the regime it is scored
in.

**`lambda`.** Not applicable. Its knobs are `tau_0 = 2.0` and
`tau_end = 0.5`, fixed here.

### Arm 1: baseline

The MATCHED-001 protocol verbatim: `epsilon = 0`, constant training temperature
0.5, REINFORCE as written. This is the reference column, and it exists so that
any change in an exploration arm is attributable to the named knob and not to
a difference in task stream, seed handling or evaluation.

### Knobs, all in one place

| Arm | knob | value | schedule | acts on | training only? |
|---|---|---|---|---|---|
| `baseline` | — | — | — | — | — |
| `epsilon_greedy` | `eps_0` | `0.30` | linear decay to 0 over `T=500` | action selection in `route`, between softmax and categorical sample | yes; `epsilon = 0` at evaluation |
| `temperature` | `tau_0` → `tau_end` | `2.0` → `0.5` | linear over `T=500` | the training softmax scale in `_softmax` | yes; `0.5` at evaluation |

Every value in this table is fixed by this pre-registration. **None is exposed
as a command-line argument** — see "Reproduction".

The `baseline` arm is what the other two are measured against, and it is the
MATCHED-001 protocol verbatim, so the comparison is against a published
reference rather than a re-measured one.

---

## The decision rule — committed in advance

For each train-H, compare each exploration arm to the baseline at the **same
H, same seeds, same evaluation**. Let `Δ` be the exploration arm's value minus
the baseline's, so `Δ > 0` is an improvement. A difference counts as material
only if it exceeds the seed spread, defined as
`spread = max(arm) - min(arm)` across that metric's values at that H over the
five seeds — the same standard MATCHED-001 used, and chosen because it is
measurable before any arm is compared.

### Primary endpoint: `routing@1` at `train-H = 256`

This is the number MATCHED-001 left at 0.0740, and the one the scarcity
hypothesis predicts exploration can move.

| Outcome | Interpretation | Consequence committed to now |
|---|---|---|
| `Δ(routing@1) > spread`, material | exploration lifts argmax routing at large H | **The failure is at least partly a training-dynamics problem.** Record the mechanism, then re-queue **F1** (the retrieval index) on a router whose quality is a property of the scorer rather than of the optimiser. |
| `Δ(routing@1)` within spread | exploration does not fix argmax routing at large H | **The failure is not solely exploration.** Fall through to the secondary endpoints before concluding anything. This is *not* a representation verdict. |
| `Δ(routing@1) < -spread` | exploration actively hurts | The intervention is harmful; record and stop. No further arm is added from this doc. |

### Secondary endpoints, in order

1. **`recall@4`, `recall@8`, `recall@16`** — the shortlist signal. The case
   the decision rule specifically allows for: exploration fails to move
   `routing@1` while top-K recall remains useful. Then the top-K signal is
   intact and the retrieval architecture is still the right next step — F1
   re-queues even though the primary endpoint did not fire, *and* the
   re-queueing reason is recorded as "top-K signal intact", not as
   "`routing@1` improved". Those are different evidence.
2. **`gold_rank`** — mean rank of gold. Detects whether the field moved past
   gold without gold moving at all, which is a population effect, not a
   learning effect.
3. **`delta_1 = s_gold − s_best_distractor`** — the separation. This is the
   diagnostic that separated readings A and B in F0. If exploration widens
   `delta_1` at H=256 without lifting `routing@1`, the scorer is
   *discriminating better* while the best distractor still wins — a
   population-discrimination problem, not a training problem.
4. **`entropy`** against `ln H` — how concentrated the router's distribution
   is. Expected to fall in an exploration arm that works, because a policy
   with a working signal sharpens. Distinguishes "exploration made the policy
   flatter without making it better" (entropy falls, recall does not rise)
   from "exploration found the signal" (entropy falls, recall rises).
5. **`accuracy`** — the system consequence, from the executed loop. The metric
   a user cares about; included so a routing improvement can be seen to
   propagate (or not) to the outcome.
6. **`C_router`** — actual routing cost, reported as the candidate count
   scored. Constant across arms by construction (`O(H)` for every arm,
   including exploration). **Reported but never the basis of a claim**: F2 is
   a quality experiment, not a cost experiment, and `C(|R|)` is an `M2.1`
   quantity that does not exist yet.
7. **`prob_margin`** — the gap in the softmax units the decision samples from.
   Reported for the same reason as in MATCHED-001: the loop is governed by the
   softmax, so the decision-relevant margin is the probability margin even
   though the cross-H-comparable one is `delta_1`.
8. **Seed variance** — per-seed values for every metric, not only the mean.
   The user's `spread` test runs on these, and a 5-seed mean that hides a
   4-seed collapse and a 1-seed success is not a result.

### The interactions that matter

`routing@1` and `recall@16` are the pair. A material `Δ(routing@1)` **and** a
material `Δ(recall@16)` together is the strong outcome: the scorer got better
at every budget. A material `Δ(routing@16)` with flat `routing@1` is the weak
outcome and re-queues F1 on top-K evidence alone. **Flat on both** is the
negative result, and it is reported as a negative result — not as evidence
that the routing premise fails, for the reason in the next section.

---

## What a negative result does *not* license

This is the most important paragraph in the document, and it is written before
the run for that reason.

A negative F2 does **not** establish that persistent-state relevance routing
does not scale. It establishes that two specific named interventions —
epsilon-greedy with `eps_0 = 0.30` linearly decayed, and a temperature
schedule from 2.0 to 0.5 — do not fix the trained router at H=256 under this
protocol. Those are different claims of very different strength, and
conflating them is exactly the error the measurement-layer contract exists to
prevent.

The distinction the contract enforces:

- **"the current router doesn't scale"** — a claim about one trained linear
  scorer under one update rule. F2 can address this.
- **"persistent-state relevance routing doesn't scale"** — a claim about the
  mechanism. F2 cannot address this, and neither can any single intervention
  experiment, because the hypothesis class provably contains a perfect
  H-invariant solution (`analytic_weights`, `routing@1 = 1.0000` at
  H ∈ {8, 64, 256, 512}). A mechanism whose hypothesis class is adequate
  cannot be pronounced inadequate by a failure to *find* the solution with
  one optimiser.

On a negative result the inspection order, committed to now, is:

```
representation  →  training dynamics  →  population discrimination  →  retrieval architecture
```

1. **Representation** — already addressed, and already negative: the §34 gate
   passes and the analytic vector is perfect at every H. Re-checking is cheap
   and remains a Layer 1 obligation, but this is not where a negative F2
   points.
2. **Training dynamics** — F2 is this step. A negative result closes *these
   two interventions*, not the category. A different update rule (a surrogate
   reward from the analytic margin, the option MATCHED-001 recorded, or a
   policy-gradient baseline, or off-policy replay of the 5 successes) is a
   different experiment and is not licensed or excluded by F2's result.
3. **Population discrimination** — `gold_rank` and `delta_1` are what
   addresses it, and they are in the secondary endpoints precisely so that a
   negative primary endpoint does not skip this step.
4. **Retrieval architecture** — F1. Last, because it is downstream of the
   others and a cost-quality trade is only meaningful once the quality term
   is a property of the scorer.

**If F2 fails, F1 is not re-queued.** The order above puts F1 last for a
reason, and the reason is that a cost-quality trade measured against a
baseline whose quality is an optimiser artefact is unsound in both directions.
F1 re-queues on the primary endpoint firing, or on the top-K signal being
shown intact — and *only* on those.

---

## Layer 1 obligations — the gate runs here too

The contract in `docs/MEASUREMENT_LAYERS.md` applies to this experiment in
full, and the exploration arms create one new Layer 1 hazard the design has to
close:

**Warm-start confusion.** The baseline and the exploration arms both start
from `w = [0]*n`. If an exploration arm is implemented by *continuing* from a
baseline-trained router rather than retraining from zero, its numbers measure
"baseline plus more training", not "exploration". The two arms must be
independently trained from initialisation, from the same seed, consuming the
same task stream. `assert_trained` only refuses the all-zero vector — it
deliberately does not compare two trained checkpoints, because that would be a
modelling judgement the gate is not designed to make (see its own docstring).
So this is a design obligation, not a gate: the script builds every arm from
`AblationConfig` through `build_model`, exactly as MATCHED-001 did, and the
only difference between arms is the named knob.

Everything else is inherited:

| Layer 1 check | How it is enforced here |
|---|---|
| weights are not the initialisation | `assert_trained` before every training and evaluation cell, unconditional |
| checkpoint corresponds to the arm under test | `snapshot_router` → `load_weights`, hash in the manifest |
| feature basis represents the relation | §34 via `builder.check_representability`, at construction |
| gold is the unique satisfier, and not exposed to the router | generator integrity tests + `leakage.py` |
| oracle = 1.0000 | printed **first**, per eval H, before any arm table |
| the frozen baseline is reproduced | the `baseline` arm must match the published MATCHED-001 numbers |
| seeds reproduce the task | `WorldEnvironment` is deterministic given the seed |
| benchmark version pinned | the frozen protocol table above |

**The gate's scope is unchanged.** F2 adds no Layer 1 check and weakens none.
The integrity gate is deliberately narrow — it refuses only the all-zero
vector, because that is the only check with a threshold that is not a
modelling judgement. Warm-starting two trained checkpoints is *not* caught by
it, by design, and is therefore closed by construction instead.

---

## How to verify the protocol was followed

Pre-registration is only as strong as its audit trail. The run records:

1. The frozen protocol table from this document, printed at the head of the
   output — so a reader can confirm the run used the values it claims, and
   does not have to trust this document alone.
2. Per-arm knob values: `epsilon` and `temperature` at the first and last
   training steps, proving the schedule ran as specified and that evaluation
   used `epsilon = 0`, `temperature = 0.5`.
3. The integrity checkpoint for every arm: `parameter_hash`,
   `parameter_norm`, `n_updates`, `weight_signature`, `source`. The
   `weight_signature` is the block means — `gated_agreement`, `slot_gated`,
   `query_agreement` — so a reader can see *what rule* each arm learned, and
   whether an exploration arm that improved `routing@1` did so by finding the
   analytic structure or by something else.
4. Success counts per arm per H during training, so the mechanism the
   hypothesis predicts is visible directly: an exploration arm that works
   should show more successes per step, and the improvement in `routing@1`
   should track that count.
5. Per-seed values for every metric, not only the mean.
6. Oracle accuracy per eval H, printed before any arm table.
7. **The `argmax@1` audit column.** Not a registered endpoint, and never part
   of the decision rule. `routing@1` is MATCHED-001's `raw[gold] == max(raw)`,
   in which a tie for first counts as a hit — the convention HS-001 and F0
   also use, so the matrices are comparable. `LearnedRelationalRouter.evaluate`
   breaks ties toward the last tied index instead, so a strict reading of the
   same scores counts a tie as a miss. Both are printed: where they differ,
   the difference is ties, and a reader can see whether a tie rule or a
   learned ranking is doing the work.

   The evaluation cell also advances the loop with the action `route`
   *sampled* at the configured temperature, as MATCHED-001 did — not with the
   argmax. The persistent state and the task stream depend on which action
   ran, so an argmax-driven trajectory would not be MATCHED-001's protocol and
   the gate would fail for a reason that is not about the arm. The ranking
   endpoints are still computed from `router.score`; only the executed action
   differs from it, which is also why `accuracy` is not `routing@1`.

8. **Audit 8's `delta@K` fix, and why it cannot move this run's numbers.**
   This script carried the same `sorted(raw, ...)` line MATCHED-001 did, in
   which the K-th overall score includes gold and `delta@1` is zero by
   construction for a perfect scorer. It is fixed in both scripts; see Audit 8
   in `docs/TACOSM-MATCHED-001.md` for the mechanism and the proof. The fix is
   inert for F2's conclusions because the reproduction gate compares
   `delta_1` only — a key computed from `s_gold`/`s_best_distr` that never
   touched the `deltas` dict — and none of the eight numbers in the Outcomes
   table above is a `delta@K`. The published F2 result is unchanged; the key
   is now correct should a future arm read it.

**Layer 3 is untouched.** No cost claim is made by this experiment. `C_router`
is reported because it is present in the loop, and it is `O(H)` for every arm
including the exploration arms — no arm changes routing cost, and none may.
`C(|R|)` remains unmeasured and unclaimed, per C5.

---

## Cost

3 arms × 3 train-H × 5 seeds = 45 training runs, plus
3 arms × 3 train-H × 3 eval-H × 5 seeds = 135 evaluation cells. Training
dominates; MATCHED-001's 9 training cells took ~90 s, and the marginal cost is
linear in arms, so the full F2 run is on the order of 5–8 minutes on this
device. Scripted in `scripts/measure_learn.py`, the same single-construction
path every other measurement script uses.

The three diagnostics from MATCHED-001 are not re-run here: the analytic
vector result, the success-count probe and the multi-step probe answered *why*
and are recorded in that document. F2 answers a different question — *does a
named intervention move the number* — and re-running diagnostics that already
have answers is how a pre-registered design drifts into a post-hoc one.

---

## Reproduction

```
python scripts/measure_learn.py --steps 500 --eval-steps 100
```

`--steps`, `--eval-steps`, `--seeds` and `--levels` are exposed and match
MATCHED-001's defaults. **No knob is exposed that this document does not
name.** In particular, `eps_0`, `tau_0` and `tau_end` are *not* CLI
arguments: they are constants fixed by this pre-registration, and exposing
them would make the intervention a free variable the run could tune. That is
the failure mode this document exists to prevent.

---

## Outcomes

**RUN.** `scripts/measure_learn.py --steps 500 --eval-steps 100 --seeds
0,1,2,3,4`; 3 arms x 3 train-H x 5 seeds. Full log at
`results/learn001_full.txt`.

**Reproduction gate: PASSED.** The baseline arm reproduced all nine published
MATCHED-001 numbers exactly (diff = 0.0000 on every metric at every H), so the
comparison below is against the same reference column MATCHED-001 published.

*(The decision rule above is fixed and is not amended by the result.)*

| train-H | endpoint | baseline | epsilon-greedy | temperature |
|---|---|---|---|---|
| 8 | `routing@1` | 0.6240 | 0.6280 | 0.6140 |
| 64 | `routing@1` | 0.1320 | 0.1380 | 0.1140 |
| 256 | `routing@1` | 0.0740 | 0.0520 | 0.0600 |
| 256 | `recall@4` | 0.1340 | 0.1060 | 0.1040 |
| 256 | `recall@16` | 0.3020 | 0.2260 | 0.2700 |
| 256 | `delta_1` | -0.4383 | -0.6221 | -0.3010 |
| 256 | `entropy` | 5.3533 | 5.1472 | 5.4521 |
| 256 | `successes / 500` | 2.4 | 3.8 | 1.4 |

Primary endpoint, `routing@1` at H=256: `Δ(epsilon_greedy) = -0.0220`,
`Δ(temperature) = -0.0140`, against a materiality threshold of 0.0600 (the
baseline's own seed spread). **Both are within seed noise.** The decision rule
row that fires is therefore *"`Δ(routing@1)` within spread"*:

> **The failure is not solely exploration.** Fall through to the secondary
> endpoints before concluding anything. This is *not* a representation verdict.

The secondary endpoints do not rescue it. `recall@16` at H=256 moves
**against** both arms: `Δ(epsilon_greedy) = -0.0760`, which is material harm
(beyond the 0.0600 threshold), and `Δ(temperature) = -0.0320`, within noise.
The paired strong outcome — `routing@1` *and* `recall@16` both materially
improved — did not occur; `epsilon_greedy` materially *harmed* the shortlist.

Note the direction the arms move in training-time behaviour versus the
endpoint. `epsilon_greedy` does explore as registered (15.2% of steps) and
does see more successes per step at H=256 (3.8 vs 2.4, against a 0.0039 chance
rate) — the condition the scarcity hypothesis predicts — **and the endpoint
did not improve.** More successes under exploration did not produce a scorer
with a better large-H ranking. `temperature` flattens the policy (tau 2.0 ->
0.5, annealed as registered) and sees *fewer* successes than baseline at every
H, with the endpoint unchanged. Neither knob is the binding constraint.

The interpretation row that fires, and the consequence committed to in the
decision rule, is recorded above verbatim rather than paraphrased. Per
"**What a negative result does not license**" and **C10**, this does not
falsify the mechanism or the representation: it narrows the location of the
failure to something other than the *selection* stage of training, and leaves
the learning rule, the gradient signal, and the reward shaping (all
deliberately untouched by both arms) as the remaining candidates. The
consequence committed to in the decision rule follows from the row: fall
through, do not conclude, and do not add another arm from this document.
