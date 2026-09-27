# TACOSM-SURROGATE-001 (F3) — dense reward from the analytic margin, pre-registered

**Status: pre-registered, then revised once before the run.** This document is
written *before* the experiment runs. The decision rule, the reward
definition, the frozen protocol and the failure taxonomy are fixed here and
are not amended whatever the result turns out to be. A pre-registration is not
revised to fit a result; the row that occurs is recorded, not substituted —
the convention MATCHED-001 and LEARN-001 both established when their own
committed branches failed.

**The revision, recorded before any arm ran.** Three things in the first draft
were wrong. All three were found by measurement or by reading the code rather
than by argument, and none was found by running an arm:

1. **The leakage claim was overstated.** The first draft called the surrogate
   a "gold-free" reward. It is not, and neither is the baseline: the
   environment sets `target_action = gold_index` (`environment.py:370`) and
   `success = action == task.target_action` (`environment.py:695`), so
   `float(outcome.success)` **is** `1[a_t = gold]`. Both rewards are
   gold-anchored scalars of the same logical kind. §"The leakage boundary" is
   rewritten around the distinction that actually holds, and audit item 8 is
   rewritten so it no longer asks the script to assert a falsehood.
2. **The centring rule defeated its own stated purpose.** The first draft
   centred the surrogate by subtracting the per-step mean of the candidate
   scores. Measured by `scripts/probe_surrogate_design.py` over 180 episodes
   per H, `mean(scores)` is *negative* under the analytic vector (−0.99 at
   H=8, −1.58 at H=64, −1.65 at H=256), because distractors score strongly
   negative on the marked positions. Subtracting it therefore *adds*, and the
   centred reward is positive on 180/180 steps at every H (min +1.75 at H=8,
   +2.84 at H=256) — precisely the "constant positive reward" the document
   said the rule existed to prevent. Replaced with a running mean of the
   surrogate itself; see §"The centring rule".
3. **The arms had no implementation surface.** The first draft said the
   intervention acts "in the reward argument to `LearnedRelationalRouter
   .update` only". As the loop is written there is no such argument to vary:
   `model.py:219` hardcodes `reward=float(outcome.success)` and is the sole
   call site of `update`. A reward hook on the loop is required — see §"Arm
   2", "Where it acts" — and its contract is pinned here so it cannot become
   a per-arm code path. Without this the pre-registered arm table is
   unimplementable, which makes it the most consequential of the three.

None of the three touches the decision rule, the primary endpoint, the frozen
protocol, the seed handling or the arm structure. All three make the
experiment test what it claims to test. The amendments are committed before
the run and are not withdrawn whatever the result.

It supersedes nothing and follows `TACOSM-LEARN-001` (F2), whose negative
result is the reason this experiment exists rather than another exploration
schedule.

---

## The question, stated as a falsifiable claim

> **Does a dense surrogate reward move `routing@1` at H=256 where two named
> exploration interventions did not — under the same task distribution, the
> same evaluation protocol and the same frozen baseline?**

"Dense", not "gold-free": the surrogate is anchored to the gold index exactly
as `float(outcome.success)` is. The intervention is the *density* of a signal
of the same logical kind. See "The leakage boundary" for the distinction, and
for the narrower hypothesis this licenses.

F2 established something narrow and worth restating, because it is what makes
this the next experiment rather than a guess:

- `epsilon_greedy` explored as registered (15.2% of steps) and saw **more**
  successes per step at H=256 (3.8 vs 2.4, against a 0.0039 chance rate).
- The endpoint did not move (`Δ(routing@1) = −0.0220` against a 0.0600
  materiality threshold — within seed noise), and the shortlist got
  materially worse (`Δ(recall@16) = −0.0760`).
- MATCHED-001 had already shown that *neither* the representation nor the
  training population is the bottleneck: the analytic vector reaches
  `routing@1 = 1.0000` at H ∈ {8, 64, 256, 512} with `delta_1 ≈ +3.07`, and
  matched-H training is the worst row at every population.

So the router can be *handed* more useful interactions — the exact condition
the scarcity hypothesis predicts — and still not convert them into ranking.
Selection is not the binding constraint. What both arms deliberately left
alone is the reward itself, and that is the one remaining component of the
update that is still binary and terminal.

This experiment changes that, and only that.

---

## Why this is the experiment that separates the two remaining hypotheses

F2 left two live candidates, and they are **not separable by observation**,
because both predict the same thing — more useful interactions, no endpoint
movement — which is what F2 measured:

| hypothesis | prediction under exploration | prediction under dense reward |
|---|---|---|
| **H_credit** — the learning rule cannot convert interactions into ranking | endpoint flat | endpoint flat |
| **H_sparse** — the signal is too rare to estimate a gradient from | endpoint flat | **endpoint rises** |
| **H_scale** — the reward's magnitude, not its availability, is what the update cannot use | endpoint flat | endpoint rises under clipping only |

The two columns differ in exactly one place, and the dense reward is what
makes them differ. A dense reward is not "more exploration with extra steps":
it changes what the update is *about*, on every step, including the 99% of
H=256 steps that currently carry no gradient signal at all.

If `routing@1` stays flat under a dense reward, H_credit is supported and
H_sparse is refuted — and the conclusion is then that REINFORCE with a linear
basis is the wrong learning rule at this population, which reopens the
architecture question on firmer ground than "the current router doesn't
work". If it rises, H_sparse is confirmed and the fix is a training-signal
change, not an architectural one.

---

## The leakage boundary — the load-bearing argument

This is the part that has to be right, and it is worth being explicit about
why a dense reward is not automatically cheating. Two distinct questions get
conflated under the single word "leakage", and only one of them is settled by
the existing gate:

1. **Does the *feature extractor* see gold?** — the interface question. This
   is what `leakage.py` checks, and it is settled.
2. **Is the *gradient direction* equivalent to information that would be
   unavailable to a deployed router under the intended learning setting?** —
   the supervision question. This is the question the first draft reached for
   and answered by asserting the negative of (1).

The first draft's error was to argue (1) and conclude (2). Both questions have
to be answered, and the second has a different answer from the first.

### (1) The interface boundary — settled, and unchanged by this experiment

`leakage.py` forbids the router from seeing `target`, `gold`, `gold_index`,
`answer`, `oracle`, `reward` and a substring scan over the same semantics. The
surrogate is computed in the harness, *outside* the router, and reaches
`update` through the one scalar argument it already accepts:

    surrogate = s_gold(w_t) − max_{j≠gold} s_j(w_t)

where `s(w_t)` is the router's own scoring function over the same feature
basis it already scores. `LearnedRelationalRouter.update` takes `reward:
float` and has **no gold parameter** (`router.py:399`), and `features()` takes
only `query`, `candidate.descriptor` and the state read. No forbidden name
crosses the router's API in any arm, including the baseline. So (1) is closed,
and the surrogate adds nothing to the router's *observable inputs*.

### (2) The supervision boundary — the question that decides whether F3 is honest

The first draft claimed the surrogate was "gold-free" because the gold index
never reaches the feature extractor. That does not follow, and the reason is
in the environment, not in the router:

> `environment.py:370` sets `target_action = gold_index`.
> `environment.py:695` sets `success = action == task.target_action`.
> So `float(outcome.success)` **is** `1[a_t = gold]`.

**The baseline reward is already a gold-indexed scalar.** The outcome reward
is the indicator of the gold action under a different name; the surrogate is
the margin to the gold action. Both are functions of the gold index. Both are
unavailable to a deployed router that receives only an outcome it can observe.
The difference between them is not *whether* gold anchors the signal but
*how much* of it is communicated per step: one bit, versus a graded margin.

So the correct statement of the boundary is not "the surrogate is gold-free"
but:

> **F3 holds the interface boundary fixed and loosens the supervision
> boundary, by a measured amount, in a named direction.**

The amount is what makes the experiment interpretable, and it is measurable
before the run. Measured over 180 episodes per H under the analytic vector,
the surrogate is non-zero on 180/180 steps at every H, while the outcome
indicator for the greedy action is non-zero on 180/180 at H=8 and falls to
0.0056 at H=256 — the 1-in-179 sparsity MATCHED-001 already established as the
failure's mechanism. That is the intervention: the *density* of a signal of
the same logical kind, with the anchor held fixed.

**What makes the surrogate a legitimate *training* signal rather than a leak
into the evaluation.** The analytic vector proves the relation is *computable
from the router's own inputs* — `features()` takes only `query`,
`candidate.descriptor` and the state read, all permitted inputs, and
`analytic_weights` reaches `routing@1 = 1.0000` at every H from those inputs
alone. A signal derivable from the same features is not information the router
was barred from computing; it is a *denser encoding* of information it already
lawfully has. What it does **not** have is a way to *learn* that encoding from
the terminal outcome alone, which is the gap F3 measures.

**The probe.** The density figures in this section come from
`scripts/probe_surrogate_design.py`, which scores the task stream under the
analytic vector and under the zero vector and reports, per H, the non-zero
rate of the outcome indicator against the surrogate and the distribution of
`mean(scores)`. It trains nothing and builds no arm — it exists so that the
claims this section makes about *how much* the intervention changes are
measured before the run rather than asserted after it.

### What F3 does and does not license, stated before the run

**The narrower hypothesis F3 tests, stated honestly:**

> Can the existing router parameterization learn the already-representable
> relevance relation when the training signal is made dense?

**A positive result licenses:** "the relation is learnable by this
parameterization under dense supervision." It licenses investigating how to
obtain an equivalent dense learning signal from legitimate observations.

**A positive result does not license:** "the router learned relevance from
outcomes", or that an operational PLM could acquire that supervision from
ordinary outcomes. The gradient direction under the surrogate is *not*
equivalent to anything obtainable from the deployed loop's own feedback — it
is anchored to gold, and the outcome reward is the only thing the deployed
loop can observe. The mechanism claim (C1) rests on outcome learning; this
experiment does not extend it.

**A negative result does not license:** "persistent-state relevance routing
does not scale" — the argument is the same as C10, and the hypothesis class
provably contains a perfect H-invariant solution.

This is the caveat the first draft placed in one sentence at the end of the
section. It is now the section's conclusion, because it is the distinction the
whole experiment turns on.
---

## The interventions — pinned before the run

### Arm 1: `baseline`

The MATCHED-001 protocol verbatim: reward = `float(outcome.success)`,
`epsilon = 0`, training temperature 0.5, REINFORCE as written. This is the
reference column. It exists so that any change in a surrogate arm is
attributable to the reward and not to the task stream, seed handling,
exploration or evaluation.

The baseline arm **must reproduce the published MATCHED-001 numbers**, or the
run is invalid and no arm is reported. This is the same reproduction gate F2
used, unchanged.

### Arm 2: `analytic_margin`

**The reward.** On every training step, including the steps where the
terminal outcome is a failure:

    surrogate_t = s_gold(w_t) − max_{j ≠ gold} s_j(w_t)

computed from `router.score` — the raw dot products, in the same
non-normalised units `delta_1` uses, so the margin is comparable across H.
The update receives `surrogate_t` in place of `float(outcome.success)`.

**Centring.** The outcome reward is centred by `chance = 1/H`. The surrogate
is centred by a **running mean of the surrogate itself**:

    baseline_t = (t · baseline_{t−1} + surrogate_t) / (t + 1)
    centred_t  = surrogate_t − baseline_t

This is the standard policy-gradient baseline: a state-independent estimate of
the reward's mean, which reduces the update's variance without biasing it.

**Why a running mean and not the per-step score mean.** The first draft
centred by subtracting the per-step mean of the candidate scores. Measured
over 180 episodes per H under the analytic vector, `mean(scores)` is
*negative* — −0.99 at H=8, −1.58 at H=64, −1.65 at H=256 — because the
distractors score strongly negative on the marked positions the analytic vector
keys on. Subtracting a negative quantity *adds*, so the centred reward was
positive on 180/180 steps at every H, with a minimum of +1.75 at H=8 and
+2.84 at H=256. That is exactly the failure the centring term was written to
prevent, in the first draft's own words: "a margin that is positive on every
step would otherwise be a constant positive reward and the update would
degrade to plain gradient ascent on the selected features, which is a
different learning rule and is not what this arm tests."

The running mean does not have this failure mode, because it tracks the
surrogate rather than the scores: under the analytic vector it converges
toward ≈ +3.0, so `centred_t ≈ 0` with graded step-to-step deviation, and the
update stays signed by *deviation from average performance* rather than by raw
magnitude.

**Density survives the baseline.** The centred quantity is still a graded,
continuous function of the weights on every step. The contrast with the
baseline arm is unaffected: the outcome reward carries one bit per step and
0.0056 non-zero rate at H=256, this arm carries a real number on every step.

**Where it acts, and the change the loop needs for there to be a "where".**
The first draft said the intervention acts "in the reward argument to
`LearnedRelationalRouter.update` only". That is the right *scope*, but as the
loop is written there is no such argument to vary: `model.py:219` hardcodes

    self.router.update(..., reward=float(outcome.success), ...)

and `update` is called from exactly one place, inside `TacOsmModel.step`. So
the arm cannot be selected by a config field the way F2's exploration was —
F2's knob lives on `ExplorationSchedule`, which the router itself consults,
whereas the reward is computed by the loop before the router sees it.

The change is therefore a **reward hook on the loop**, added once and used by
all three arms, and the pre-registration pins its contract so it cannot
become a per-arm code path:

- `ModelConfig` gains one field, `reward_fn: Callable[[Step-ctx], float] | None`.
  `None` is the baseline and must be bit-for-bit identical to the current
  hardcoded call — this is what makes the baseline arm reproduce
  MATCHED-001, and it is asserted by the reproduction gate, not assumed.
- The hook receives everything the loop has at the moment of the update: the
  task (hence `target_action`, the gold index), the candidates, the decision
  and the outcome. It returns the scalar passed as `reward`.
- **The hook is the entire intervention surface.** It touches nothing else:
  not `features`, not `basis_size`, not the candidate generator, not the
  evaluator, not the exploration schedule (`epsilon = 0`, temperature 0.5
  throughout, as baseline), not the transition dynamics, and not the executed
  action. The loop still advances on the action `route` sampled at the
  configured temperature, exactly as MATCHED-001 did.
- The three arms are three named `reward_fn` implementations, reached by name
  from the arm name — the same lookup-not-constructor pattern F2's
  `arm_exploration` uses, so a caller cannot pass a reward function of its
  own any more than it can pass an exploration schedule of its own. The
  centring rule, the clip bounds and the reward definition are not
  command-line arguments, and neither is the hook's identity.
- The hook is consulted **only when `config.learn` is true**. At evaluation
  the router is rebuilt fresh, `learn = False`, and the endpoint is computed
  from `router.score` exactly as MATCHED-001 computed it. There is no path
  from the hook to the reported `routing@1`.

This is a loop change, not a router change, and it is the minimal one: without
it the pre-registered arm table has no implementation, and with it every arm
still differs from the baseline in exactly one named place.

**Why the analytic margin and not the outcome-shaped margin.** Shaping the
outcome (a bonus on success, a penalty on failure) changes one number and
leaves the signal binary; the analytic margin is dense on every step, which
is the property H_sparse is about.

### Arm 3: `analytic_margin_clipped`

**The reward.** As `analytic_margin`, with the surrogate clipped to
`[0, 1]` before centring.

**Why this arm exists.** It separates the *density* of the learning signal
from its *scale and normalisation*. The unclipped arm delivers a graded margin
whose magnitude varies by an order of magnitude across training; the clipped
arm delivers a bounded magnitude in a fixed range. If they differ, the
mechanism is the scale of the surrogate rather than its availability, and that
distinction is what the arm-vs-arm comparison below turns on.

**The saturation this arm is subject to, recorded before the run.** Under the
analytic vector the margin is never below +2.0 (measured minimum +2.00 at
every H), so `clip(margin, 0, 1) = 1.0` on 100% of steps once the scorer is
behaving well. The clipped arm therefore carries a *graded* signal only in the
early-learning region where `margin ∈ (0, 1)`, and a constant reward
thereafter. That is a known limitation, not a defect: it makes arm 3 a test of
the gradient magnitude rather than the gradient direction, and it is reported
as such. A reader must not read a flat clipped arm as "density does not help"
— density is arm 2's claim.

### Knobs, all in one place

| arm | reward | centring | clip | exploration | training only? |
|---|---|---|---|---|---|
| `baseline` | `float(outcome.success)` | `1/H` | — | none | — |
| `analytic_margin` | `s_gold − max_{j≠gold} s_j` | running mean of the surrogate | none | none | reward only |
| `analytic_margin_clipped` | `clip(s_gold − max_{j≠gold} s_j, 0, 1)` | running mean of the surrogate | `[0, 1]` | none | reward only |

Every value in this table is fixed by this pre-registration. **None is
exposed as a command-line argument** — see "Reproduction".

---

## The frozen protocol — what does not change

| Frozen | Value | Source |
|---|---|---|
| task families | `relational`, `state_lookup`, `replay`, cycling | `WorldConfig.families` |
| `dim` | 8 | `WorldConfig.dim` |
| noise | 0.10 | `WorldConfig.noise` |
| train H | 8, 64, 256 | the MATCHED-001 levels |
| eval H | 8, 64, 256 | matched to the training levels |
| training steps | 500 | MATCHED-001, so the baseline reproduces |
| eval steps | 100 per cell | MATCHED-001 |
| seeds | 0–4, train and eval matched within a cell | MATCHED-001 |
| evaluation router temperature | 0.5 | `RouterConfig.temperature`, unchanged |
| evaluation protocol | `router.score` raw dot products; `routing@1 = raw[gold] == max(raw)` | MATCHED-001 |
| exploration | none, in any arm | F2's negative result |
| feature basis | the six blocks, `analytic_weights` unchanged | §34 gate |
| integrity gate | unconditional, before every cell | C4 |
| oracle control | printed first, per eval H; must be 1.0000 | C1 |
| leakage audit | `leakage.py` runs at construction, as every arm does | G1 |

**Only the reward differs between arms.** Everything else is identical by
construction, and every arm is built through `build_model` from an
`AblationConfig` — the single construction path, so no arm has its own code
path. An arm that needed to change the basis, the evaluator or the
exploration would be a different experiment and is rejected here, not added.

**Evaluation is identical across arms.** The surrogate reward is a
*training-time* quantity. At evaluation, learning is disabled
(`config.learn = False`), the router is rebuilt fresh with only `load_weights`
applied, so no surrogate reaches the ranking. `routing@1` is computed from
`router.score` exactly as MATCHED-001 computed it, ties included, so the
endpoint is the same number the published tables report.

---

## The decision rule — committed in advance

For each train-H, compare each surrogate arm to the baseline at the same H,
same seeds, same evaluation. `Δ` is the surrogate arm's value minus the
baseline's, so `Δ > 0` is an improvement. A difference counts as material
only if it exceeds the seed spread, `spread = max(arm) − min(arm)` across
that metric's values at that H over the five seeds — the standard
MATCHED-001 and F2 used, chosen because it is measurable before any arm is
compared.

### Primary endpoint: `routing@1` at `train-H = 256`

This is the number MATCHED-001 left at 0.0740, F2 failed to move, and the
scarcity hypothesis predicts a dense reward can.

| Outcome | Interpretation | Consequence committed to now |
|---|---|---|
| `Δ(routing@1) > spread` | dense reward lifts argmax routing at large H | **H_sparse is supported and H_credit is refuted.** The failure was the sparsity of the signal, not the rule that consumes it. Record the mechanism, then re-queue **F1** (the retrieval index) on a router whose quality is a property of the scorer rather than of the optimiser. |
| `Δ(routing@1)` within spread | dense reward does not fix argmax routing either | **H_credit is supported and H_sparse is refuted.** REINFORCE with a linear basis is the wrong learning rule at this population, and this is now evidence *for* a different rule or a different architecture rather than a bare negative. See the next section for what this does *not* license. |
| `Δ(routing@1) < −spread` | the surrogate actively hurts | The intervention is harmful; record and stop. No further arm is added from this document. |

### Secondary endpoints, in order

1. **`recall@4`, `recall@8`, `recall@16`** — the shortlist signal. The case
   the decision rule specifically allows for: the primary endpoint stays flat
   while top-K recall becomes useful. Then F1 re-queues on top-K evidence
   alone, recorded as "top-K signal intact", *not* as "`routing@1` improved".
2. **`delta_1 = s_gold − s_best_distractor`** — the separation. If the
   surrogate widens `delta_1` at H=256 without lifting `routing@1`, the
   scorer is discriminating better while the best distractor still wins — a
   population-discrimination problem, not a reward problem.
3. **`delta@K` for K ∈ {2, 4, 8, 16}** — the K-th competitor margins, now
   corrected (Audit 8 in MATCHED-001: the metric previously subtracted the
   K-th *overall* score including gold, making `delta@1` zero by construction
   for a perfect scorer). Read paired with the recall tables: where
   `recall@K` is high and `delta@K` is small, gold sits near the edge of the
   budget.
4. **`gold_rank`** — mean rank of gold. Detects whether the field moved past
   gold without gold moving at all.
5. **`entropy`** against `ln H` — how concentrated the policy is. Expected to
   fall in an arm that finds the signal, because a working policy sharpens.
6. **`accuracy`** — the system consequence, from the executed loop. Included
   so a routing improvement can be seen to propagate (or not) to the outcome.
   **Not the primary endpoint**: an arm that raises `accuracy` without
   raising `routing@1` has made the *sampler* luckier, not the scorer better,
   and is reported as such.
7. **`prob_margin`** — the gap in the softmax units the decision samples
   from. Reported for the same reason as in MATCHED-001: the loop is governed
   by the softmax, so the decision-relevant margin is the probability margin
   even though the cross-H-comparable one is `delta_1`.
8. **Seed variance** — per-seed values for every metric, not only the mean.
   The spread test runs on these, and a 5-seed mean that hides a 4-seed
   collapse and a 1-seed success is not a result.

### The interactions that matter

`routing@1` and `recall@16` are the pair. A material `Δ(routing@1)` **and** a
material `Δ(recall@16)` together is the strong outcome: the scorer got better
at every budget. A material `Δ(recall@16)` with flat `routing@1` is the weak
outcome and re-queues F1 on top-K evidence alone. **Flat on both** is the
negative result, and it is reported as a negative result.

### The arm-vs-arm comparison

If `analytic_margin_clipped` fires and `analytic_margin` does not, the
mechanism is the *scale* of the surrogate rather than its density. That is a
distinct finding and is recorded separately, because it points at gradient
magnitude rather than signal availability — the next experiment would then be
about normalisation, not about reward design.

**But read the saturation caveat first.** Because `clip(margin, 0, 1)` is
saturated at 1.0 whenever `margin > 1` — and the analytic vector's margin
never falls below +2.0 — arm 3 carries a graded signal only while the router
is still bad enough to produce a margin in `(0, 1)`. So the asymmetric
outcome above has two readings, and the run has to record which occurred:

- arm 3 fired *because* clipping bounded a harmful magnitude (a genuine
  H_scale result), or
- arm 3 fired because a constant unit reward is simply a better-shaped
  gradient for this update than a variable one — which is a result about
  reward *shape*, not about the surrogate.

The per-step reward statistics in "How to verify the protocol was followed"
are what separates them: the fraction of steps on which arm 3's pre-clip
margin exceeded 1.0 tells a reader how much of the run arm 3 spent saturated,
and an arm that was saturated for 95% of its steps has not tested scale at
all.

---

## What a negative result does *not* license

This is the most important paragraph in the document, and it is written
before the run for that reason.

A negative F3 does **not** establish that persistent-state relevance routing
does not scale. It establishes that a specific named dense reward — the
analytic margin, centred by a running mean of itself, with and without clipping
to `[0, 1]` — does not fix the trained router at H=256 under this protocol.
Those are different claims of very different strength, and conflating them is
exactly the error the measurement-layer contract exists to prevent.

The distinction the contract enforces:

- **"the current router doesn't scale"** — a claim about one trained linear
  scorer under one update rule with one reward. F3 can address this.
- **"persistent-state relevance routing doesn't scale"** — a claim about the
  mechanism. F3 cannot address this, and neither can any single intervention
  experiment, because the hypothesis class provably contains a perfect
  H-invariant solution (`analytic_weights`, `routing@1 = 1.0000` at
  H ∈ {8, 64, 256, 512}). A mechanism whose hypothesis class is adequate
  cannot be pronounced inadequate by a failure to *find* the solution with
  one optimiser.

On a negative result the inspection order, committed to now, is unchanged
from F2 — the order is a contract, not a convenience, and a negative result
is not a reason to reorder it to fit the finding:

```
representation  →  training dynamics  →  population discrimination  →  retrieval architecture
```

1. **Representation** — already negative, twice: the §34 gate passes and the
   analytic vector is perfect at every H. This is not where a negative F3
   points.
2. **Training dynamics** — F2 closed two exploration interventions; F3
   closes the reward-shaping option MATCHED-001 recorded, and in doing so
   *adds* the policy-gradient baseline this category was waiting on (the
   running-mean centring in §"Arm 2" is that baseline, in the surrogate's own
   units). What remains in this category after F3 is a *different update
   rule*: off-policy replay of the successes, or an off-policy estimate that
   reuses the 500-step trajectory rather than discarding it. Those are
   different experiments and are neither licensed nor excluded by F3's
   result.
3. **Population discrimination** — `gold_rank`, `delta_1` and the corrected
   `delta@K` are what address it, and they are in the secondary endpoints
   precisely so a negative primary endpoint does not skip this step.
4. **Retrieval architecture** — F1. Last, and re-queues only on the primary
   endpoint firing or the top-K signal being shown intact.

**A negative F3 is, however, the strongest evidence yet that the failure is
not where we have been looking.** Two named interventions on selection and
one on reward, all three failing to move the same number, is a constraint on
the *location* of the failure rather than a list of things that do not work.
That is a result, not an absence of one.

---

## Layer 1 obligations — the gate runs here too

The contract in `docs/MEASUREMENT_LAYERS.md` applies in full. The surrogate
reward creates two new Layer 1 hazards the design must close:

**Hazard 1 — supervision leaking into evaluation.** The surrogate is computed
from the gold-anchored margin. If the *evaluation* cell computes it, the
endpoint is measuring the supervision rather than the learned ranking. Closed
by construction: the eval router is rebuilt fresh via `build_model`, only
`load_weights` is applied, `config.learn = False`, and `routing@1` is
computed from `router.score` exactly as MATCHED-001 computed it. There is no
path from the surrogate to the reported endpoint.

**Hazard 2 — warm-start confusion.** As in F2: the baseline and the surrogate
arms must be independently trained from `w = [0]*n`, from the same seed,
consuming the same task stream. `assert_trained` only refuses the all-zero
vector — it deliberately does not compare two trained checkpoints, because
that would be a modelling judgement the gate is not designed to make. So this
is a design obligation, not a gate: the script builds every arm from
`AblationConfig` through `build_model`, and the only difference between arms
is the named reward.

| Layer 1 check | How it is enforced here |
|---|---|
| weights are not the initialisation | `assert_trained` before every training and evaluation cell, unconditional |
| checkpoint corresponds to the arm under test | `snapshot_router` → `load_weights`, hash in the manifest |
| feature basis represents the relation | §34 via `builder.check_representability`, at construction |
| gold is the unique satisfier, and not exposed to the router | generator integrity tests + `leakage.py` |
| oracle = 1.0000 | printed **first**, per eval H, before any arm table |
| the frozen baseline is reproduced | the `baseline` arm must match the published MATCHED-001 numbers |
| seeds reproduce the task | `WorldEnvironment` is deterministic given the seed |
| the surrogate does not reach the endpoint | eval router rebuilt fresh, `learn = False`, endpoint from `router.score` |
| the reward is dense where it claims to be | per-arm non-zero rate, min and mean reported per H (audit item 2) |
| arm 3's saturation is visible | fraction of steps with pre-clip margin > 1.0, reported per H (audit item 2) |
| benchmark version pinned | the frozen protocol table above |

**The gate's scope is unchanged.** F3 adds one Layer 1 check (the
surrogate-evaluation separation, closed by construction) and weakens none.

---

## How to verify the protocol was followed

The run records:

1. The frozen protocol table from this document, printed at the head of the
   output.
2. Per-arm reward definition and its per-step statistics: the mean, min and
   max surrogate actually seen during training at each H, and the fraction of
   steps on which it was non-zero. This is the audit trail that the reward
   *is* dense — a surrogate that is zero on 99% of steps would be the
   terminal reward with extra steps, and would not test H_sparse.
3. The integrity checkpoint for every arm: `parameter_hash`,
   `parameter_norm`, `n_updates`, `weight_signature`, `source`. The
   `weight_signature` is the block means — `gated_agreement`,
   `slot_gated`, `query_agreement` — so a reader can see *what rule* each
   arm learned, and whether a surrogate arm that improved `routing@1` did so
   by finding the analytic structure (`gated_agreement` positive,
   `slot_gated` dominant) or by something else.
4. Success counts per arm per H during training, retained so the comparison
   against F2's `epsilon_greedy` (3.8 successes at H=256, more than
   baseline's 2.4, with a flat endpoint) is on the same footing.
5. Per-seed values for every metric, not only the mean.
6. Oracle accuracy per eval H, printed before any arm table.
7. The `argmax@1` audit column — not a registered endpoint, never part of the
   decision rule. Where it differs from `routing@1` the difference is ties,
   and a reader can see whether a tie rule or a learned ranking is doing the
   work. (See LEARN-001 audit item 7 for the full statement.)
8. A **leakage re-audit** on the surrogate's computation, asserting the claim
   "The leakage boundary" actually makes. Two separate assertions, because
   there are two separate boundaries:

   - **Interface assertion (the one `leakage.py` already makes).** The
     computation of the surrogate passes the router's own observable inputs
     and parameters only: no forbidden name (`gold`, `gold_index`, `target`,
     `oracle`, `answer`, `reward`) is read by the router or by `features()`.
     This is the assertion the first draft asked for, and it is true and
     checkable.
   - **Supervision assertion (the one the first draft *claimed* and is false).**
     No runtime assertion is made that the surrogate is "gold-free", because
     it is not: the surrogate is anchored to the gold index, exactly as
     `float(outcome.success)` is. What the run records instead is the measured
     *density* of each reward — the non-zero rate per arm per H — which is the
     quantity the intervention actually changes. A reader verifying the
     experiment checks that density, not an impossible absence of gold.

   The distinction matters because audit item 8 as first written would have
   asked the implementation to assert a falsehood, and a check that cannot
   pass is a check that gets deleted.

**Layer 3 is untouched.** No cost claim is made by this experiment. `C_router`
is reported because it is present in the loop and is `O(H)` for every arm —
no arm changes routing cost, and none may. `C(|R|)` remains unmeasured and
unclaimed, per C5.

---

## Cost

3 arms × 3 train-H × 5 seeds = 45 training runs, plus
3 × 3 × 3 × 5 = 135 evaluation cells. MATCHED-001's 9 training cells took
~90 s and F2's full run 5–8 minutes, so F3 is the same order — the surrogate
adds one margin computation per training step, which is already dominated by
`score`'s dot products. Scripted in `scripts/measure_surrogate.py`, the same
single-construction path every other measurement script uses.

`scripts/probe_surrogate_design.py` is not part of this cost: it is the
pre-run probe that measured the density and centring claims above, it trains
nothing, and it is re-run only if the task stream changes. It is committed
alongside the pre-registration so the numbers it produced are reproducible
rather than quoted.

The diagnostics are not re-run: the analytic-vector result, the success-count
probe and the multi-step probe answered *why* and are recorded in
MATCHED-001. F3 answers a different question — *does a dense reward move the
number* — and re-running diagnostics that already have answers is how a
pre-registered design drifts into a post-hoc one.

---

## Reproduction

```
python scripts/measure_surrogate.py --steps 500 --eval-steps 100
```

`--steps`, `--eval-steps`, `--seeds` and `--levels` are exposed and match
MATCHED-001's defaults. **No knob is exposed that this document does not
name.** In particular the centring rule, the clip bounds and the reward
definition itself are *not* CLI arguments: they are fixed by this
pre-registration, and exposing them would make the intervention a free
variable the run could tune. That is the failure mode this document exists
to prevent.

---

## Status after the run

*(This section is filled in when the run happens. The decision rule above is
fixed and is not amended by the result.)*

| train-H | endpoint | baseline | analytic_margin | analytic_margin_clipped |
|---|---|---|---|---|
| 8 | `routing@1` | 0.6240 | — | — |
| 64 | `routing@1` | 0.1320 | — | — |
| 256 | `routing@1` | 0.0740 | — | — |
| 256 | `recall@16` | 0.3020 | — | — |
| 256 | `delta_1` | −0.4383 | — | — |
| 256 | `successes / 500` | 2.4 | — | — |

The row that fired, and the consequence committed to in the decision rule, is
recorded verbatim rather than paraphrased.
