# TACOSM-SURROGATE-001 (F3) — dense reward from the analytic margin, pre-registered

**Status: pre-registered.** This document is written *before* the experiment
runs. The decision rule, the reward definition, the frozen protocol and the
failure taxonomy are fixed here and are not amended whatever the result turns
out to be. A pre-registration is not revised to fit a result; the row that
occurs is recorded, not substituted — the convention MATCHED-001 and LEARN-001
both established when their own committed branches failed.

It supersedes nothing and follows `TACOSM-LEARN-001` (F2), whose negative
result is the reason this experiment exists rather than another exploration
schedule.

---

## The question, stated as a falsifiable claim

> **Does a dense, gold-free surrogate reward move `routing@1` at H=256 where
> two named exploration interventions did not — under the same task
> distribution, the same evaluation protocol and the same frozen baseline?**

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
why a dense reward is not automatically cheating.

The environment's reward is `float(outcome.success)` at `model.py:219` — a
binary terminal outcome. `leakage.py` forbids the router from seeing
`target`, `gold`, `gold_index`, `answer`, `oracle`, `reward` and a substring
scan over the same semantics. The surrogate reward must pass that boundary
without reading any of them.

**The argument that it does.** The surrogate is computed from the *router's
own inputs and its own current parameters*, not from the gold index:

    surrogate = s_gold(w_t) − max_{j≠gold} s_j(w_t)

where `s(w_t)` is the router's own scoring function over the same feature
basis it already scores. The gold *index* appears in this expression only to
select which score to subtract — it does not enter the weights, the features
or the gradient direction beyond naming the anchor of a margin the router
could itself have computed by ranking.

This is the crucial distinction and it is the same one the §34
representability gate already licenses: `representable()` passes
`gold_fn` as `(episode) -> gold index` and uses it *only to locate the target
for measurement*, never as an input to the mechanism. The gate's own docstring
says so: "Used only to locate the target; it is never an input to the
mechanism." The surrogate reward is the training-time analogue of exactly
that: the gold index locates which score to anchor the margin to, and the
*margin itself* is a function of the router's parameters and its own feature
basis.

**What makes this a legitimate supervision signal rather than a leak.** The
analytic vector proves the relation is *computable from the router's own
inputs* — `features()` takes only `query`, `candidate.descriptor` and the
state read, all of which are already permitted inputs, and `analytic_weights`
reaches `routing@1 = 1.0000` at every H from those inputs alone. So a signal
derivable from the same features is not information the router was barred
from; it is a *denser encoding* of information it already lawfully has.

**What this is not.** It is not a per-candidate label, not a demonstration of
the gold candidate, and not a reward that depends on the gold index through
any path other than which score anchors the margin. The gradient remains
`grad = lr * (surrogate − centring) * (1 − p_selected) * features`, so the
update still only touches the selected action's features and the sign still
comes from a surprise term.

**The honest caveat, recorded before the run.** A surrogate reward is a
*different experiment* from outcome learning, and its result does not license
the claim "the router learned relevance from outcomes". If the endpoint
rises, what has been demonstrated is that the hypothesis class is learnable
under dense supervision — which is weaker than what the loop claims to test,
and is reported as such. The mechanism claim (C1) rests on outcome learning;
this experiment does not extend it.

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
is centred by its own per-step mean across candidates, computed from the same
score list, so the centring term stays a *surprise* measure rather than a raw
magnitude. A margin that is positive on every step would otherwise be a
constant positive reward and the update would degrade to plain gradient
ascent on the selected features, which is a different learning rule and is
not what this arm tests.

**Where it acts.** In the reward argument to `LearnedRelationalRouter.update`
only. It touches nothing else: not `features`, not `basis_size`, not the
candidate generator, not the evaluator, not the exploration schedule
(`epsilon = 0`, temperature 0.5 throughout, as baseline), not the transition
dynamics, and not the executed action. The loop still advances on the action
`route` sampled at the configured temperature, exactly as MATCHED-001 did.

**Why the analytic margin and not the outcome-shaped margin.** Shaping the
outcome (a bonus on success, a penalty on failure) changes one number and
leaves the signal binary; the analytic margin is dense on every step, which
is the property H_sparse is about.

### Arm 3: `analytic_margin_clipped`

**The reward.** As `analytic_margin`, with the surrogate clipped to
`[0, 1]` before centring.

**Why this arm exists.** An unclipped margin at H=256 can be large and
negative (the trained router's `delta_1` at H=256 is −0.4383, and at
H_eval=256 for the H=8-trained router, −2.0126). A negative surrogate centred
on a negative mean is a *double negative* on the selected action, which can
push the update the opposite way from what the sign of the margin intends.
Clipping tests whether that scale is doing the work: if the clipped arm
improves and the unclipped does not, the mechanism is the *scale* of the
surrogate, not its density, and that distinction is what the decision rule
below turns on.

### Knobs, all in one place

| arm | reward | centring | clip | exploration | training only? |
|---|---|---|---|---|---|
| `baseline` | `float(outcome.success)` | `1/H` | — | none | — |
| `analytic_margin` | `s_gold − max_{j≠gold} s_j` | per-step mean across candidates | none | none | reward only |
| `analytic_margin_clipped` | `s_gold − max_{j≠gold} s_j` | per-step mean across candidates | `[0, 1]` | none | reward only |

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

---

## What a negative result does *not* license

This is the most important paragraph in the document, and it is written
before the run for that reason.

A negative F3 does **not** establish that persistent-state relevance routing
does not scale. It establishes that a specific named dense reward — the
analytic margin, with per-step mean centring, with and without clipping to
`[0, 1]` — does not fix the trained router at H=256 under this protocol.
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
   closes the reward-shaping option MATCHED-001 recorded. What remains in
   this category is a *different update rule*: a policy-gradient baseline, or
   off-policy replay of the successes. Those are different experiments and
   are neither licensed nor excluded by F3's result.
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
8. A **leakage re-audit** on the surrogate's computation, asserting that the
   surrogate is a function of the router's own parameters and permitted
   inputs only — the claim made in "The leakage boundary" above, verified at
   runtime rather than argued in prose.

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
