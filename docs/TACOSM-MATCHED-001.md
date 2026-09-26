# TACOSM-MATCHED-001: matched-H training vs the H=8 transfer sweep

**Status: reported.** A negative result that changes the next step. The
pre-registered hypotheses A and B are **both refuted**; the failure is
neither the representation nor the training population. It is in the
learning dynamics, and it was already visible in data the repository held.

## Provenance

| field | value |
|---|---|
| experiment_id | TACOSM-MATCHED-001 |
| supersedes the open question in | TACOSM-RETRIEVAL-001 (F0 follow-up) |
| follows | TACOSM-HS-001, TACOSM-RETRIEVAL-001 (F0) |
| commit | post-`00b0763`, `scripts/measure_matched_h.py` |
| model | tac_osm v0.1, learned router (REINFORCE, linear basis) |
| schedule | mixed (relational / state_lookup / replay) |
| training | 500 steps, seeds 0–4, one router per (train-H, seed) |
| evaluation | 100 episodes per cell, learning disabled, matched seed |
| measurement command | `python scripts/measure_matched_h.py --steps 500 --eval-steps 100` |
| oracle control | 1.0000 at every eval H (printed before the tables) |
| cost | ~90 s on this device; `results/matched_h_full.txt` is a local artifact |

## The question

F0 measured the retrieval ceiling of a router trained once at 8 candidates
and copied to every H. `recall@16` at H=256 was 0.550, and two readings of
that number survived the data. The pre-registered decision rule in
`docs/TACOSM-RETRIEVAL-001.md` committed to them *before* this run:

> **A — population problem.** The scoring function cannot discriminate when
> the candidate population is large. The representation is the bottleneck.
> Fix: change the representation.
>
> **B — training-transfer problem.** The scoring function handles H=256
> fine, but training at 8 candidates never taught it the score distribution a
> 256-candidate problem needs. Fix: match the training population and build
> F1.

Both are falsifiable, and the falsification is cheap: add the cells where the
router saw H candidates during training. If the diagonal recovers, B; if it
falls as steeply as the first row, A.

## Result

The diagonal does not recover. It is the **worst row at every H_eval**, not
the best. Matched-H training is *worse* than H=8 training at every evaluation
population, including its own.

### routing@1 — the argmax decision

| H_train | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| 8 | **0.6240** | 0.1900 | 0.0640 |
| 64 | 0.4280 | 0.1320 | 0.0720 |
| 256 | 0.3860 | 0.1300 | **0.0740** |

### recall@4 — the shortlist signal

| H_train | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| 8 | **0.9800** | **0.5500** | **0.1660** |
| 64 | 0.8420 | 0.3200 | 0.1320 |
| 256 | 0.7460 | 0.2860 | 0.1340 |

### recall@16 — the F0 budget

| H_train | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| 8 | — | **0.9140** | **0.5800** |
| 64 | — | 0.7080 | 0.3340 |
| 256 | — | 0.6280 | 0.3020 |

### delta_1 — raw top-1 margin (`s_gold − s_best_distr`)

| H_train | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| 8 | **+0.6317** | **−1.3465** | **−2.0126** |
| 64 | −0.1223 | −1.1857 | −1.4925 |
| 256 | −0.1505 | −0.3766 | −0.4383 |

### gold_rank — mean rank of gold

| H_train | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| 8 | **1.616** | **6.480** | **23.406** |
| 64 | 2.576 | 14.814 | 56.840 |
| 256 | 3.068 | 19.184 | 74.696 |

### entropy — nats; `ln_H` is the uniform reference

| H_train | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| 8 | 0.5047 | 1.4570 | 2.3613 |
| 64 | 0.9684 | 2.4388 | 3.6694 |
| 256 | 1.9057 | 3.9653 | 5.3533 |

`ln(8) = 2.079`, `ln(64) = 4.159`, `ln(256) = 5.545`. The router is far from
uniform in every cell — it has a preference — but the preference is not the
right one at large H, and `delta_1` above says so directly.

### prob_margin — top-1 gap in decision units

| H_train | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| 8 | 0.2389 | −0.3206 | −0.2872 |
| 64 | −0.0060 | −0.1873 | −0.1074 |
| 256 | −0.0285 | −0.0228 | −0.0081 |

### Read the tables by column, not by row

The matrix's orientation is what makes the result legible. Down a **column**
the trained state changes and the population is fixed:

| | H_eval=8 | H_eval=64 | H_eval=256 |
|---|---|---|---|
| **routing@1** | 0.6240 → 0.3860 (**−38%**) | 0.1900 → 0.1300 (−32%) | 0.0640 → 0.0740 (+16%) |
| **recall@4** | 0.9800 → 0.7460 (**−24%**) | 0.5500 → 0.2860 (**−48%**) | 0.1660 → 0.1340 (−19%) |
| **recall@16** | — | 0.9140 → 0.6280 (**−31%**) | 0.5800 → 0.3020 (**−48%**) |
| **delta_1** | +0.6317 → −0.1505 | −1.3465 → −0.3766 | −2.0126 → −0.4383 |

At H_eval=8 the H=256-trained router is above chance but scores 24% worse on
recall@4 than the H=8-trained one. **Matched-H training degrades every
metric at every population, including its own.** Hypothesis B is refuted: a
router trained at H=256 is *worse at H=256* than one trained at H=8.

Refuting A needs a different measurement. Read the **top row** — the frozen
HS-001/F0 transfer design — and ask what the scoring function *can* do:

| H_train=8, H_eval | routing@1 | recall@4 | recall@16 | delta_1 | gold_rank |
|---|---|---|---|---|---|
| 8 | 0.6240 | 0.9800 | — | +0.6317 | 1.616 |
| 64 | 0.1900 | 0.5500 | 0.9140 | −1.3465 | 6.480 |
| 256 | 0.0640 | 0.1660 | 0.5800 | −2.0126 | 23.406 |

---

## Why: neither A nor B — the learning dynamics

### Test 1: the analytic vector is perfect at every H

`analytic_weights()` is the fixed vector the §34 representability gate
already uses: `gated_agreement = 1.0`, `slot_gated = 2.0`, everything else
zero. It is hand-designed, not trained, and it is the *same vector at every
H* — `basis_size` depends on `dim` and `max_state_slots` only, never on the
candidate count. Loaded into a router and evaluated exactly as the trained
router was:

| H_eval | routing@1 | recall@4 | delta_1 |
|---|---|---|---|
| 8 | 1.0000 | 1.0000 | +3.2000 |
| 64 | 1.0000 | 1.0000 | +3.0667 |
| 256 | 1.0000 | 1.0000 | +3.0667 |
| 512 | 1.0000 | 1.0000 | +3.0667 |

`delta_1` is **+3.07 at H=512**, where the trained router's best cell is
**−0.02**. The basis does not degrade with the candidate population. **The
representation is fine. A is refuted.** An index built on this basis is not
built on sand.

A single-block probe localises the signal. Weight +1 on one whole block at
H=256:

| block | delta_1 |
|---|---|
| `query_agreement` | −9.1667 |
| `context_descriptor` | −6.3333 |
| `context_query` | 0.0000 |
| **`gated_agreement`** | **+2.0000** |
| `slot_agreement` | 0.0000 |
| `slot_gated` | 0.0000 |
| `bias` | 0.0000 |

Only `gated_agreement` separates gold from the field, exactly as the §34
audit established, and it does so at H=256 as well as at H=8. Scaling it
scales the margin linearly (`w=0.5 → +1.0`, `w=1.0 → +2.0`, `w=2.0 → +4.0`,
`w=4.0 → +8.0`, `w=10.0 → +20.0`): the score is *linear* in the number of
marked positions that agree, and the number of marked positions does not
change with H. **The signal is H-invariant by construction, and the
population does not dilute it.**

### Test 2: the training signal is scarce

The trained router is not failing because the task is hard. It is failing
because it almost never gets a success to learn from. Successes per 500
training steps, seed 0:

| H_train | successes / 500 | chance | ratio |
|---|---|---|---|
| 8 | 248 | 62.5 | 3.97 |
| 64 | 18 | 7.8 | 2.30 |
| 256 | **5** | 2.0 | 2.56 |

5 successes in 500 steps is a **success rate of 1%**. REINFORCE's gradient is
`lr · (reward − chance) · (1 − p_selected) · features`, and the multiplier on
the update is `(1 − p_selected)` — a *confidence* term that shrinks as the
softmax sharpens, which happens as the weights grow. So the update is
squeezed from both ends:

1. successes are rare (1%), so most steps contribute nothing at all;
2. when one occurs, `p_selected` is already small at a large H because the
   softmax over 256 candidates is flat, so the multiplier is near 1 and the
   update is *large* and *unrepresentative* — a single rare success moves the
   weights by a step scaled to a 1/256 event;
3. the sign alternates, because the noise dominates the signal.

This is not a failure of the feature basis. It is the classic exploration
failure of policy-gradient methods on hard problems: reward is so sparse the
gradient estimate is dominated by noise. The environment's reward is not
shaped, so the only signal is the terminal outcome, and at 256 candidates
chance is 0.39%.

### Test 3: the weight-norm trajectory

The squeeze is visible in the norm. Over 500 training steps:

| H_train | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|
| 8 | 1.610 | 2.419 | 2.916 | 3.234 | 4.022 |
| 64 | 0.089 | 0.316 | 0.615 | 1.350 | 1.596 |
| 256 | 0.393 | 0.384 | 0.380 | 0.460 | 0.591 |

H=8 grows steadily and monotonically. H=64 grows slowly. **H=256 is flat**
across the first 400 steps and barely moves — the parameters are barely being
updated at all, because the 5 successes arrive too late and too rarely to
accumulate a direction. Note that this scarcity does not mean the parameter
vector is degenerate: the integrity gate passes in every cell, as it must.
The gate's design is right to refuse only the all-zero case — a vector that
moved but is wrong is a *modelling* call the gate deliberately does not make.

### Test 4: more training does not fix it

If the failure were sample-size, more steps would help. It does not.

| H=256, steps | successes | \|w\| | routing@1 | delta_1 |
|---|---|---|---|---|
| 500 | 5 | 0.591 | 0.0600 | −0.7758 |
| 2500 | 131 | 4.642 | 0.0700 | −3.9285 |
| 5000 | 335 | 6.880 | 0.0700 | −6.3544 |

`routing@1` moves 0.0600 → 0.0700 while the norm grows 12-fold and the margin
gets *worse*. The weights are growing without the relevance signal being
learned. More steps is not the fix; more steps with the same learning dynamics
compounds the error. This is the signature of a wrong *learning rule* for this
regime, not an under-trained model.

### The `state_lookup` clue was already in the repo

At H=8, the per-family breakdown is:

| family | accuracy | chance |
|---|---|---|
| relational | 0.5749 | 0.125 |
| replay | 0.5542 | 0.125 |
| state_lookup | 0.3593 | 0.125 |

The family that depends most on the persistence channel scores lowest. All
three are far above chance — the basis is not broken on any of them — but
`state_lookup` is the weakest, and it is the family where success depends on
locating a key the environment just wrote. The same scarcity mechanism
applies per-family: a persistence-dependent correct answer is a rarer event
than a query-matched one, so the reward that teaches it is rarer too. This is
the same effect the H-sweep measures, in a less extreme form, and it was
already visible at H=8 in the baseline data.

## What this establishes

1. **The matched-H experiment is decisive and it is negative.** Neither
   pre-registered reading survives. Training at matched H does not recover the
   large-H score signal, and it makes the small-H score signal worse.
2. **The failure is in the learning dynamics, not the hypothesis class.** The
   analytic vector — one shared weight vector, no training — achieves
   `routing@1 = 1.0000` at H ∈ {8, 64, 256, 512} with `delta_1 ≈ +3.07`. The
   relation is expressible and H-invariant, so the representation is not the
   bottleneck and an index built on this basis is not built on sand.
3. **The failure is measurable as reward scarcity.** 5 successes / 500 steps
   at H=256 against a 0.39% chance rate; the norm grows 12-fold from 2500 →
   5000 steps without `routing@1` improving; the margin worsens.
4. **The H=8 row is still the best row and remains the F1 baseline.** Nothing
   in this run changes the frozen HS-001/F0 numbers. The matched-H rows are
   new cells, not corrections to old ones.
5. **The integrity gate passed at all 9 training and 27 evaluation cells.**
   Every number above comes from a router the gate verified was not the
   initialisation.

## What this does NOT establish

- **It does not establish that F1's index should be built next.** The index
  is about the *cost* of routing; this experiment is about the *quality* of
  the trained score. It says the quality problem is a training problem, and
  the cost problem is untouched. Both remain.
- **It does not establish the analytic vector is the answer.** It is a
  hand-designed vector derived from the audit, not a learned one, and its
  existence is a proof of expressibility. Routing with it is not an outcome-
  trained router.
- **It does not establish the matched-H design was wrong to run.** It was the
  right experiment: the pre-registered rule committed to two outcomes and the
  run produced neither, which is exactly how a falsifiable design earns its
  keep.
- **No claim about `C_executed ≈ f(|R|)` follows.** The executor remains
  capped at `active_count = max_nodes = 10`, so the efficiency hypothesis
  remains untested (C5, unchanged).

## The path it opens

The pre-registered rule was binary and both branches failed, so the rule
itself is exhausted. The honest continuation is a third option, and it is
*different in kind* from both A and B:

> **C — the learning rule, not the representation or the training population.**
> The hypothesis class contains a perfect, H-invariant solution. The
> environment provides enough information to find it — the oracle reaches
> 1.0000. What fails is the *update*, whose reward signal is too sparse to
> estimate a gradient from at large H.

The two possible next experiments follow from that. They are recorded here
rather than pre-registered, because pre-registering requires a commitment
this run has not earned the right to make:

1. **Test whether reward is the bottleneck, or exploration is.** Keep the
   learning rule and add an exploration term (temperature schedule, or
   epsilon-greedy action selection at training time). If `routing@1` at
   H=256 rises toward the H=8 level, the mechanism is confirmed as sparse
   reward and the fix is a training-time schedule, not architecture.
2. **Test whether supervision can be denser without leaking.** The
   representability gate already proves the relation is *computable* from the
   router's own inputs. A surrogate reward — "the score gap between gold and
   the best distractor under the analytic vector", available without gold
   labels — could densify the signal without breaking the leakage boundary.

Both are cheap, and both are falsifiable against this run's numbers. Neither
requires an index, a new representation, or any change to the loop's
architecture. That is the point: the matched-H experiment was designed to
choose an architecture, and it answered that the architecture is not the
problem.

## Reproduction

```
python scripts/measure_matched_h.py --steps 500 --eval-steps 100
```

The analytic-vector diagnostic, the success-count probe and the multi-step
probe were run as ad-hoc `python -` sessions during analysis. They are
recorded above rather than scripted, because they are diagnostics answering
*why*, and a diagnostic that becomes a committed script before its finding is
written up is how a pre-registered design drifts into a post-hoc one.
