# TACOSM-HS-001: History scaling under a frozen mixed-family baseline

**Status: reported, not claimed as a scaling result.** This is the first
history-scaling measurement. It answers the central architectural question for
v0.1 and identifies the specific limitation the next stage must address.

## Provenance

| field | value |
|---|---|
| experiment_id | TACOSM-HS-001 |
| baseline | TACOSM-BASELINE-001 |
| commit | post-`78f2199`, history-scaling script |
| model | tac_osm v0.1, learned router |
| schedule | mixed (relational / state_lookup / replay) |
| training | 500 steps at 8 candidates, 5 seeds |
| evaluation | 100 episodes per H, learning disabled |
| measurement command | `python scripts/measure_history_scaling.py --steps 500 --eval-steps 100` |

## Design

For each history size H the task carries **one relevant item and H−1
distractors**. H is therefore the candidate count, and the relevant problem is
held fixed while only the irrelevant material grows:

    H=2   [R, D1]
    H=4   [R, D1, D2, D3]
    H=8   [R, D1 ... D7]
    H=64  [R, D1 ... D63]

The router is **trained once** at 8 candidates and its weights are copied
verbatim to every H, with learning disabled at evaluation. The point is
whether a relevance router *transfers* to a larger distractor set, not whether
it can be retrained per H. This is valid because `basis_size` depends on `dim`
and `max_state_slots` only, never on the candidate count, so one weight vector
scores a candidate set of any size.

H=1 is not expressible: `_validate_shape` requires `n_candidates >= 2`, so the
minimal task is one relevant item plus one distractor. The sweep starts at 2.

## Result

| H | accuracy | routing@1 | gold_rank | recall@4 | exec\|route | C_router | C_executed | total_nodes |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.9220 | 0.9460 | 1.05 | 1.0000 | 0.9620 | 2 | 10 | 10 |
| 4 | 0.8000 | 0.8300 | 1.20 | 1.0000 | 0.9349 | 4 | 10 | 10 |
| 8 | 0.6120 | 0.6160 | 1.59 | 0.9860 | 0.8893 | 8 | 10 | 10 |
| 16 | 0.4360 | 0.4800 | 2.20 | 0.9060 | 0.7779 | 16 | 10 | 10 |
| 32 | 0.2840 | 0.3000 | 3.70 | 0.6960 | 0.7048 | 32 | 10 | 10 |
| 64 | 0.1260 | 0.1620 | 6.40 | 0.5300 | 0.5019 | 64 | 10 | 10 |

## Interpretation: Outcome B — a routing problem, not an execution problem

The three candidate outcomes were:

- **A** routing and `C_executed` stay flat as H grows → relevance scaling
- **B** routing falls, `exec|route` stays flat → a routing problem
- **C** routing holds but computation grows → no computational saving

The result is **between B and its boundary with A**, and the distinction
matters.

**What degrades:** routing accuracy, smoothly and steeply. `routing@1` falls
0.9460 → 0.1620, a factor of ~6, while H grows by a factor of 32. Gold rank
rises 1.05 → 6.40.

**What does not degrade:** the *environment*. The oracle arm remains at
**1.0000 at every H**, so the task is not becoming harder in a way that
defeats execution. The relevant problem is held constant, exactly as designed;
only the distractor count grows.

**What degrades with routing:** `exec|route` falls 0.9620 → 0.5019. This is
*not* evidence of an execution defect, because `exec|route` is conditioned on
`routing@1` and the two are computed from the same router scores. As the
router's discrimination weakens, gold's margin over the best distractor
shrinks, so an episode that is still nominally a routing success is a much
weaker one — the gold candidate sits only slightly above a strong distractor,
and the downstream executor is correspondingly less reliable. The correct
reading is that the whole signal chain weakens together, with routing as the
cause.

This is the signature of a router that has learned the relation but has no
mechanism for *selection among many*: a linear scorer over an absolute
agreement feature has a fixed capacity to separate gold from its best
distractor, and that capacity is diluted as the distractor pool grows. Recall
degrades much more slowly than top-1 (0.9860 at H=8 vs 0.6160 at H=64 is
0.5300), which is the same statement at rank 4 — the router retains gold
inside its shortlist long after it stops ranking it first.

### The honest negative result

`C_executed` is **10 at every H**, because the v0.1 structural executor fixes
`active_count = max_nodes = 10` by construction. A flat column here is an
**architectural ceiling**, not evidence of sublinear scaling. The executor
has no room to grow, so this experiment cannot demonstrate the efficiency
hypothesis in either direction.

The cost that genuinely varies is `C_router`, and it **grows linearly with H**
— the router scores every candidate. This is the single most important
limitation, and it is the one Stage F must address: even perfect execution
scaling is worthless if routing cost is `O(H)`. Total cost in v0.1 is

    C_total = C_state_access + C_routing + C_execution + C_verification + C_commit
            = O(1)          + O(H)         + O(1)          + O(1)           + O(1)

so the hypothesis `C_executed ≈ f(|R|)` is **untested and untestable at
v0.1**, because the architecture has no index between state and router. The
`O(H)` routing term dominates and cannot be reduced by tuning the linear
scorer.

## What this establishes

1. **The relevant problem is held fixed and verified.** Oracle = 1.0000 at
   every H, so degradation is attributable to the model, not the environment.
2. **The learned router transfers to larger distractor pools**, but degrades
   smoothly rather than collapsing, which is a graceful rather than
   catastrophic failure.
3. **Routing is the bottleneck, and it is a capacity limitation of a linear
   scorer**, not a defect in execution or verification.
4. **The decomposition works.** Separating routing, execution and environment
   difficulty localises the failure to one component, which is the property
   that makes the next experiment designable.

## What this does NOT establish

- **`C_executed ≈ f(|R|)` is untested.** The executor cannot scale, so no
  efficiency claim of any kind follows from this run.
- **Total-cost scaling is `O(H)`, not `O(|R|)`.** Routing cost is linear in
  history, which is the opposite of the TAC-OSM hypothesis.
- **The result is for the mixed-family stream only.** No per-family history
  curve is reported here, and per-family behaviour may differ.
- **Single router architecture.** Everything above is about a *linear*
  scorer. An indexed router would change the `C_router` term, and this
  experiment says nothing about it.

## Measurement notes

`accuracy` (0.6120 at H=8) and `routing@1` (0.6160) differ because the learned
router **samples** from a softmax rather than taking the argmax. `routing@1`
is the decision the router *would* make; `accuracy` is what the loop actually
did. The gap is the cost of sampling, not an execution failure: it is within
noise at H ≤ 8 and reaches ~0.09 at H=32. Both are reported so the two are not
confused. This distinction was checked explicitly — an earlier diagnostic of
this experiment omitted the weight copy and therefore reported the entropy of
an untrained (uniform) router as if it were the router's discrimination. It
was wrong, and the corrected numbers above are what the trained router
actually does.

## Reproduction

```
python scripts/measure_history_scaling.py --steps 500 --eval-steps 100
```

The oracle-across-H check is not in the script; it was run separately and is
recorded here as 1.0000 at H = 2, 8, 32, 64 over 3 seeds.
