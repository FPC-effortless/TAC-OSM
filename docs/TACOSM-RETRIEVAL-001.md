# TACOSM-RETRIEVAL-001

**Status:** F0 run; F1, C and D **not** started.

**Pre-registration.** Stages C and D are specified here *before* they are
built, and their specification is contingent on F0's result — that is the
point of running F0 first. The four arms and the cost model below are
committed to now, before the index exists, so the index cannot be designed
after seeing what it has to beat.

---

## The question

> Can an indexing/retrieval layer reduce the `O(H)` routing cost to
> approximately `O(K)` while preserving the useful relevance signal already
> present in the existing router?

TACOSM-HS-001 established that the bottleneck at large H is relevance routing
(C6), and that the top-K signal survives the collapse of top-1 precision (C7).
The natural but wrong next step is to build the index. A cheap index around an
inadequate representation retrieves the wrong candidates *faster*, and the
result would read as "indexing works" while measuring nothing.

So F0 comes first: does the relevance information survive in the score
distribution at all?

---

## F0 — the retrieval ceiling

**Done.** `scripts/measure_retrieval_ceiling.py`.

The existing router scores all H candidates exhaustively. Instead of sampling
or taking an argmax, retain the top-K by score and ask whether gold is among
them:

```
H → score all H candidates → keep top-K → is gold ∈ top-K?
```

Nothing new is built. The measurement uses the router as it exists, so the
result is a property of the current scoring representation rather than of any
new mechanism.

### Held fixed

As in TACOSM-HS-001, the *relevant problem* is fixed and only the irrelevant
material grows: each task carries exactly one relevant item and `H-1`
distractors. Gold is built once per seed and the distractor set is extended,
not redrawn, so difficulty is not confounded with a new task distribution at
every H.

The router is trained once at 8 candidates and its weights loaded into each
evaluation router with `load_weights`, with learning disabled. The
model-state integrity gate runs before any measurement (C4) — the error that
produced a false HS-001 conclusion is exactly a weights-never-loaded
evaluation, and the gate is what makes that failure loud instead of
plausible.

### Result

| H | rec@1 | rec@2 | rec@4 | rec@8 | rec@16 |
|---|---|---|---|---|---|
| 8 | 0.616 | 0.854 | 0.986 | — | — |
| 32 | 0.300 | 0.490 | 0.696 | 0.904 | 0.998 |
| 64 | 0.162 | 0.306 | 0.530 | 0.748 | 0.920 |
| 128 | 0.096 | 0.172 | 0.314 | 0.550 | 0.762 |
| 256 | 0.054 | 0.088 | 0.174 | 0.336 | 0.550 |

5 seeds, 100 evaluation steps each, trained 500 steps at 8 candidates. The
oracle arm is 1.0000 at every H, so the environment is not degrading. A budget
`K ≥ H` is not a retrieval budget and is dropped rather than clamped.

### Reading

The curve at `H ≤ 64` is the shape that makes search the bottleneck: at H=64,
recall climbs 0.162 → 0.530 → 0.748 → 0.920 across K = 1, 4, 8, 16. The scoring
representation holds substantial information about where gold is, long after
it stops being first. At `H ≥ 128` it does not: rec@16 falls to 0.762 at
H=128 and 0.550 at H=256.

**This is a mixed result and it is reported as one.** Outcome A
(representation adequate → build the index) holds at `H ≤ 64`; outcome B
(problem upstream of search → change the representation) is not ruled out at
`H ≥ 128`.

Two readings survive the data, and F0 does not separate them:

1. the representation degrades as the candidate population grows;
2. the router was trained at 8 candidates, so the sweep measures *transfer*,
   and the large-H degradation is a training-horizon artefact.

Reading 2 is testable without any new architecture: train at matched H and
re-run F0. If recall@16 at H=256 recovers, the representation was fine and the
limit was training; if it does not, the representation itself is the
bottleneck and F1's index would be built on sand. **This is F0's open
follow-up and it is the next thing to run**, because it decides whether F1 is
worth building at all.

The pre-committed decision rule, so it cannot be chosen after the fact:

| F0 result | Consequence |
|---|---|
| recall@K saturates well at large H | F1: the representation is adequate, build the index |
| recall@K saturates badly at large H | the problem is upstream of search; change state/query/candidate/scoring representation |

At `H ≤ 64` the first branch holds. At `H ≥ 128` the second is open.

---

## F1 — efficient retrieval

**Not started. Contingent on F0.**

```
H → cheap index → K candidates → existing scorer → R
```

The index narrows `H` to `K ≪ H` *before* scoring, so the existing scorer is
reused unchanged — the point being to preserve F0's retrieval quality while
changing the cost:

```
C_router: O(H) → O(K)
```

F1 is only worth building if F0 says the scoring representation is adequate.
If it is not, the index returns the wrong candidates faster, and the fix is
upstream.

### What F1 must demonstrate

Not "the index works". The claim F1 tests is a *cost-quality trade*: at
matched H and matched K, the indexed arm's recall@K must be close to the
exhaustive arm's, at a routing cost that grows with K rather than H. A small
cost reduction with a large quality loss is a negative result, and it is
reported as one.

### Cost model

Both stages report the same cost terms, along the same axis, so a
capability-versus-computation curve is a plot of one table rather than a
narrative:

| term | definition | varies with |
|---|---|---|
| `candidates_inspected` | candidates the scorer evaluated | H (exhaustive) or K (indexed) |
| `router_operations` | scoring calls made | K |
| `index_operations` | index probes, zero in F0 and in arms A/B | index depth |
| `executor_nodes` | nodes the executor evaluates | `active_count` |
| `verifier_operations` | verification calls | 1 per step |
| `wall_clock` | measured, not derived | all of the above |

`executor_nodes` is reported because C5 is untested and the column must not be
read as a scaling result: in v0.1 it is fixed at `max_nodes = 10` by the
executor, so it is an architectural constant, not a measurement.

---

## The four arms

A and B are F0; C and D are F1. Specified now, before the index exists.

### A. exhaustive + top-1

The current architecture, unchanged. Scores all H candidates, selects the
argmax. This is the baseline every other arm is measured against, and it is
the arm TACOSM-HS-001 already characterised.

### B. exhaustive + top-K

The retrieval ceiling, F0. Scores all H candidates, retains the top-K, and
reports `P(gold ∈ top-K)` at every K. Costs the same as A — it still scores
everything — and answers what quality is *available* if a retrieval boundary
existed.

### C. indexed + top-K

**The proposed architecture, and the reason F1 exists.**

```
H → cheap index → K ≪ H → existing scorer → R
```

The index is the only new mechanism, and it is measured against B at matched
K. The comparison C vs B isolates the *cost* of indexing: same budget, same
scorer, one arm with an index and one without. If C's recall@K is materially
below B's, the index is discarding the gold candidate, and the loss is the
index's and not the scorer's.

### D. oracle index + top-K

The index cost/quality ceiling: the index is told where gold is, so it
retrieves the right neighbourhood by construction. C vs D isolates how much
of the retrieval loss is the *index* and how much is the scorer's ranking
within the retrieved set. If D is barely better than C, the index is not the
limiting factor and the scorer is; if D is much better, the index is throwing
away gold the scorer would have ranked.

D is a diagnostic, not a candidate architecture. It reads information no
runtime component may have, exactly like `OracleRouter`, and for the same
reason: to measure headroom rather than to occupy it.

---

## Metrics

### Capability

| metric | definition |
|---|---|
| `exact_success` | the loop's task success — the thing being improved |
| `routing@1` | `P(gold is the argmax of the scores)` |
| `recall@K` | `P(gold ∈ top-K by score)` |
| `gold_rank` | mean rank of gold under the score |

`routing@1` and `exact_success` differ because the learned router samples
from a softmax rather than taking the argmax, and the gap is the cost of
sampling, not an execution failure. Both are reported so the two are not
confused — the error that motivated the C4 gate was reading one as the other.

### Cost

The six terms above, reported per `(arm, H, K)`.

### Failure taxonomy

Every failed step is attributed to exactly one category, so a drop in
`exact_success` can be read as a *kind* of failure rather than a quantity:

| category | meaning |
|---|---|
| `retrieval miss` | gold was not in the retrieved set; the index or the budget lost it |
| `ranking miss` | gold was retrieved but not selected; the scorer ranked it below a distractor |
| `execution failure` | the correct candidate was selected and the computation failed |
| `verification failure` | the computation succeeded and the verifier rejected it |

The taxonomy is what makes the C-vs-B comparison interpretable. Without it, a
quality loss in C could be an index that discards gold (`retrieval miss`) or a
scorer that mis-ranks within the retrieved set (`ranking miss`) — different
failures with the same headline number, and only the first is an argument
against the index.

---

## What this experiment does not claim

* It does not test `C_executed ≈ f(|R|)` (C5). The v0.1 executor fixes
  `active_count = max_nodes`, so there is no execution-side scaling to
  measure. That claim becomes testable when a retrieval boundary sits between
  routing and execution, which is F1's structural contribution and not this
  experiment's measurement.
* It does not claim the router is a good retrieval mechanism. C7 is stated at
  `H ≤ 64` and no further, because that is where it is supported.
* It does not compare against CDL, whose per-candidate LM forward pass is the
  cost the architecture exists to avoid. `CDLTeacher` remains a diagnostic
  bound, not an arm.
