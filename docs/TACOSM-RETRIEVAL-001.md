# TACOSM-RETRIEVAL-001

**Status:** F0 run. **F1 is built and pre-amended; no confirmatory run has been
made.** Arms C and D are implemented and the index exists
(`src/tac_osm/retrieval.py`, `scripts/measure_retrieval.py`), but the
suspension note below has been **superseded by M1.8's top-K evidence**, which
is what re-queued it. Amendment A1 is on the contract
(`contracts/TACOSM-RETRIEVAL-001.json`) and is recorded in §F1-A1 below.

**Pre-registration.** Stages C and D are specified here *before* they are
built, and their specification is contingent on F0's result — that is the
point of running F0 first. The four arms and the cost model below are
committed to now, before the index exists, so the index cannot be designed
after seeing what it has to beat.

---

## F0's open follow-up has been run, and it closed the question

The follow-up specified below — *"train at matched H and re-run F0"* — is
`docs/TACOSM-MATCHED-001.md`. It ran, and it refuted **both** branches of the
pre-committed decision rule:

| F0 result | consequence (pre-committed) |
|---|---|
| recall@K saturates well at large H | F1: the representation is adequate, build the index |
| recall@K saturates badly at large H | the problem is upstream of search; change the representation |

Neither happened. Matched-H training does not recover the large-H score
signal; it is the **worst row at every evaluation population, including its
own**. And the representation is not inadequate — the analytic vector
(`gated_agreement = 1.0`, `slot_gated = 2.0`, the §34 gate's own reference)
reaches `routing@1 = 1.0000` at H ∈ {8, 64, 256, 512} with `delta_1 ≈ +3.07`,
because `basis_size` never depends on the candidate count.

The failure is in the **learning dynamics**, and it is measurable as reward
scarcity: 5 successes in 500 training steps at H=256 (1%, against a 0.39%
chance rate), a weight norm flat across 400 steps, and `routing@1` unchanged
at 0.0700 from 2500 → 5000 steps while the norm grows 12-fold and the margin
*worsens*. REINFORCE's `(1 − p_selected)` multiplier is squeezed from both
ends when successes are rare and the softmax over H candidates is flat.

**Why F1 is suspended rather than abandoned.** F1's hypothesis is a
*cost-quality trade*: can routing computation be reduced while preserving the
relevant candidate? The matched-H result says the quality half of that trade
is not a representation property, so an index built now would be measured
against a baseline whose weakness is a training artefact, not a property of
the basis. That is not a sound measurement of an index, in either direction:
a bad baseline makes a mediocre index look good. F1's specification below is
kept intact and un-amended, because it was pre-registered and a
pre-registration is not revised to fit a result. It is re-queued, not
withdrawn.

**What runs first instead.** The learning rule, not the architecture:

1. exploration vs sparse reward (a temperature schedule or epsilon-greedy
   action selection at training time); and
2. denser supervision that does not break the leakage boundary — the
   representability gate already proves the relation is computable from the
   router's own inputs, so a surrogate reward is available without gold
   labels.

Neither requires an index, a new representation, or any change to the loop's
architecture. That is the finding: the matched-H experiment was designed to
choose an architecture, and it answered that the architecture is not the
problem. See `docs/TACOSM-MATCHED-001.md`.

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

**Built and pre-amended; no confirmatory run.** Amendment A1, below, changed
the primary endpoint's definition **before any run**, which is the only way an
amendment can differ from a revision.

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

### F1-A1 — the amendment to the primary endpoint

The pre-registered primary endpoint `recall@K` was defined as
`P(gold ∈ top-K by score)` — a full-population quantity, measured the same way
in every arm. Under that definition every arm reports the same number for a
given `(H, K)`, so no branch of the decision rule could have fired on any data.
The decision rule was registered against a primary that could not
discriminate, and the instrument found it, not a reading of the contract.

**The amended definition.** `recall@K` is now
`P(gold ∈ the arm's retained set *and* is the argmax over that set)`. The
primary is measured over each arm's *own* retained set, so the arms can differ.
Arm B's retained set is the scorer's own top-K, not the full population; arms
C and D keep what the index retained.

**What is recorded.** The machine-readable contract carries the amendment with
all seven fields — the old definition, the new one, the rationale, where it was
discovered, what it affected, and the statement that **no confirmatory run was
made under the previous version**. Nothing about the original definition is
called wrong; it measured a different question, and the question it measured
was not the one the decision rule reads. A reader who disagrees with the
amendment can still see the design they were promised, because the old
definition is a field and not an overwritten value.

**What the dry run found about the index itself.** The registered rule is
recorded, not tuned. The index scores with an unlearned agreement rule over
two terms — agreement with the written state vector, and agreement with the
public query bits — and the term that separates gold depends on the family:

| weights | relational | state_lookup | replay |
|---|---|---|---|
| query-only | **1.00** | 0.00 | 0.33 |
| state-only | 0.00 | **1.00** | **1.00** |
| both (registered default) | **1.00** | **1.00** | **0.375** |

`relational` publishes its target as the query bits, so the query term
separates there and the state term is inert. The persistence families hide the
target in state, where the query bits are actively misleading: the
query-deceptive distractors agree with the query off the marked positions and
violate the relation on them, so gold's public agreement is *below* the best
distractor's. The registered default sum separates two of the three families
and **partially cancels on `replay`** — the unique-argmax rate there is 0.375,
and retention at large H falls below the state-only index's.

That is not a defect and not a tuning question. It is why the C-vs-B
comparison is reported in both directions and why the arm-D ceiling is
*measured* rather than assumed. **The registered default is unchanged**: a
weight chosen to fix `replay` would be chosen after seeing the dry run, and
the index is registered at these weights. All of this is held in place by
`tests/test_retrieval.py`.

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
| `recall@K` | `P(gold ∈ the arm's retained set and argmax over it)` — the primary, as amended by F1-A1 |
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
