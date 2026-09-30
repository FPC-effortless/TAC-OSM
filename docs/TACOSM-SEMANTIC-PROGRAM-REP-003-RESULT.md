# TACOSM-SEMANTIC-PROGRAM-REP-003 RESULT

## Run provenance

- workflow run: `36653578237`
- job/check: `109692929987`
- code head: `d99009fb94d304e318187bd864447d49f1ca42dc`
- branch: `research/rep003-semantic-program-routing`
- full test suite: **823 passed**
- baseline current test count: **823**
- artifact: `TACOSM-SEMANTIC-PROGRAM-REP-003`
- artifact id: `11070988101`
- artifact zip SHA-256: `cf81f434b453c5e0370d3f689852dabc2ca19e2405fb5b80d63aeb4c78cf31a0`

## Protocol actually run

- 5 seeds: 0, 1, 2, 3, 4;
- 512 training episodes/seed;
- 256 held-out episodes/seed;
- 8 candidates;
- query width 5;
- candidate topology width 7;
- candidate encoder 7 -> 16 tanh -> 8;
- query encoder 5 -> 8;
- learning rate 0.01;
- margin 0.1;
- exact explicit graph executor.

## Structural controls

The semantic probe produced 8 unique raw candidate topologies and exactly one
candidate with the queried semantic signature. The query is therefore not an
edge mask and does not encode the target candidate index.

The analytic structural matcher achieved Top-1 = 1.000 on all five seeds
(1,280/1,280 held-out tasks) with mean hard-negative margin 4.0.

Exact graph execution matched each selected candidate's declared topology at
rate 1.000 for both learned and no-learning arms.

## Main result

| Seed | Learned B Top-1 | No-learning C Top-1 | B − C |
|---:|---:|---:|---:|
| 0 | 0.3438 | 0.0781 | +0.2656 |
| 1 | 0.0938 | 0.0938 | +0.0000 |
| 2 | 0.0000 | 0.0000 | +0.0000 |
| 3 | 0.5352 | 0.2969 | +0.2383 |
| 4 | 0.0313 | 0.0195 | +0.0117 |
| **Pooled** | **0.2008 (257/1280)** | **0.0977 (125/1280)** | **+0.1031** |

Chance with eight candidates is 0.1250. Thus the learned arm is above chance
when pooled; the no-learning control is below chance.

Mean target rank across seeds was 4.378 for B and 4.845 for C.

The learned arm exceeded the no-learning control in 3/5 seeds, tied in 2/5,
and did not show a uniform per-seed improvement.

## Interpretation

### What the result establishes

REP-003 provides bounded evidence that a learned router can extract some
semantic dependency information from explicit candidate executable topology
when the query does not directly contain the target edge mask.

This is stronger than REP-002's exact topology equality test because the query
and candidate are compared through a derived semantic property rather than
identical raw edge masks.

### What remains unresolved

The result is not a robust learned-routing result. Seed variance is large:
two seeds have no improvement over the no-learning arm, and two seeds have
very low absolute recall.

The analytic witness remains perfect while the learned arm remains weak. That
localizes the remaining problem to learning/objective dynamics rather than
basic representability of this synthetic relation.

The target semantic relation is also deliberately small and deterministic.
It does not test broad graph reasoning, natural-language program semantics, or
general compositional generalization.

## Cost

For the registered learned router:

`C_route = 8*5 + H*(16*7 + 8*16 + 8) MACs`

At H=8 this is 2,024 MACs per routing decision before nonlinear functions,
Python overhead, candidate materialization, verification, and execution.
The routing cost is still O(H).

REP-003 therefore adds no evidence for the original selective-computation
claim. It improves the understanding of the representation boundary only.

## Research decision

Do not tune the optimizer on this result and then reinterpret the same task as
confirmatory evidence. The next experiment should retain the explicit program
boundary but introduce persistent state into the semantic relation.

The most useful next test is a **persistent semantic program routing** task:

1. the candidate still exposes executable topology;
2. the query names a required semantic dependency;
3. the required dependency is partly specified by a persistent state item;
4. candidate action remains randomized and non-informative;
5. an analytic witness, learned arm, no-learning arm, and oracle are retained;
6. the experiment reports routing cost and state-address cost separately.

That would connect the identifiability result to the actual PNDS loop without
yet claiming large-scale retrieval or selective hardware execution.

## Frozen evidence

REP-001, REP-002, and prior C5 evidence remain separate historical artifacts.
No prior capability result was overwritten by this run.