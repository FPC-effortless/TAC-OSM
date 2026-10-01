# TACOSM-IDENTIFIABILITY-REP-002 RESULT

**Run status: COMPLETE**

- branch: `research/rep002-explicit-program-routing`
- code head used for the run: `33a3187c8de6d9c85bf9d43af61ae2cfdc75081e`
- workflow run: `36653246763`
- artifact: `TACOSM-IDENTIFIABILITY-REP-002`
- artifact id: `11071067689`
- artifact zip SHA-256: `85a3f5ef6ac35a56099c19d68e8281ea8c555d9479e904eefd75169d357bd5bd`
- full repository gate: **815 passed**
- baseline current test count: **815**

## Result

The diagnostic separates two questions that REP-001 left conflated.

| Measure | Result |
|---|---:|
| descriptor-only unique candidate observations | 1 / 8 |
| explicit-program unique topology observations | 8 / 8 |
| analytic A Top-1 | 1.000 per seed |
| analytic A hard-negative margin | 4.000 per seed |
| learned B pooled Top-1 | 142 / 640 = 0.2219 |
| no-learning C pooled Top-1 | 81 / 640 = 0.1266 |
| chance | 1 / 8 = 0.1250 |
| learned B mean target rank | 4.2203 |
| no-learning C mean target rank | 4.7047 |
| exact edge execution | 1.000 for B and C |
| oracle D selection success | 1.000 per seed |

Per-seed learned-minus-control Top-1 differences were:

`+0.2266, +0.1172, +0.0000, +0.0703, +0.0625`.

Thus B exceeded C in four of five seeds and tied in one. The pooled gain over
the no-learning control is 61 additional correct selections over 640 held-out
episodes.

## Interpretation

### 1. The candidate-observation collision is real in the registered control

All eight candidates share the same descriptor. Under the legacy descriptor
boundary, the router-visible feature row is identical for all eight candidates.
The explicit-program intervention exposes the candidate-owned executable edge
set and produces eight distinct topology observations.

This is a bounded identifiability result for the synthetic construction.

### 2. The new information is usable, but learning remains weak

The analytic identity witness is exact, establishing that the explicit topology
representation is sufficient for this task.

The learned arm reaches 0.2219 Top-1 against 0.1266 for the frozen random
control, so the new information is not completely useless to the registered
learner. However, 0.2219 remains only modestly above the 0.125 chance baseline,
and the per-seed variability is large.

Therefore REP-002 does **not** support the conclusion that the router problem is
solved. It supports a more precise decomposition:

`information available: yes`

`information exploitable by this learner: partially`

`robust high-quality routing: not established`

### 3. Execution is still not the bottleneck

Exact execution is 1.000 for the selected candidates, with exactly two active
edges per program and seven candidate substrate edges.

The remaining error is selection, not graph execution.

### 4. This does not yet validate the original C5 claim

The task deliberately makes the desired topology explicit in the query as a
seven-bit mask. That is appropriate for diagnosing the information boundary,
but it is easier than the intended persistent relevance problem.

REP-002 therefore should not be cited as evidence of:

- semantic routing;
- compositional program understanding;
- generalization across graph families;
- selective execution scaling;
- sublinear addressing/retrieval;
- PLM/PNDS validity.

## Research consequence

The next experiment should not tune the existing optimizer broadly.

The next useful intervention is a **semantic program-topology task** in which:

1. the candidate carries explicit executable topology;
2. the query states a semantic dependency/relevance requirement rather than the
   exact target edge mask;
3. an analytic structural matcher can establish representability;
4. learned routing is compared with no-learning and a topology-masked control;
5. the candidate population is scaled only after the fixed task passes.

That experiment would test whether explicit executable structure helps with
actual structural relevance rather than exact mask equality.

## Relation to external architecture literature

The result is directionally consistent with a broader design pattern: systems
often separate a cheap selection/addressing mechanism from expensive selected
computation. RETRO retrieves a subset of document chunks before cross-attention,
and Switch Transformer uses a router to select sparse experts rather than
activating every expert. citeturn278791academia3turn117616academia0

The important difference for TAC-OSM is the object being routed: this experiment
routes over **candidate executable programs**, so the representation of the
candidate's structure becomes part of the addressing problem itself. Program
and code representation research likewise finds that explicit data/control
structure can carry information not present in a flat token-local view.
GraphCodeBERT explicitly injects data-flow structure, while recent GraphAlignCoder
uses implementation/proof graphs to expose correctness structure. citeturn227480academia1turn227480academia0

## Frozen evidence

REP-001 and the prior C5 evidence were not rewritten or recomputed into this
result. This document records the separate REP-002 measurement only.
