# AUDIT — TACOSM-GRAPH-CASM-ADAPTIVE-PROBE-014

**Status: PREREGISTERED / IMPLEMENTATION UNDER AUDIT**

## Scientific purpose

G-CASM-013 showed that exact public-behavior retrieval is not the limiting
operation in the four-bit domain. At small support sizes, multiple candidates
can remain behaviorally indistinguishable. 014 therefore tests the upstream
information-acquisition policy.

## Intervention

adaptive_minimax chooses the next unseen input row by minimizing the worst-case
size of the current compatible hypothesis set after the row's binary output is
revealed.

The choice is made before the target output for that row is read. The selector
receives candidate public truth tables and the current compatible set only.

fixed_random uses the exact deterministic row permutation from G-CASM-013.

The same target, seed, population prefix, task and candidate execution budgets
are used in both arms, giving a paired intervention.

## Primary endpoint

Frozen before measurement:

- K = 4 observations;
- B = 8 candidate executions;
- fixed exact verifier;
- seed-level mean of the paired adaptive-minus-fixed success difference;
- 4000 bootstrap resamples over the five seeds.

This is deliberately the same evidence budget that exposed the G-CASM-013
large-M ambiguity.

## Leakage boundary

Adaptive selection may use:
- public candidate truth tables;
- current compatible candidates;
- row identities not yet observed.

It may not use:
- target index or target identity;
- target output before the row is chosen;
- verifier labels before the probe action;
- any result from this confirmatory run.

## Heuristic interpretation

The selector is an exact finite-domain identity-blind heuristic, not a
learned policy. A positive result says that adaptive observation placement has
causal leverage on the registered workload. The next experiment, if warranted,
should learn the probe policy from training behavior while retaining this exact
selector as an upper bound.

A negative result means the exact identity-blind ceiling does not improve the
fixed four-row evidence budget. It does not prove that a different observation
space or representation cannot help.

## Execution accounting

Probe observations are reported separately from candidate verifier work. We do
not merge environment-query cost with graph execution work.

Fixed candidate-verifier accounting remains exact:

    rows_evaluated = selected_candidates * verifier_rows

All 16 behavior rows are partitioned into probe rows and held-out verifier rows;
no row serves both purposes in one query.

## Evidence inheritance

- 010: executable graph representation and exact execution.
- 011: behavioral identifiability diagnosis.
- 012: compatibility-conditioned scoring.
- 013: exact support retrieval and cached ranking.
- 014: adaptive evidence acquisition.

## Non-claims

This experiment does not establish:
- learned autonomous experiment design;
- generic active learning superiority;
- asymptotic sublinear retrieval;
- hardware speedup;
- language, image or audio capability.


## Premeasurement infrastructure note

The branch-local repository test count was synchronized to 880 after the five 014 regression tests were added. The frozen baseline value remains unchanged.


Confirmatory trigger record: [run-graph-casm-adaptive-probe-014-full]


Confirmatory trigger record 2: [run-graph-casm-adaptive-probe-014-full]; current measurement head includes the corrected smoke summarization grid and tuple-key interface.
