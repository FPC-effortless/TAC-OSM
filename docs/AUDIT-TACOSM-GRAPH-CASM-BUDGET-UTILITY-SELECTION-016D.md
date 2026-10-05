# G-CASM-016D — Budget-capped probe objective

Status: PREREGISTERED / IMPLEMENTATION.

016D directly tests the conceptual shift from Shannon information to the
terminal verifier's actual budget. Both arms use the same six-bit activation
trace and the same target-blind one-step action domain. Only the action
objective changes.

For a candidate partition with M hypotheses and terminal budget B:

U_B = sum_e min(B, |H_e|) / M.

The objective is the expected utility gain over the current no-probe terminal
baseline, divided by acquisition work.

The primary question is whether this changes downstream hypothesis resolution.
Because G-CASM-015 already shows the trace partition is near its finite-domain
collision floor, the preregistered hypothesis explicitly allows a near-zero
effect.

No submodularity claim is made here. Adaptive submodularity is a separate
property requiring proof under the exact observation model.

## Confirmatory trigger
[run-graph-casm-budget-utility-selection-016D-full]

## Confirmed disposition

Full confirmatory run 37292064458 passed smoke, contract validation, the full
repository suite, focused tests, and measurement. Artifact 11337067153.

At M=512/B=8:
- mean U_8: 0.848046875 for information-per-work, 0.848828125 for
  budget-utility-per-work;
- seed-level U_8 differences: [0, 0.001953125, 0, 0.001953125, 0];
- primary mean difference: 0.00078125;
- 4000-resample seed bootstrap 95% CI: [0, 0.0015625];
- verified exact success: 0.83125 for both selectors.

Only seeds 1 and 3 selected different rows. On both, the U_B selector accepted
a slightly higher raw U_8 partition at a slightly lower information-per-work
score; the downstream exact-success rate did not change.

Therefore 016D does not provide evidence that replacing Shannon information with
one-step U_B improves verified capability on this workload. It does establish
that the two objectives are nearly aligned when the six-bit trace is already
close to its partition ceiling.

This result does not reject multi-step budget-aware planning. The separately
verified non-submodularity counterexample means one-step U_B optimization can
have complementarity failures, so a future planner may require lookahead or a
learned value function rather than relying on myopic U_B greediness.
