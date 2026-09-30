# TACOSM-C5-FRONTIER-REPLICATION-001

Status: PREREGISTERED.

## Question

Does the accepted C5 configuration `(factor_size=16, factor_beam=6)` reproduce the preregistered 0.80 capability-retention floor across M={128,256,512} on independent seeds 5–9, and does it remain the minimum-compute fixed configuration among the same registered 14-configuration search surface?

## Relationship to C5-004

`TACOSM-C5-COVERAGE-FRONTIER-004` produced the first valid C5 frontier on seeds 0–4. This replication changes only the seed set to the independent set 5–9. Model, task, index, factor configurations, capability threshold, evaluation count, and executor remain fixed.

The replication does not retune the configuration. It tests the accepted `(16,6)` result as a fixed hypothesis.

## Registered protocol

- M = {128, 256, 512};
- H = 256;
- K = 32;
- 14 factor-size/beam configurations identical to C5-004;
- seeds = {5, 6, 7, 8, 9};
- 100 evaluation steps per seed/configuration/M;
- capability-retention floor = 0.80.

## Primary replication criteria

1. `(16,6)` must have pooled capability retention >= 0.80 at every M.
2. `(16,6)` must be the minimum mean `states_scored_over_M` among fixed configurations eligible at all three M levels under the same tie-break order as C5-004.

The runner reports these criteria explicitly and does not modify the search surface based on the observed outcome.

## Validity controls

The state-distinct executor and exact count-conserving pooled aggregation are inherited directly from the validated C5-004 runner. The runner retains the executor distinctness audit, exact seed-cell count conservation, and evaluation-count invariants.

## Interpretation boundary

A successful replication strengthens confidence in the bounded C5 retrieval/computation result on this synthetic workload. It does not establish sublinear asymptotic scaling, universal semantic retrieval, universal optimality of `(16,6)`, or hardware wall-clock speedup.

The accepted C5-004 result remains the primary evidence record; this replication is an independent-seed reproducibility test.
