# TACOSM-C5-COVERAGE-FRONTIER-004

Status: PREREGISTERED.

## Question

For the registered persistent-state retrieval workload, what is the minimum fraction of states reranked that preserves at least 80% of exhaustive reference capability under a state-distinct downstream task?

## Lineage

`TACOSM-C5-COVERAGE-FRONTIER-001` is VOID because the downstream endpoint was non-discriminating.

`TACOSM-C5-COVERAGE-FRONTIER-002` is VOID because eligibility was computed from a different aggregation quantity than the reported capability field.

`TACOSM-C5-COVERAGE-FRONTIER-003` is VOID because averaged success counts were truncated with `int()` before ratio computation.

This registration preserves the same search surface and corrects the aggregation by summing exact integer success counts across seed cells.

## Registered configurations

`(factor_size, factor_beam)` = (8,2), (8,4), (8,6), (8,8), (16,2), (16,4), (16,6), (16,8), (16,12), (32,2), (32,4), (32,6), (32,8), (32,12).

M = {128,256,512}; H = 256; K = 32; five seeds; 100 evaluation steps.

## Capability aggregation

Each seed/configuration/M cell records integer selective and exhaustive success counts over exactly 100 evaluations.

For every pooled cell, the runner sums those integer counts across all five seeds. It separately sums the evaluation counts and asserts conservation.

`capability_retention = min(1, total_selective_successes / total_exhaustive_successes)`.

The same stored scalar is used for the 0.80 eligibility test. The runner asserts the threshold boolean and scalar agree.

Per-M frontier selection minimizes `states_scored_over_M` among eligible configurations, with total MACs, factor size, and beam as tie-breakers.

A fixed-configuration frontier requires eligibility at all three M levels and minimizes mean `states_scored_over_M`.

## Validity controls

The downstream executor is state-distinct: successful output is a deterministic function of the selected state code. The runner performs an explicit two-state distinctness audit before measurement.

The pooled evaluator requires exact count conservation and rejects truncated or mismatched counts before frontier selection.

## Interpretation boundary

This experiment can establish a capability-constrained frontier on the registered synthetic workload. It cannot establish universal optimality, universal semantic addressing, universal sublinear retrieval, or hardware wall-clock speedup.

The exhaustive reference is reported separately because its absolute capability may degrade as M increases. Retention therefore measures preservation of the reference, not high absolute task capability.

The full measurement is triggered only by `[run-c5-frontier4-full]`.


Registered full-measurement trigger: `[run-c5-frontier4-full]`.
