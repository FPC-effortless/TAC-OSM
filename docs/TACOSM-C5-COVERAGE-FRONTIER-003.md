# TACOSM-C5-COVERAGE-FRONTIER-003

Status: PREREGISTERED.

## Question

For the registered persistent-state retrieval workload, what is the minimum fraction of states reranked that preserves at least 80% of exhaustive reference capability under a state-distinct downstream task?

## Lineage

`TACOSM-C5-COVERAGE-FRONTIER-001` was void because its downstream executor was not state-discriminating.

`TACOSM-C5-COVERAGE-FRONTIER-002` was void because its pooled capability field and eligibility flag were computed from inconsistent aggregation quantities.

This registration retains the same scientific search surface and corrects only the aggregation implementation. The voided experiments remain immutable evidence.

## Registered configurations

`(factor_size, factor_beam)` = (8,2), (8,4), (8,6), (8,8), (16,2), (16,4), (16,6), (16,8), (16,12), (32,2), (32,4), (32,6), (32,8), (32,12).

M = {128,256,512}; H = 256; K = 32; five seeds; 100 evaluation steps.

The teacher, query construction, training-only codebook fitting, product-key implementation, and downstream executor remain fixed.

## Capability aggregation

For each M/configuration, the runner records the aggregate number of successful selective executions, aggregate number of successful exhaustive executions, and total evaluation count.

The single stored capability measure is:

`capability_retention = min(1, selective_success_count / exhaustive_success_count)`.

The eligibility flag is computed directly from that same stored scalar. The runner asserts that the boolean flag and scalar threshold agree for every pooled row.

A configuration is eligible when capability retention is at least 0.80.

Per-M frontier selection minimizes `states_scored_over_M` among eligible configurations, with total MACs, factor size, and beam as tie-breakers.

A fixed-configuration frontier requires the same configuration to be eligible at all three M levels and minimizes mean `states_scored_over_M`.

## Validity controls

The downstream executor maps the selected state value to a deterministic state-code-dependent output. An explicit pre-measurement audit verifies distinct outputs for two target state codes and verifies fixed executor work/call counts.

The runner and contract are machine-readable and the full run is gated by `[run-c5-frontier3-full]`.

## Interpretation boundary

This experiment can establish a capability-constrained frontier on the registered synthetic workload. It cannot establish universal optimality, universal semantic addressing, universal sublinear retrieval, or hardware wall-clock speedup.

The exhaustive reference is itself reported because its absolute capability can degrade as M increases. A high retention ratio therefore means preservation of the reference, not high absolute task capability.
