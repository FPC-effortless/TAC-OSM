# TACOSM-C5-COVERAGE-FRONTIER-002

Status: PREREGISTERED.

## Question

For the registered persistent-state retrieval workload, what is the minimum fraction of states reranked that preserves at least 80% of the exhaustive reference capability when downstream success is state-distinct?

## Why this is a new experiment

The predecessor, `TACOSM-C5-COVERAGE-FRONTIER-001`, is VOID. Its downstream executor could return the same successful aggregate output for different persistent-state values, so end-to-end success did not discriminate correct state retrieval.

This registration preserves the predecessor search surface but changes the downstream task so that successful output is a deterministic function of the selected state value. The predecessor is not retroactively repaired.

## Registered intervention

- configurations: (8,2), (8,4), (8,6), (8,8), (16,2), (16,4), (16,6), (16,8), (16,12), (32,2), (32,4), (32,6), (32,8), (32,12);
- M: 128, 256, 512;
- H: 256;
- K: 32;
- five seeds;
- 100 evaluation steps per seed/M/configuration;
- capability-retention floor: 0.80.

The teacher, query construction, training-only codebook fitting, product-key implementation, factor codebooks, and task stream remain fixed.

## Capability criterion

For each pooled M/configuration cell:

`capability_retention = min(1, selective_end_to_end_success / exhaustive_end_to_end_success)`.

A configuration is eligible when pooled capability retention is at least 0.80.

For each M, the primary frontier point minimizes `states_scored_over_M` among eligible configurations, with total MACs then factor size and beam as tie-breakers.

A global frontier is also reported: configurations must satisfy the 0.80 floor at all three M levels, and the selected global point minimizes mean `states_scored_over_M`.

## Executor validity

The corrected executor adds a state-code-dependent contribution to every matching candidate. Two distinct target state codes therefore produce distinct expected outputs. The runner performs an explicit distinctness audit before measurement.

A wrong state cannot pass the end-to-end check merely because another state has the same output.

## Interpretation boundary

The experiment can establish a capability-constrained frontier on the registered synthetic workload. It cannot establish a universal optimum, universal semantic addressing, universal sublinear retrieval, or hardware wall-clock speedup.

The teacher's own exhaustive retrieval and capability quality remain the reference condition and are reported separately.

## Integrity

The machine-readable contract is `contracts/TACOSM-C5-COVERAGE-FRONTIER-002.json`.

The full measurement is triggered only by a commit containing `[run-c5-frontier2-full]`.

The voided predecessor remains unchanged and auditable.
