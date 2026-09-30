# TACOSM-C5-COVERAGE-FRONTIER-001

Status: PREREGISTERED.

## Question

For the registered persistent-state retrieval workload, what is the minimum fraction of states reranked that preserves at least 80% of the capability achieved by the exhaustive reference?

## Intervention

Jointly vary:

- factor size: 8, 16, 32;
- factor beam: 2, 4, 6, 8, 12;
- M: 128, 256, 512;
- H: 256;
- K: 32;
- five seeds.

The teacher, task construction, training-only codebook fitting, downstream executor, and query stream remain fixed.

## Capability criterion

Capability retention is defined before measurement as:

`selective_end_to_end_success / exhaustive_end_to_end_success`.

A configuration is eligible for the frontier only when capability retention is at least 0.80.

For each M, the primary frontier point is the eligible configuration with minimum `states_scored_over_M`. Ties are resolved by lower total MACs, then lower factor size, then lower factor beam.

## Why this follows the previous result

The granularity experiment established a direct occupancy tradeoff:

- factor size 8: approximately 56% of M scored;
- factor size 16: approximately 15% of M scored;
- factor size 32: approximately 5% of M scored.

It also showed that proposal coverage falls as granularity increases. The next scientific question is therefore not which factor size is “best,” but how much beam is required to recover capability at each granularity.

## Interpretation boundary

This experiment can establish a capability-constrained frontier on the registered workload. It cannot establish a universal optimum, universal semantic addressing, universal sublinear retrieval, or hardware wall-clock speedup.

## Integrity

The protocol is machine-readable in `contracts/TACOSM-C5-COVERAGE-FRONTIER-001.json`. The full measurement is gated by the commit marker `[run-c5-frontier-full]`.

## Execution trigger

This commit carries the CI marker `[run-c5-frontier-full]` only to start the registered measurement; the protocol above is unchanged.
