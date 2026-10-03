# Scientific Audit — TACOSM-GRAPH-CASM-IDENTIFIABILITY-011

## Purpose

G-CASM-010 produced a very low learned-routing signal even when executable
program wiring was public. Before interpreting that as a router failure, this
experiment audits whether the four public support rows determine a unique
target within the actual candidate pool.

This experiment does not train a router and cannot establish a capability
result. It is a benchmark-information audit.

## Prespecified warning condition

The primary warning condition is support_size=4 with target-unique fraction
materially below 1 and/or repeated support-consistent candidate counts above 1.

The support-size sweep is fixed before measurement at 4, 8, 12, and 16 rows.
The candidate pool is identical across those four conditions.

## Exact semantic addressing control

For every candidate, the pinned executable graph is evaluated on all 16 input
assignments and its output is inserted into an inverted index keyed by
(input assignment, output bit).

A query intersects the postings for the public support rows. The result is
required to equal a complete linear scan of the candidate pool.

The full graph truth table is an index-build representation, not a target-label
oracle: every candidate is treated identically, and the target truth table is
never passed separately to the query operation.

## Cost accounting

The artifact reports:

- exact graph execution work used to build the behavior index;
- posting entries inspected during each query;
- intersection operations;
- support-consistent candidate count.

These are mechanism-level arithmetic proxies. They are not hardware wall-clock
or end-to-end online-memory economics.

## Integrity

No learned model is used. There is no hyperparameter search. The candidate
pool is fixed across support sizes. Candidate truth signatures must be unique.
The target index is used only for evaluator diagnostics, never by the query.

A result can invalidate the public-support benchmark interpretation without
invalidating G-CASM-010's software integrity or exact-executor result.
