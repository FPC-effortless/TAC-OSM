# TACOSM-GRAPH-CASM-SELECTIVE-010

## Purpose

This is the direct continuation after the executor-identifiability problem discovered in M2-009.

M2-009 asked the pinned CASM-S gate to infer hidden parent wiring from node-local descriptors. The pinned gate cannot distinguish programs that share those descriptors but differ in hidden wiring. GRAPH-CASM-010 therefore makes the executable program graph part of the public candidate representation and isolates the remaining question: can learned routing reduce the amount of executable computation needed to find the task program?

## Experimental decomposition

The executor control is exact topological Boolean execution over the candidate public graph. It is deterministic and is checked against complete 16-row truth tables before routing results are accepted.

The graph representation contains per-node operation, depth, position, arity, existence, and the complete public wiring encoded in a fixed 200-bit edge field.

The matched summary representation contains the same node-local field but has the 200-bit wiring field removed. Both arms use the same candidate feature dimension, candidate projection, query tower, optimizer, support examples, budgets, seeds, and evaluation construction.

## Selective execution

For each task, four input/output examples are visible to routing. Twelve disjoint truth-table rows are verifier-only.

The router ranks M candidate programs. A fixed budget executes only the top B candidates. Adaptive execution evaluates those candidates sequentially until one passes all verifier rows.

Execution work is counted from actual graph operations: edge traversals plus non-input node operations. The exhaustive control executes all M candidates.

## Interpretation

A successful primary cell establishes only bounded evidence for capability-constrained selective execution when the executable candidate graph is public.

It does not establish hidden-wiring inference, asymptotic sublinear scaling, hardware speedup, or language-level intelligence.

## Checkpointing

The runner writes a result checkpoint after each seed/M block. The workflow uploads the current artifact with if: always(), so a timeout preserves the measurements completed before cancellation.

This experiment has a six-hour workflow ceiling because the preceding M2 run demonstrated that the original two-hour ceiling was undersized. Timeout sizing is an execution concern, not a scientific endpoint.
