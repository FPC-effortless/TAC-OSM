# TACOSM-M2-CASM-SELECTIVE-009

## Scientific objective

Test the unresolved PLM/C5 boundary with an actual CASM-S executor:

`persistent task state -> structural representation -> selective candidate admission -> CASM-S execution -> outcome -> verifier`.

The substantive question is whether selecting a small subset of structurally relevant programs reduces **actual executed structural work** while preserving task capability relative to exhaustive execution.

This is not a test of hardware speedup or asymptotic sublinear scaling.

## Executor

Pinned external source: `FPC-effortless/cdl-attention-experiment` commit `c31554413301e3c9d3e6b3f8c8c6be572a74a748`.

The pinned CASM-S structural encoder uses public node-local descriptors and does not consume the oracle true-edge set. Execution is single-pass topological DAG execution.

## Task construction

Each task is a public set of four input/output examples generated from a target Boolean program. Candidate populations contain the target plus structurally distinct decoys with unique truth tables and unique structural programs. The generator rejects candidate pools in which a decoy agrees with the target on every public example, so semantic success is well-defined without checking hidden target identity.

The evaluator knows the target only for diagnostics. The router does not.

Primary evaluation uses unseen exact programs from the same generator distribution. Secondary evaluation excludes the `NOT -> XOR` role pair from training and evaluates on programs containing that pair.

## Routing arms

The primary structural arm freezes the trained CASM-S structural representation and learns a query-to-structure cross-modal retrieval tower.

The matched summary arm replaces the CASM representation with a fixed node-local structural summary while keeping the trainable tower dimensions and training schedule identical.

Random selection is a chance control. Exhaustive CASM-S execution is the capability ceiling and cost baseline.

## Execution accounting

For every executed program, record:

- gate evaluations;
- edge message aggregation operations;
- active non-input node operations;
- total structural work units;
- wall-clock;
- input-example count.

The selective arm reports actual executed candidates, not merely the admission budget.

Routing work is reported separately because a dense router can still perform O(M) query-time scoring even when CASM execution is O(B).

## Capability definition

A selected candidate is semantically successful when CASM-S execution matches every public input/output example.

At each M and budget B:

`routing_recall@B = P(target candidate admitted)`

and

`semantic_success@B = P(at least one executed candidate satisfies all public examples)`.

The primary capability-retention ratio is selective semantic success divided by exhaustive semantic success, capped at 1 for reporting.

No cell is promoted when exhaustive CASM capability is below the registered 0.80 floor.

## Interpretation boundaries

A successful result would establish selective execution savings under the registered synthetic structural benchmark.

It would not establish:

- asymptotic sublinear routing;
- general language intelligence;
- superiority to Transformers or SSMs;
- universal program retrieval;
- end-to-end wall-clock advantage;
- autonomy of operator discovery.

A failure can be localized to representation/routing, CASM execution, or verifier/task-interface limitations.



## Verified persistent experience (secondary)

Task state is addressed through an opaque task ID. The router resolves only the public input/output examples associated with that address.

After execution, only a candidate that satisfies all public task examples under the actual CASM-S executor is eligible for a durable experience write. The stored experience identifies the verified candidate structure, not the evaluator's target ID.

The replay test inserts unrelated task states between an anchor task's first execution and replay. It measures whether verified experience changes Top-1 routing or reduces adaptive CASM execution work.

This is a persistence mechanism test, not a claim of generalization from memorized task identities.