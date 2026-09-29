# TACOSM-C5-001 — CASM-S selective structural execution

## Purpose

This is the confirmatory execution experiment that follows TACOSM-SELECTIVE-001. The earlier experiment established a real bounded candidate-retention boundary, but its executor ran once in both arms. That means its executor-invocation metric could not test structural compute scaling.

C5-001 changes only the execution question:

    H -> address -> retained set R -> CASM-S execution of every member of R -> select -> observe -> verify

The same CASM-S evaluator is used for exhaustive, exact-indexed and representation-addressed arms.

## CASM-S pin

The runner requires the external checkout to be exactly:

    FPC-effortless/cdl-attention-experiment
    c31554413301e3c9d3e6b3f8c8c6be572a74a748

A different checkout fails closed.

## CASM-S bridge

TAC-OSM relevance circuits are compiled to the CASM-S candidate substrate. The relevance program's true wiring is not transported. Instead, the bridge constructs the full upper-triangular candidate-edge substrate in the same destination -> port -> source ordering used by the CASM-S generator.

Runtime input values are passed separately to the external CASM-S module.

The transport therefore contains:

    nodes + candidate_edges + inputs + output + active_count

and does not contain:

    true_edges + target + input_values + truth_table + acceptable_actions

## Calibration

Because TAC-OSM's current benchmark asks CASM-S to execute a relevance circuit rather than the original random-DAG CASM Phase-1 task, the runner performs a separate bridge calibration before C5 evaluation.

The calibration trains CASM-S on 512 held-out-generation relevance circuits and checks 128 validation circuits. This calibration is not a capability result and is not mixed with the C5 task stream.

## Arms

**exhaustive** executes all H candidate structures.

**exact-indexed** builds the existing content-addressed equality index once per static population, then executes only the returned K candidates.

**representation-addressed** uses a fixed cosine-similarity baseline over the public query bit vector and candidate descriptor, retaining K candidates. It is explicitly not a learned semantic-addressing result.

For every arm the CASM model, graph compiler, runtime input path, output selection, and verifier are identical.

## Measured work

For each executed structure the runner records:

    structures_executed
    active_nodes
    candidate_edges
    gate_evaluations
    executed_structural_operations
    node_outputs

The aggregate is reported per query and across the cell.

The work units are derived from the actual CASM graph submitted to the external model, not from the Python candidate-count metric.

`casm_forward_batches` is reported separately because one CASM-S forward can process multiple retained structures. It must never be mistaken for the number of structures executed.

## Capability

After all retained structures have been executed, the candidate with the highest CASM-S output is selected, with deterministic first-index tie-breaking. The environment then reveals success. The verifier runs only after that observation and independently checks the public relation and computed output.

Thus a CASM-S execution failure can be localized as:

    wrong graph output
    -> capability loss or verifier mismatch

while an addressing failure is visible as:

    acceptable structure absent from retained set

No fallback from an empty retained set to H is permitted.

## Reproduction

    python scripts/measure_c5_casm.py --casm-root /path/to/cdl-attention-experiment --train-bridge --checkpoint-out /path/to/casm-s-c5.pt

For a frozen model after calibration:

    python scripts/measure_c5_casm.py --casm-root /path/to/cdl-attention-experiment --checkpoint /path/to/casm-s-c5.pt

The confirmatory run must use the pre-registered values in contracts/TACOSM-C5-001.json unless an explicit deviation is recorded.