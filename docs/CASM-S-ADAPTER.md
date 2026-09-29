# CASM-S adapter boundary

This document freezes the torch-free boundary between TAC-OSM and the CASM-S
implementation in cdl-attention-experiment.

## Transport object

A CASM-S Episode is translated into a TAC-OSM Structure whose spec is a
JSON-serializable mapping with:

- schema: casm-s/1
- kind: structural_graph
- active_count
- nodes
- candidate_edges
- inputs
- output

Each node carries index, op, depth, slot and arity.

Each candidate edge carries index, src, dst and port. The edge index is the
exact zero-based position in Episode.candidate_edges. It must remain stable:
CASM-S returns one gate per candidate edge in that same order.

The following Episode fields are deliberately not transported:

- true_edges / true_edge_set
- target
- input_values
- truth_table
- acceptable_actions

Runtime inputs are passed as the second argument to CasmExecuteFn rather than
being embedded in Structure.spec.

## Execution callback

The compute runner supplies a callable with this contract:

    execute_fn(spec, inputs) -> {
        "output": scalar,
        "gates": one-dimensional sequence of length len(candidate_edges),
        "node_values": one-dimensional sequence of length active_count,
        "provenance": string
    }

The callback may import torch. TAC-OSM's casm_adapter module must not.

The adapter accepts tensor-like scalar and vector values through item() and
tolist() duck typing, converts them to ordinary Python floats, and returns the
repository-native ExecutionResult.

For a structured CASM-S spec, cardinality mismatches fail closed before a
result can be interpreted as evidence.

## Runner bridge

The runner should construct the Episode using the real CASM-S generator,
call casm_structure_from_episode, and then inject CasmExecutorAdapter around
the real CASM-S execution callback.

The callback should operate on one episode at a time at this boundary. If the
underlying CASM-S module is batched, the callback is responsible for selecting
the requested batch row before returning output, gates and node_values.

This boundary does not claim CASM-S correctness, routing quality, or runtime
scaling. It only makes the CASM-S execution substrate replaceable inside the
TAC-OSM loop while preserving the anti-leakage and edge-indexing contracts.
