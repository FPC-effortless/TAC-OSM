# M2 Executor Ceiling Diagnostic 001

## Scientific status

This diagnostic is separate from and does not modify the frozen TACOSM-M2-CASM-SELECTIVE-009 protocol.

The M2 smoke artifact recorded zero exhaustive semantic success at the tested levels. Before allocating a multi-hour full run, the relevant question is whether the pinned CASM-S executor has enough information to represent the registered program family at all.

## Result

The pinned executor gate is computed from source-node structural embedding, destination-node structural embedding, and syntactic argument port. The true edge set is constructed by the model's `_edge_tensors()` method and returned as metadata, but it is not consumed by `CASMS.gate()` when producing gate logits.

The benchmark generator independently samples each node's true parent wiring. Consequently, two programs can have identical public node descriptors (operation, depth, position, arity, existence) while differing in true wiring and therefore differing in their Boolean truth tables.

The diagnostic constructs exactly such a pair:

- program A: AND(input0, input1) -> NOT;
- program B: AND(input2, input3) -> NOT;
- identical node descriptors and candidate-edge substrate;
- different hidden true wiring;
- different truth tables.

For the pinned CASM-S implementation, the structural encodings and gate values are identical for the two programs.

## Interpretation

This is an executor identifiability ceiling, not evidence that the 25-epoch optimizer simply failed to converge.

For arbitrary wiring drawn independently of the public descriptors, the registered CASM-S gate cannot condition its routing/execution on which parent edges are actually true. Increasing executor training time cannot recover information absent from the gate's inputs.

Therefore the zero exhaustive ceiling in the M2 smoke run is scientifically consistent with an information bottleneck in the executor itself.

## Consequence for M2

The frozen M2 contract correctly says that a low exhaustive CASM-S ceiling prevents a capability-relative selective verdict. The full 5-seed × 5-level run should not be spent until the executor contract is resolved.

A valid continuation requires a new experiment identifier/amendment that chooses one of two scientifically distinct directions:

1. Expose program structure required for execution: the executor receives an encoding of the actual public program wiring. This tests selective execution conditional on an executable structural representation.
2. Change the program family: restrict the generator so true wiring is derivable from public descriptors. This tests a different, more constrained executor problem and must not be presented as the original M2 question.

An oracle true-edge mask may be used as a diagnostic control, but it must not be silently substituted into M2 because that changes the registered executor.

## Provenance

- TAC-OSM M2 head inspected: 72f7d89be7a148bf81ea8ff5e249ddd6651de4cc.
- Frozen contract: TACOSM-M2-CASM-SELECTIVE-009, version 0.3.4-final.
- External CASM-S: FPC-effortless/cdl-attention-experiment@c31554413301e3c9d3e6b3f8c8c6be572a74a748.
- Smoke artifact: workflow run 36940235309, artifact 11199873829.
- This diagnostic is not a measurement result for M2 and does not promote or demote the M2 primary endpoint.
