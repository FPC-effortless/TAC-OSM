# TAC-OSM Architecture Hardening v1

This document records the implementation boundary added after the v0 audit.
It is an engineering contract, not a capability result.

## Evidence classification

| Change | Class | What is inherited | What remains unproven |
|---|---|---|---|
| Explicit write/read delay | L1 | persistent state as a causal container | learned temporal persistence across k boundaries |
| Address/read separation | L1 | keyed state addressing vocabulary | scalable semantic memory |
| Content-address index | L1 control | deterministic lookup as a sparsity mechanism | semantic retrieval and learned addressing |
| Runtime retrieval boundary | L2 integration surface | selective candidate sets | economic scaling until measured |
| Structured verifier | L1 mechanism | verify-localize-repair shape | verifier quality on real tasks |
| Executable bounded repair | L2 integration surface | bounded retries | useful repair on learned structures |
| First-class trajectory | L1 protocol | trace/provenance practice | downstream learning value |
| Generalized task generator | L1 protocol | controlled benchmark construction | real-world relevance |
| Synthetic relation executor | L0 control | structural execution interface | CASM learned execution |

Code existing in this document does not promote a scientific claim by
itself. Capability status moves only through a named measurement with its
contract.

## Runtime call graph

The hardened loop is:

    world write
        -> temporal boundary
        -> address
        -> retrieve/retain R
        -> route over R
        -> execute one selected structure
        -> observe outcome
        -> verify with structured evidence
        -> bounded repair/re-execution
        -> delayed learning write
        -> trajectory record

The crucial difference from v0 is that retrieval is no longer a report-only
calculation. The retained candidate set is the actual input to the router, and
only the selected candidate is executed.

## Temporal contract

A persistence probe has explicit:

    write_step = t
    delay = k
    read_step = t + k

TemporalBenchmark rejects skipped decision steps. A write is staged at the
write boundary and cannot be read until its availability boundary. A test must
therefore execute all intervening decision steps before reading the value.

The required experiment should report capability as a function of k, with:

    carry
    reset
    shuffle
    corruption
    distractor/noise controls

The implementation establishes the boundary; it does not establish the
result.

## Retrieval and cost contract

There are now two distinct costs:

1. index construction cost, charged when a candidate population is materialised;
2. query-time address cost, charged per lookup.

The query-time content-address control indexes exact equality signatures and
returns only a bounded bucket. Its query cost is a function of the number of
marked positions and returned candidates, rather than a scan of H.

This is deliberately not called a semantic retrieval solution. It is an exact
synthetic control that gives M2 a real execution boundary. A learned/semantic
addresser can later implement the same interface.

The critical measurement must report, separately:

    H
    |R|
    index_build_candidates
    address_query_positions
    candidates_routed
    executor_invocations
    verifier_checks
    wall_clock_seconds
    capability

The one-time build cost should also be amortized over an explicitly declared
number of queries when comparing against an exhaustive baseline.

## State decomposition

The v1 interfaces keep these operations distinct:

    address(query) -> memory item
    read(query) -> representation
    represent(memory item) -> features
    route(query, candidates) -> computation selection

The old v0 pattern of returning the whole state pool and truncating it inside
the feature function is removed from the hardened path. This prevents
max_state_slots from masquerading as retrieval.

## Verification and repair contract

Verification now emits:

    valid
    failed_constraint
    counterexample
    repair_target
    confidence
    evidence

The verifier checks consistency between the executed output and observed
outcome and checks trace values against declared bounds. It does not receive
the hidden acceptable action.

Repair is no longer a text-only placeholder. It re-executes alternate retained
candidates under a fixed attempt budget. A successful repair records the
candidate replacement and the verification evidence.

This still does not prove useful verifier-guided repair in a CASM/PLM system;
that requires a separate evaluation.

## Benchmark generalization

benchmark_v1.py supports:

    relations:
      equality
      xor_parity
      majority
      any_match

    validity:
      unique
      multiple
      none

The candidate population is sampled before validity mode is imposed. This
removes the v0 construction in which every distractor was deliberately made
to fail the same known gold relation.

Tiering remains:

* Tier A — controlled synthetic mechanism isolation.
* Tier B — held-out relation/generator/validity distributions.
* Tier C — real structured relevance tasks.

Only Tier B/C results can address generality beyond the synthetic mechanism
control.

## CASM and PLM boundary

The hardened runtime intentionally does not rename the synthetic executor as
CASM, and it does not claim that a dictionary-like state object is PLM
memory.

The integration points are:

    PLM: temporal persistent state
    CDL: candidate relevance/router implementation
    CASM: structural executor implementation
    TAC-OSM/PNDS: transition loop
    verifier: epistemic control
    slow optimizer: later strategy/model updates

The currently shipped executor is hardened_v1.synthetic_executor.

## Experiment gate

Before an expensive confirmatory run, the following must be true:

1. temporal boundary tests pass;
2. retrieval boundary tests pass;
3. empty retrieval fails closed;
4. index build/query costs are separated;
5. trajectory is emitted by the runtime;
6. verifier and repair tests pass;
7. benchmark validity/leakage tests pass;
8. the research contract names whether the population is static or dynamic;
9. the run reports amortized index-build cost;
10. claims/evidence ledgers are updated only after the measurement result.

The next scientific step is therefore not another REINFORCE tweak. It is a
contracted temporal/selective-execution experiment on the hardened call graph.
