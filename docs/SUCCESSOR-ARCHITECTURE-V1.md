# TAC-OSM Successor Architecture v1

Status: implementation/pre-capability gate.

This branch is a successor architecture. It does not replace or reinterpret
C5-001, C5-002, or C5-003.

## Semantic boundaries

The successor makes four objects explicit.

### 1. State addressing

`AddressState(Q_t,S_t) -> M_t`

The query is addressed first. Only the addressed memory value is passed to
candidate scoring. Unrelated historical slots are not part of the query
representation.

The current adapter is intentionally conservative: because the v0
`PersistentStore.read()` API materialises a `StateRead`, the addressor does
not claim sublinear asymptotic lookup yet. The next storage implementation can
replace the addressor's internals without changing the routing API.

### 2. Candidate routing

`z_q=f_theta(Q_t,M_t)`

`z_i=g_phi(C_i)`

`E(q,c_i)=-<z_q,z_i>`

`R_t=TopK_B(-E)`

The representation maps are learned. Candidate action indices are excluded
from the representation.

Training is pairwise and outcome-derived. A successful execution supplies the
positive candidate; the strongest current competitor is the hard negative.
A failure supplies no invented positive label.

The current implementation still scores the complete candidate set. Its
diagnostic `candidate_coverage` is therefore reported as 1.0. This is
intentional: the branch does not claim sublinear candidate retrieval.

### 3. Explicit executable graph

`G=(V,E_true)` is the executable object.

Execution consumes `E_true` directly and checks that each required input port
has exactly one source. It does not infer hidden wiring from node-local
structure or candidate-edge statistics.

Exact Boolean semantics are the reference path. The soft `alpha` computation
strength path is available only as an explicit ablation.

### 4. Packed work accounting

`PackedGraphBatch` stores active nodes and true edges contiguously and records
per-graph offsets.

The first version is a transport/measurement layer rather than a GPU kernel.
Work is reported as active nodes, active true edges, and structural operations.
Candidate-edge counts are recorded as metadata so that substrate size is not
silently confused with executed work.

## What this branch does not claim

It does not yet establish:

- sublinear end-to-end history scaling;
- learned semantic addressing;
- learned graph execution;
- capability improvement over C5;
- economic FLOP reductions on a hardware kernel.

Those require separate experiments.

## Pre-capability gates

The required sequence before a new capability experiment is:

1. exact executor correctness on explicit topology;
2. soft semantics separated from exact reference;
3. addressed-memory boundary free of unrelated state exposure;
4. learned router free of action-index leakage;
5. successful-outcome hard-negative update path;
6. packed active-work accounting;
7. representability and anti-leakage re-run against the successor inputs;
8. only then register the next capability experiment.

C5 evidence remains frozen.
