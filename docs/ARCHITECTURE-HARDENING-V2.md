# TAC-OSM Architecture Hardening v2

Status: engineering review / successor architecture. This document does not alter any historical experiment.

## Scope

Review the current TAC-OSM implementation as an executable model, not only as an experiment harness. The review targets semantic mismatches, weak hypothesis classes, stale optimization choices, hidden coupling, and places where the implementation is measuring a different object from the architecture it claims to implement.

## High-severity findings

### H1 — Program topology is not a first-class field

`Program.candidate_edges` is documented as the full substrate, but `Program.true_edge_set` currently derives its "true" topology from **all candidate edges**. This makes the copy-mask oracle vacuous and destroys the distinction between substrate and program.

This is a semantic implementation defect, not a modeling preference.

**Fix:** make `true_edges` an explicit first-class field; validate `true_edges ⊆ candidate_edges`; make `copy_mask` read only that field.

### H2 — Internal CASM routing is solving an unidentifiable inverse problem

The pinned CASM-S router computes edge gates from observable node/edge-local structural attributes while the required wiring can vary independently. If two programs expose the same observable substrate but differ in hidden `true_edge_set`, no deterministic router can recover the correct wiring in general.

C5-003's diagnostic intervention is consistent with this: oracle topology works when computation strength is sufficient, while learned topology fails even after computation strength is fixed.

**Architectural correction:** do not ask a router to infer program topology that is already part of the executable program. Program topology should be represented explicitly and routed only when there is a genuine candidate-selection problem. Internal execution should consume an executable graph/edge set.

### H3 — CASM computation strength is entangled with Boolean capability

The external CASM implementation uses `softplus(alpha_eta)` as a computation-strength parameter. That is useful for a learnable substrate, but it is not an exact Boolean semantics layer. A fixed Boolean relation should not depend on a learned analog amplitude crossing an arbitrary 0.5 threshold.

**Architectural correction:** exact program execution is a separate semantic mode. Learned computation strength is an ablation, not the default semantics for an exact relation.

### H4 — The learned router is a hand-designed linear bandit inherited from an earlier experiment

`LearnedRelationalRouter` is a linear score over manually engineered agreement features trained by one-step REINFORCE. This is useful as a historical control, but it is not a strong implementation of the intended representation-addressed architecture.

It has three limitations:
1. feature engineering determines the hypothesis class;
2. REINFORCE wastes signal when the candidate set and outcome are directly observable after execution;
3. the score is not learned as a reusable query/candidate representation or energy function.

**Successor:** retain the linear router as `legacy_linear_bandit`; introduce a separate representation/energy router trained with outcome-derived positive/negative pairs and hard negatives. Do not relabel the old router as the new architecture.

### H5 — Candidate selection and internal graph routing are conflated

There are currently two different routing problems:
- external routing: select which candidate structure to execute;
- internal routing: select which graph edges carry computation.

They have different information boundaries, targets, and cost models. Combining them makes a failure ambiguous.

**Correction:** external router selects candidate IDs; executable programs carry their topology explicitly; an optional learned graph operator may later learn *which computation to perform*, but it must have an observable target representation.

### H6 — State addressing is still mixed with candidate scoring

The current state read returns the addressed slot first, but the router feature basis still materializes fixed-width slot blocks. This creates an artificial dependence on `max_state_slots` and makes state capacity a feature-vector layout concern.

**Successor:** separate:
`AddressState(S_t,Q_t) -> M_t`
from
`Encode(M_t,Q_t,candidate) -> z`
and keep the candidate scorer independent of the number of unrelated state slots.

### H7 — The current model is a component-injection harness before it is a coherent learned architecture

`TacOsmModel` is a good scientific integration harness, but most intelligence is delegated to deterministic components. That is appropriate for ablations but should not be mistaken for the integrated PLM/TAC-OSM model.

The successor should make the learned state representation, candidate energy, execution plan, outcome model, verifier, and state update explicit interfaces.

## Medium-severity findings

- `Program.input_values` is duplicated in the dataclass declaration.
- Several implementations use position/index as a proxy for structure; this is brittle under graph isomorphism.
- Python list/dict loops are used where batched tensor execution is the intended compute path.
- The CASM runtime pads variable-width inputs to the maximum width of a batch; this is a transport optimization, not semantic batching. It should eventually use packed graph representations with explicit offsets.
- The flattened baseline is a scalar mean/sum control, not a computationally matched dense program executor. It is useful as a crude control but should not be described as a matched architecture baseline.
- The representation router currently provides the right boundary but is only an injected encoder interface; it is not yet a learned representation model.
- The semantic LSH address index is a classical approximate-index control, not evidence for learned semantic addressing.
- Oracle arms are useful ceilings, but they must remain mechanically isolated from learned routing and never enter training inputs.
- The environment uses task descriptors whose gold structure is partly constructed by benchmark logic. This is appropriate for controlled experiments, but it is not yet a general world-model interface.

## Modernization direction

The successor architecture should use:

[
Q_t,S_t
ightarrow
M_t=operatorname{Address}(Q_t,S_t)
ightarrow
z_q
]

[
(Q_t,M_t,c_i)
ightarrow
z_i
ightarrow
E(q,c_i)
ightarrow
R_t=operatorname{TopK}_B(-E)
]

then:

[
R_t
ightarrow
	ext{explicit executable programs}
ightarrow
C_t
ightarrow
A_t
ightarrow
O_t
ightarrow
V_t
ightarrow
S_{t+1}.
]

The routing objective should be listwise/pairwise rather than one-step REINFORCE whenever outcome labels can produce reliable positives and hard negatives. Hard negatives should be mined from the current candidate pool, because the scientific question is separation *within a query*, not independent binary classification.

This follows current retrieval/routing practice: modern sparse routing explicitly optimizes a constrained selection budget, while current retrieval work emphasizes hard-negative selection and structured execution rather than relying on independent candidate classification. citeturn1academia0turn2search1turn2search2

Executable structure should remain a first-class object. Structured code/program representations are increasingly used precisely because they provide decomposable, executable and verifiable computation paths. citeturn2search0

## Implementation order

1. Fix program topology semantics.
2. Make exact execution the reference semantic layer.
3. Separate external candidate routing from internal program execution.
4. Replace the legacy linear REINFORCE router with a new energy/representation router behind a new interface.
5. Make state addressing an independent module.
6. Add packed/batched execution and explicit work accounting.
7. Re-run falsification tests before any new capability experiment.
8. Only then register the next capability experiment.

Historical C5-001/C5-002/C5-003 results remain frozen and are not rewritten by this architecture work.
