# Unified research state — 2026-10-02

## Current architecture

The target architecture is now:

I_t
 -> D_t
 -> Z_t
 -> M_t
 -> P_t
 -> R_t
 -> Pi_t
 -> C_t
 -> A_t
 -> O_(t+1)
 -> V_t
 -> W_t
 -> L_t
 -> (S_(t+1), K_(t+1), E_(t+1))

with four explicit persistent substrates:
S_t world/task state,
K_t executable operator/library state,
E_t verified experience/evidence,
V_t verifier/repair/uncertainty state.

## Mandatory gates

Representability and identifiability are separate gates.
The learner-visible observation map must determine the target relation.
No hidden truth may cross the routing/planning boundary.
Proposal coverage and post-admission selection are separate endpoints.
Verifier output is post-execution only.
Memory writes are post-verification.
Retraction is explicit rather than silent overwrite.
Compute and memory budgets are first-class constraints.

## Evidence inherited

- TAC persistent state read/intervention vocabulary: bounded controlled evidence.
- Temporal write/read persistence through k=32: bounded TAC-OSM mechanism result.
- Conditional relevance routing: strongest earlier learned-router evidence.
- Exact executable structural computation: bounded CASM substrate evidence.
- Verified repair loop shape: architecture-neutral control shape only.
- Product-key addressing: validated capability/computation frontier, not
  asymptotic scaling.
- PST/StructMeans/AXON/SECA phase-2 mechanisms: implementation/validation
  evidence exists, but the broader integrated experiment remains gated by
  repository/provenance cleanliness.

## Open scientific problems

1. Learn predictive/action-sufficient representations that are identifiable
   from observables.
2. Learn persistent write/consolidation/retraction policies.
3. Learn sparse proposal/addressing without capability loss as population grows.
4. Connect selective execution to the pinned CASM-S substrate.
5. Use the predictive model for multi-step planning rather than one-step
   operator choice.
6. Establish that verified experience improves future held-out performance.
7. Test modality-general structure learning across language, image and audio,
   followed by real-data validation if the synthetic gate passes.

## Scientific rule

No unified-system claim is promoted by adding more components. Each mechanism
must pass its own gate, and the integrated model must pass a preregistered
causal comparison against frozen controls.
