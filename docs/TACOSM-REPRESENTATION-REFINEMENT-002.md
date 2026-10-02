# TACOSM-REPRESENTATION-REFINEMENT-002

## Purpose

This is the next representation-level research phase after
TACOSM-VRS-DYNAMIC-REPRESENTATION-001.

It is not a replacement for VRS-001 and does not modify G-CASM-010.

The question is whether a representation that fails a dynamic-validity test can
be repaired using **structured verifier counterexamples** without erasing
previously verified structure.

## Motivation from the current research

The unified PLM program now treats representation as an explicit computational
object:

    Phi_t

rather than an incidental side effect of a router.

VRS-001 tests whether Phi is dynamically valid in the first place.

This phase asks the next question:

    Phi_t
      -> counterexample
      -> targeted refinement
      -> Phi_{t+1}
      -> re-validation

The update is not permitted to be judged by semantic plausibility alone.

## Research lineage

Abstraction Agent explicitly identifies solver-in-the-loop feature refinement
as a natural continuation of its representation-synthesis pipeline. Its
published experiments establish useful static semantic features, but not
persistent dynamic repair. (arXiv:2609.04303)

A September 2026 feature-engineering study separates proposal and extraction
and converts concrete predictive failures, such as ranking inversions, into
feedback for the next feature proposal. The transferable mechanism is the use
of **specific downstream errors to steer representation refinement**, rather
than scalar success feedback alone. (arXiv:2609.21894)

VERDI makes the stronger epistemic distinction that retrieved experience is a
hypothesis until target-side validation, and uses contradictions between nearby
diagnostic fingerprints to evolve the representation used for transfer.
(arXiv:2608.09537)

RefineICL shows that support-driven representation updates can change later
behavior without changing model parameters, making the evolving representation
itself a causal computational object. (arXiv:2609.27679)

LOTUS uses behavioral/bisimulation constraints for task representations,
providing another precedent for treating representation validity as a
behavioral property rather than merely a geometric one. (arXiv:2608.15509)

## Core mechanism

The proposed architecture is:

    current representation
          |
          v
    downstream computation
          |
          v
       verifier
          |
          +---- pass ----> persist / reuse
          |
          +---- fail ----> structured counterexample
                              |
                              v
                       representation proposer
                              |
                              v
                     calibrated Phi_{t+1}
                              |
                              v
                       dynamic re-validation

The crucial boundary is:

> Verification produces evidence about the representation. It does not
> directly mutate the representation.

A repair engine receives only the registered counterexample record and the
current representation contract.

## Counterexample object

The persistent failure record should contain:

    representation_version
    pair_id
    state_a
    state_b
    action
    horizon
    distance_before
    distance_after
    outcome_a
    outcome_b
    verifier_reason
    provenance

It must never be replaced by a generic scalar reward.

The record is itself typed evidence and can later become a training/repair
input.

## No-regression condition

A repaired representation is admissible only if:

    previously_verified_invariant(Phi_t)
        =>
    previously_verified_invariant(Phi_{t+1})

within registered tolerance.

This includes point calibration anchors and any behavioral anchors established
by the previous version.

Therefore representation repair follows the same epistemic rule as the existing
PLM state commit path:

    proposal != verified state.

A proposed Phi_{t+1} remains tentative until the verifier accepts it.

## Train/evaluation separation

Three sets are required:

1. counterexample-driving cases;
2. repair-validation cases;
3. final hidden cases.

Only the first set may influence the repair proposal.

The second checks whether the repair generalizes beyond the observed failure.

The third remains hidden until the repair version is frozen.

This avoids a simple repair loop that memorizes each failure case.

## Primary endpoint

The primary endpoint is the final dynamic-validity violation rate after the
fixed three-cycle refinement budget, subject to:

    capability >= 0.80 * raw_state_reference

and

    verified_invariant_retention = 100%

within tolerance.

This makes the desired quantity:

    fewer future behavioral violations
    without buying that improvement by destroying prior structure.

## Important negative result

A repair that fixes old counterexamples but creates new violations is not a
successful persistent representation.

It is reported as a repair tradeoff:

    old violations down
    new violations up

rather than collapsed into one success score.

This is particularly important for continual PLM because a sequence of locally
successful edits could otherwise produce cumulative semantic drift.

## Two-timescale representation model

The research should ultimately separate:

    Phi_shared
        = slow, consolidated computational structure

    DeltaPhi_t
        = fast task/experience-conditioned refinement

with:

    Phi_shared <- consolidate(DeltaPhi_t)

only after verification.

This maps naturally onto the existing PLM slow/fast learning distinction.

It also creates a concrete target for later continual-specialization
experiments: determine whether local representation updates can increase
capability without globally rewriting the shared substrate.

## Relationship to operator routing

Representation repair should eventually be measured on both routing surfaces.

State side:

    Phi_S(S_t), Q_t -> R_t

Operator side:

    Phi_E(E), R_t, Q_t, G_t -> C_t

A representation repair that improves state addressing but destroys operator
selection is not a global improvement.

The eventual objective therefore becomes:

    verified representation
      -> state selectivity
      -> operator selectivity
      -> actual execution work
      -> verified outcome

rather than a representation benchmark isolated from computation.

## Scope

This phase does not establish:

- universal self-improving representations;
- asymptotic scaling;
- that LLM repair is better than non-LLM optimization;
- that every verifier counterexample has a representational solution;
- that reduced geometric error implies improved useful computation;
- that a repaired representation should replace the raw evidence.

Its intended result is narrower:

> a test of whether verifier-derived counterexamples can safely drive bounded
> representation refinement in a persistent computational system.
