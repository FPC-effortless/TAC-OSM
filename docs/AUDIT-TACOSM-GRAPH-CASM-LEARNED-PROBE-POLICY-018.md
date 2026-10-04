# G-CASM-018 — Learned Probe Policy

**Status: PREREGISTERED / DESIGN ONLY**

G-CASM-018 is the first learned version of active evidence acquisition.

It does not replace the exact G-CASM-017 selector. The exact indexed selector is
the ceiling. The learned policy is evaluated by downstream verified capability,
not by agreement with the ceiling.

## Policy state

At time t the policy receives a compact description of the current hypothesis
state:

- normalized evidence-signature histogram for each legal probe;
- expected environment cost for each legal probe;
- current probe budget;
- current evidence summary;
- compact public task/state features.

It does not receive target identity, target index, target-only hidden state,
future target evidence, or evaluator labels.

## Training

Training uses independent program libraries and task streams reserved exclusively
for policy fitting.

The teacher target is the exact indexed information-per-cost action on those
training libraries.

No held-out evaluation library is used to:

- fit parameters;
- select architecture;
- choose training duration;
- choose feature dimensions;
- choose the winning seed;
- tune the stopping rule.

A fixed validation split, established before held-out evaluation, is used for
training diagnostics only.

## Policy architecture

The first candidate implementation should be deliberately small:

    per-action encoder
        ↓
    shared MLP
        ↓
    action utility scalar
        ↓
    masked argmax

The action encoder consumes an indexed partition histogram rather than the raw
candidate list. This keeps the learned policy aligned with the computational
representation introduced by G-CASM-017.

A later experiment can test whether the action utility estimator itself can be
made sublinear, but that is not assumed here.

## Objective

The teacher loss may be cross-entropy over the exact training action.

The scientific endpoint is not teacher agreement.

At evaluation:

    probe policy
      → structured evidence
      → exact hypothesis update
      → terminal candidate selection
      → exact verifier

The primary endpoint is the difference in K=4/B=8 verified success from the
exact indexed ceiling at M=512.

This forces the learned policy to be judged on useful computation rather than
imitation.

## Failure interpretations

High action agreement + low verified success:
the learned policy imitates the information teacher but fails to translate
evidence into useful terminal selection.

Good information gain + poor terminal success:
the evidence policy is informative but the downstream representation/orderer is
limiting.

Large gap to exact ceiling:
the remaining learning problem is policy approximation or insufficient policy
state representation.

Good M=512 performance + poor smaller-M performance:
the policy has learned the large-population partition structure but may be
over-specialized to the high-ambiguity regime.

## Persistence path

The first 018 implementation should not silently introduce persistent learning.
Probe trajectories are stored for diagnostics. A later PLM integration experiment
will require verifier-gated consolidation of evidence into persistent
experience.

## Scientific boundary

018 would establish learned active acquisition only on the registered synthetic
workload.

It would not establish:

- generic active learning;
- optimal experiment design;
- autonomous scientific discovery;
- asymptotic scaling;
- multimodal reasoning;
- hardware speedup.
