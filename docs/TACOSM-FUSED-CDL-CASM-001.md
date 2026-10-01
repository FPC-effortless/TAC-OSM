# TACOSM-FUSED-CDL-CASM-001

## Status

Pre-registered implementation and integration experiment.

## Purpose

Fuse the previously validated CDL and CASM research lines into the same causal
loop as the successor architecture without importing the expensive CDL language
model into runtime.

The runtime path is:

    S_t
      -> StateAddress
      -> CDL student relevance score
      -> top-1 action, with top-K admission measured
      -> CASM computation selector
      -> exact structural execution
      -> O_t
      -> post-execution verifier
      -> verifier-derived router update
      -> verified experience write
      -> S_{t+1}

The experiment is a full-loop integration test, not a C5 claim.

## CDL boundary

CDL is implemented as the cheap Q/K student form validated in the standalone
CDL experiment: a query projection and candidate projection followed by a dot
product. The external SmolLM2 conditional-description-length teacher is not run
at inference because it requires a full language-model forward pass for each
candidate.

The student sees only:

- public query bits;
- context/marked positions;
- the addressed persistent value when an address is present;
- candidate descriptors.

It does not receive:

- target action;
- gold index;
- outcome;
- verifier result.

## CASM boundary

CASM is implemented as an explicit computation-selection adapter. Once CDL has
selected a candidate, CASM compiles the relation between the current reference,
candidate descriptor, and marked positions into an exact Boolean computation
graph.

The selected graph is executed exactly and then independently checked by the
verifier. Learned CASM gate optimization is intentionally not combined with
this first fusion run because the earlier learned CASM result failed its
capability test while the exact execution substrate passed.

## Learning boundary

Learning occurs only after execution and verification.

- Success: selected candidate becomes a positive.
- Rejection: the hidden gold index exposed by the post-hoc verifier becomes a
  repair positive.
- The update is pairwise against the strongest current competitors.
- Verified experience writes use a separate experience: namespace and never
  overwrite the environment's addressed world facts.

## Measurements

Primary:

1. end-to-end top-1 task success;
2. CDL target admission recall at K=4;
3. mean target rank;
4. verifier agreement with task success;
5. verified write count and learned update count.

Secondary:

1. CASM active nodes;
2. CASM candidate-edge substrate size;
3. state slots inspected and pool size;
4. comparison against static routing;
5. oracle execution ceiling;
6. reset-state intervention.

## Controls

The same deterministic task stream is used for each seed class. The static
router is the routing control. The oracle router is the ceiling. A persistent
vs reset comparison tests whether the fused policy depends on retained state.

## Falsifiers

The fused architecture should not be promoted as a useful integrated result if:

- oracle top-1 is below 1.0;
- exact CASM execution disagrees with the benchmark relation;
- verifier pass rate diverges from successful execution;
- CDL admission recall at K=4 does not materially exceed chance (0.5 in an
  eight-candidate, four-admitted setting);
- verifier-derived updates do not change the student state;
- learning depends on target/outcome leaking into route();
- addressed world facts are overwritten by learning writes.

## Explicit non-claims

This experiment does not establish sublinear end-to-end compute, hardware FLOP
reduction, semantic retrieval at open scale, learned CASM gating, or the C5
claim C_executed = f(|R|). The runtime CDL scorer still evaluates every
candidate. The K=4 number is an admission-capability measurement, not a hidden
compute claim.
