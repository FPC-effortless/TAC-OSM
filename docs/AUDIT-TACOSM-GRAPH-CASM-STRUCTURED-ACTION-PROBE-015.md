# G-CASM-015 — Structured Action Probes for Higher-Entropy Evidence

**Status: PREREGISTERED / IMPLEMENTATION**

## Scientific question

G-CASM-013 established that four binary observations have a hard target-weighted
collision bound of M/16 under a uniform hypothesis prior. G-CASM-014 tests
whether adaptive placement can recover the gap between the frozen support order
and that bound.

G-CASM-015 changes the observation channel itself. It asks whether an
instrumented execution trace can provide more model information per unit
environment execution work than a scalar external output.

## Mathematical correction

The quantity M/2^K is not a Shannon lower bound in the strict sense. For a
deterministic K-bit observation map with a uniform target over M candidates, it
is a collision/partition lower bound:

    E[|H(E)|] = sum_e |H_e|^2 / M >= M / 2^K.

The proof follows from Cauchy-Schwarz. Shannon entropy is a related but distinct
quantity: for deterministic evidence I(H;E) = H(E) <= K bits.

The distinction is material because G-CASM-015 measures both entropy and
posterior candidate-set size rather than treating them as interchangeable.

## Observation channels

### scalar_row

One action selects an input row and observes only the binary program output.
The evidence alphabet has size at most 2.

### activation_trace

One action selects an input row. CASM executes the program and returns the
ordered output bits of all non-input active nodes. The registered generator has
10 active nodes and 4 inputs, so the trace has six internal activation bits and
a nominal alphabet ceiling of 2^6 = 64.

This is explicitly an instrumented sandbox measurement. It is not treated as a
generic assumption that real black-box environments expose internal state.

## Identity and leakage boundary

The action selector sees only:
- the current compatible hypothesis set;
- public candidate representations;
- candidate-predicted evidence for each legal action;
- action costs available before execution.

The selector never receives:
- target identity or target index;
- target evidence before action selection;
- held-out verifier labels;
- confirmatory outcomes.

The target evidence is revealed only after the selected action is committed.

## Cost accounting

Scalar acquisition cost is one exact CASM execution.

Trace acquisition cost is one exact CASM execution plus one read unit per returned
activation bit. The benchmark separately records candidate-side prediction/cache
work so that an information-efficiency result cannot be misreported as total
system speedup.

The primary endpoint uses the environment-facing work term. Candidate-side
planning/cache work is a secondary accounting dimension and is not silently
merged into the primary.

## Primary endpoint

For each task and each channel, score all legal one-step actions using:

    I(H_t; E_a) / E[cost(a) | H_t].

The hypothesis prior is uniform over the current compatible candidates.
Because evidence is deterministic conditional on a candidate,
I(H_t;E_a)=H(E_a).

Seed-level means are computed first, then the channel difference is aggregated
across the five preregistered seeds with a seed bootstrap. No pooled-trial
endpoint is used for the primary claim.

## Interpretation boundary

A positive result licenses only the statement that the registered structured
observation channel has higher information efficiency on this finite workload.

It does not establish:
- learned autonomous probe selection;
- optimal decision-tree search;
- generic active learning superiority;
- access to internal activations in arbitrary environments;
- asymptotic sublinear retrieval;
- language, image, or audio competence;
- hardware speedup.

## Planned sequence

015 measures the observation-channel ceiling.

A later experiment may learn a probe policy from training-program behavior, with
the exact 015 oracle retained as an upper bound. That learned policy must operate
without target identity, future target evidence, or evaluation truth-table access.

The architectural mapping is:

    CDL  -> select information-bearing probe action
    CASM -> execute the probe and emit structured evidence
    OSM  -> update the hypothesis/state representation
    VRS  -> decide whether evidence is sufficient to terminate
    AXON/slow optimizer -> learn probe utility and strategy over time

This sequence keeps observation acquisition causally separate from terminal
execution and verification.


## Pre-confirmatory cost-accounting amendment

The primary endpoint intentionally remains **environment-facing information
efficiency**:

    I(H;E) / E[environment acquisition work].

To prevent hidden computational cost from being interpreted as a system-level
efficiency result, every result artifact must also report:

1. candidate-cache construction work, separately for each observation channel;
2. deterministic selector prediction-scan work, using one unit per scalar
   evidence bit and six units per activation-trace evaluation;
3. amortized total work per registered task, with the cache amortized over the
   full 5 M-level × 32-task seed grid;
4. information gain divided by that amortized total work.

This is an accounting amendment made before confirmatory measurement. It does
not change the channel definitions, task grid, endpoint decision rule, or
leakage boundary.
