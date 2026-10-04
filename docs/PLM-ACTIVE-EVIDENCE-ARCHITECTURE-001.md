# PLM Active-Evidence Architecture 001

## Purpose

This design records the architectural consequence of the G-CASM information
bottleneck and the validated G-CASM-014 result without changing any measured
experiment.

The central unit becomes:

    represent -> hypothesize -> select information -> act -> observe
    -> update belief/state -> select computation -> verify -> learn

This is a PLM design direction, not an assertion that the current repository
implementation already realizes the full architecture.

## 1. Problem formulation

Let H_t be the current hypothesis set or belief state over executable
structures. Let D_t be accumulated evidence. A probe action a is selected from
an admissible action set A_t.

The environment returns structured evidence E_t according to:

    E_t ~ p(E | H, a, D_t)

and the state update is:

    B_{t+1} = Update(B_t, a_t, E_t)

where B_t may be a discrete candidate posterior, a continuous belief embedding,
or a hybrid.

For deterministic finite-domain experiments:

    I(H_t;E_t | a_t,D_t) = H(E_t | a_t,D_t)

under the registered uniform prior.

For stochastic environments, the general objective is:

    U(a | B_t) =
        IG(B_t; E_a | a, D_t) / Cost_env(a)
        + lambda_v * TaskValue(a)
        - lambda_r * Risk(a)

The information term is a diagnostic and optimization target; it should not be
assumed to equal task reward.

## 2. First-class probe objects

A probe is a typed computational request:

    Probe {
        action_type
        parameters
        expected_schema
        cost_model
        risk_constraints
        stopping_relevance
    }

The action is not merely a query token. It describes what computation the
environment should perform and what evidence contract it should return.

Examples:

- scalar_row(x) -> y
- activation_trace(x) -> [z_1,...,z_k]
- paired_counterfactual(x,delta) -> (E(x), E(x xor delta))
- branch_signature(x) -> branch/path descriptor
- relational_probe(x_1,...,x_n) -> relation graph
- partial_execute(prefix) -> trace prefix + frontier state

## 3. Structured evidence packet

CASM should emit an EvidencePacket rather than an untyped scalar:

    EvidencePacket {
        schema_id
        payload
        provenance
        action_id
        cost
        confidence
        counterfactual_links
    }

OSM consumes the packet and stores a compressed evidence structure.

The evidence should remain provenance-addressable to the originating probe,
environment execution and verifier result.

## 4. CDL role

CDL becomes an information-action policy rather than only a candidate scorer.

Given current belief B_t, CDL estimates:

    Q_probe(a) =
        expected information gain
        / expected acquisition cost

augmented when appropriate by downstream utility:

    Q_total(a) =
        Q_probe(a)
        + terminal-value estimate
        - risk penalty.

The first exact implementation can use finite candidate enumeration. The
learned PLM implementation should replace enumeration with a learned belief
state and amortized probe-value estimator.

## 5. OSM role

OSM becomes a persistent hypothesis/state substrate.

Required state partitions:

1. world/task state
   - current task variables
   - environment context
   - observed entities

2. computational state
   - candidate structures
   - executable operators
   - intermediate states
   - dependency/frontier information

3. experience/evidence state
   - probe history
   - evidence packets
   - verifier outcomes
   - counterexamples
   - provenance

4. policy/harness state
   - goals
   - budgets
   - risk limits
   - stopping criteria
   - active probe strategy

OSM must support both exact symbolic compatibility and learned continuous
belief representations. The latter cannot silently replace the former without
a validity gate.

## 6. VRS role

VRS becomes the state-transition and stopping authority.

At each step it evaluates:

    sufficient(B_t, goal_t, budget_t)

and either:

    continue -> request more information

or:

    terminate -> authorize terminal computation.

A useful stopping criterion is expected value of information:

    VOI(a) =
        E[downstream verified utility after a]
        - current best verified utility
        - acquisition cost(a)

Continue while max_a VOI(a) exceeds the configured threshold and risk bounds.

The threshold itself must be preregistered for scientific experiments.

## 7. CASM role

CASM is both:

- the terminal executable-computation engine;
- the environment-facing execution engine for information probes.

This distinction matters. The same computational substrate should be able to
perform an action whose purpose is to solve the task and an action whose purpose
is to learn enough about the task to solve it.

That makes probing part of computation rather than an external data-collection
module.

## 8. Learning path

The verified loop becomes:

    S_t
      -> B_t
      -> CDL(a_t)
      -> CASM(a_t)
      -> E_t
      -> OSM.Update
      -> VRS
      -> [continue | terminal CASM]
      -> O_t
      -> verifier
      -> verified transition
      -> S_{t+1}

The slow optimizer should learn:

- probe utility;
- probe ordering;
- representation composition;
- stopping policy;
- candidate abstraction;
- verifier repair strategy.

Fast competence can adapt task-local routing/representations. Slow learning
changes general policy and representation.

No update should be accepted from an unverified terminal outcome when the
experiment is intended to measure verified learning.

## 9. Critical architectural boundary

The selector must never receive hidden target identity through the observation
channel.

Permitted:

    public candidate representations
    current evidence
    prior beliefs
    predicted outcomes of candidate hypotheses
    costs and constraints

Not permitted:

    target index
    target-only hidden state
    evaluator labels
    future observations
    verifier-only rows before the corresponding action

This boundary applies equally to exact oracle ceilings and learned policies.
Oracle ceilings must be explicitly labelled as ceilings.

## 10. Information bandwidth as an architectural variable

The experiment sequence establishes three distinct failure planes:

    Plane A: insufficient information
    Plane B: insufficient hypothesis ordering/representation
    Plane C: insufficient terminal execution/verification

G-CASM-013/014 strongly isolates Plane A for the four-bit scalar channel.
A richer probe channel is therefore expected to move the bottleneck toward B.

The correct objective is not maximum raw evidence entropy. It is useful
information efficiency:

    useful_verified_progress / acquisition_cost

with:

    useful_verified_progress
        = downstream improvement in verified capability or reduction in
          expected execution work.

A trace with more entropy but no verified downstream benefit is not a successful
architectural mechanism.

## 11. Structured-probe ladder

The next channel families should be evaluated in a fixed hierarchy:

### Level 1 — multi-bit direct evidence

Examples:
- activation traces
- compact state vectors
- branch IDs

### Level 2 — relational evidence

Examples:
- equality/inequality constraints
- operator-response signatures
- dependency relations

### Level 3 — counterfactual evidence

Examples:
- paired input perturbations
- action-conditioned differences
- local sensitivity signatures

### Level 4 — temporal evidence

Examples:
- partial execution traces
- state transitions
- action/outcome sequences

### Level 5 — learned abstraction

A learned CDL predicts which evidence type is likely to have the highest
verified value per unit cost.

The levels should not be collapsed into one learned black box before each
evidence family has been independently characterized.

## 12. Research sequence

### G-CASM-015

Measure whether a six-bit activation trace contains more information per
environment work than a one-bit scalar output.

### G-CASM-016

Compare several structured observation families at matched acquisition cost:
activation trace, relational signature, and paired/counterfactual evidence.

### G-CASM-017

Compute exact small-M decision-tree ceilings. This separates:
- greedy policy limitations;
- observation-channel limitations;
- terminal execution limitations.

### G-CASM-018

Train a target-identity-blind learned CDL to choose probe actions from training
program behavior only.

The learned selector must be evaluated on a fresh held-out environment and
must consume compressed candidate representations rather than evaluator truth
tables.

### G-CASM-019

Integrate CDL, OSM, CASM and VRS into one persistent closed-loop PLM model.

The ablation matrix should remove one mechanism at a time:

    full active PLM
    - active acquisition
    - structured evidence
    - persistence
    - verifier gating
    - learned probe policy

No component should be promoted solely from a routing metric if the downstream
verified task metric does not improve.

### G-CASM-020

Transfer the mechanism to modality-controlled language, image and audio
tasks.

The first multimodal experiments should remain mechanism tests:
held-out composition, intervention, persistence and verification—not benchmark
marketing.

## 13. Scientific requirements

Every active-evidence experiment must report:

- raw per-seed evidence trajectories;
- candidate/evidence split integrity;
- exact leakage boundary;
- probe acquisition cost;
- candidate-side prediction/planning cost;
- terminal execution cost;
- verification cost;
- stopping decisions;
- seed-level aggregation;
- preregistered uncertainty;
- immutable provenance.

A result that improves information gain while worsening verified capability is a
diagnostic, not a success.

## 14. Relationship to current literature

Recent work independently motivates action-conditioned observation selection
and information-efficient acquisition. The PLM direction differs in emphasis:
the probe is a native executable operation inside the persistent computational
substrate, and its output is stored as verifiable computational experience.

The architectural objective is therefore not to reproduce active inference or
active feature acquisition. Those literatures supply useful formal tools for
information-seeking action selection; PLM uses them as mechanisms inside a
persistent executable memory-and-computation architecture.

## 15. Core design thesis

The architecture should learn not only:

    "What computation should I perform?"

but also:

    "What information-producing computation should I perform next so that I
     can know which computation is worth performing?"

That second question is the missing control loop exposed by the G-CASM-013/014
results.
