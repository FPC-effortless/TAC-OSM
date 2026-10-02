# TACOSM-VRS-DYNAMIC-REPRESENTATION-001

## Purpose

This phase is the first direct experimental bridge between the Abstraction Agent
result and the unified PLM/TAC-OSM architecture.

It does **not** reinterpret TAC-OSM as a memory system and it does not modify the
registered G-CASM-010 experiment. G-CASM-010 asks whether a learned router can
reduce actual executable graph work when the public candidate representation
contains executable structure. This phase asks a different upstream question:

> Can a representation synthesized from a domain description preserve the
> distinctions required for future computation, rather than merely producing a
> compact or semantically plausible geometry?

The distinction is load-bearing:

    Abstraction
        -> discovers useful coordinates

    Addressing
        -> selects relevant persistent state

    Operator routing
        -> selects executable computation

    CASM execution
        -> performs the computation

    Verification
        -> tests whether the representation and computation were actually useful

## Research fusion

The current unified PLM loop becomes:

    D_t
      -> Phi_t
      -> G_t
      -> R_t
      -> C_t
      -> A_t
      -> O_t
      -> V_t
      -> L_t
      -> G_{t+1}

where:

- D_t is domain description, observations, and evidence;
- Phi_t is a discovered/calibrated computational representation;
- G_t is persistent structural state;
- R_t is the relevant persistent state selected for the current computation;
- C_t is the executable operator/structure subset selected for the current task;
- A_t is actual structural computation/action;
- O_t is the environment outcome;
- V_t is external verification;
- L_t is learning, repair, and consolidation.

The semantic representation is therefore an upstream computational substrate,
not a replacement for persistence, routing, execution, or verification.

## Prior-art boundary

### Abstraction Agent

Abstraction Agent demonstrates a four-stage zero-shot pipeline:

    natural-language rules
      -> LLM feature discovery
      -> calibrated feature scoring
      -> numerical filtering
      -> k-means abstraction
      -> solver
      -> exploitability

The paper reports up to 62% lower lifted-strategy exploitability than an
expected-hand-strength baseline on HUNL turn endgames and reports that its
multidimensional representation beats a scalar rank baseline on the novel
ROVER Trials game. It also transfers the unchanged prompting pipeline to
additional games.

What transfers to PLM:

- semantic feature discovery from natural-language rules;
- explicit calibration anchors;
- separation of expensive semantic representation construction from cheap
  downstream numerical computation;
- external downstream evaluation rather than LLM self-confidence.

What does **not** transfer:

- static state geometry is not dynamic behavioral equivalence;
- game abstraction is not persistent state addressing;
- state clustering is not selective operator execution;
- no persistent-state or continual-learning result transfers;
- no asymptotic computation result transfers.

Source: arXiv:2609.04303.

### VERDI

VERDI argues that retrieved experience is an optimization hypothesis until
validated on the target. It constructs optimization fingerprints from shared
inference-time probes, retrieves candidate prior knowledge, validates it using a
frozen target-side verifier, and evolves the probing representation when nearby
cases produce contradictions.

The transferable architectural lesson is:

    representation
      -> retrieval hypothesis
      -> target-side validation
      -> evidence admission
      -> representation/probe refinement

This is highly compatible with PLM's desired verified experience and
representation repair, but VERDI does not establish the representation or
selective computation mechanism used in this experiment.

Source: arXiv:2608.09537.

### Task-conditioned routing signatures

A 2026 study of sparse MoE routing reports that prompts from the same task
category induce similar expert-activation signatures and that these signatures
contain enough information for task classification. This supports a useful
measurement idea for TAC-OSM: operator-selection traces can be stored as
structured telemetry and analyzed as computational state.

The result does **not** establish that such signatures are causal, optimal, or
computationally cheaper. For PLM they should therefore be treated as a
diagnostic representation of the operator-routing surface:

    query/state -> operator router -> routing signature

A later P3 experiment should measure whether the routing signature is stable
under task-preserving perturbations, whether it predicts useful operator
specialization, and whether any compression of the signature preserves actual
execution capability.

Source: arXiv:2603.11114.

### Compositional behavioral semantics

The ICML 2026 framework of Zhang, Luo, and Baltieri gives a more formal
foundation for the dynamic part of this phase. It treats behavioral semantics
compositionally and provides soundness-oriented conditions for transferring
behavioral structure through abstractions.

The relevant implication is that an abstraction should be evaluated by what
behavior it preserves, not only by its geometric compactness.

Source: arXiv:2606.25357; ICML 2026, PMLR 306.

## The central new criterion

Static similarity is insufficient.

For a representation Phi:

    d_Phi(s_i, s_j) <= delta

should not by itself license:

    s_i ~ s_j

for persistent computation.

Instead, the representation must satisfy a registered dynamic condition over
relevant actions:

    d_Phi(s_i, s_j) <= delta
       =>
    d_Phi(T(s_i,a), T(s_j,a)) <= epsilon

for each registered action a, and the relevant downstream outcomes must remain
within the declared tolerance.

The stronger useful concept is therefore **dynamic computational equivalence**:

> Two states are equivalent under a representation only to the extent that
> relevant future computation cannot distinguish them in a way that changes the
> task-relevant outcome.

This is the main PLM extension beyond the static Abstraction Agent setting.

## Representation object

A persistent representation is not just a vector.

Conceptually:

    X = (
        raw_evidence,
        Phi(X),
        calibration_contract,
        representation_version,
        provenance,
        validity,
        uncertainty
    )

The raw evidence is retained below the abstraction so that representation
repair does not destroy the evidence that produced an earlier representation.

The calibration contract contains the feature semantics and reference anchors.

## Calibration

Each synthesized feature f_j receives explicit anchors:

    f_j(a_j^0)   = 0
    f_j(a_j^1/2) = 0.5
    f_j(a_j^1)   = 1

The same anchors are reused across all scoring batches.

The representation is therefore required to have a stable coordinate frame
before any capability or scaling interpretation is permitted.

Parsing success is not semantic correctness. A feature that is perfectly
parseable but semantically wrong must fail downstream validation.

## Experimental arms

### raw_state

Complete router-visible persistent state.

Purpose: capability reference without abstraction.

### scalar_baseline

One scalar relevance/value coordinate.

Purpose: distinguish multidimensional representation from scalar compression.

### semantic_geometry

Frozen multidimensional representation synthesized from the domain description,
with explicit anchors and training-only numerical filtering.

Purpose: the actual representation-synthesis intervention.

### oracle_behavioral

Representation derived from exhaustive future behavior.

Purpose: diagnostic ceiling only. It cannot be promoted as a practical
representation or win the primary endpoint.

## Primary endpoint

At each registered population level:

    min execution-work fraction

among registered abstraction budgets satisfying:

    capability >= 0.80 * raw_state capability

This keeps the representation experiment connected to the central PLM thesis:
compression matters only when downstream computation actually becomes cheaper
without unacceptable capability loss.

## Required secondary measurements

The runner must record at least:

- dynamic equivalence violation rate;
- the fixed pair-set digest and the number of fixed pairs evaluated;
- future capability retention;
- anchor consistency across batches/checkpoints;
- effective representation rank;
- pairwise geometry/concentration statistics;
- state-addressing work;
- actual structural execution work;
- one-time representation construction cost;
- amortized construction cost under an explicit query horizon.

No proxy for graph execution may stand in for the primary work endpoint.

## Proposed implementation sequence

The registered contract deliberately separates this from G-CASM-010.

### Step 1 — Proposal artifact

Generate the representation once from the domain description.

Freeze:

- proposer model identifier;
- prompt;
- prompt hash;
- decoding parameters;
- feature definitions;
- anchor definitions;
- feature-scoring output;
- generation timestamp;
- artifact digest.

No outcome data is visible during proposal generation.

### Step 2 — Representation gate

Before any capability run:

- verify the representation is deterministic;
- verify anchors;
- verify non-degenerate features;
- verify no forbidden target information is encoded;
- run the existing representability gate;
- compute geometry diagnostics.

Failure blocks the confirmatory run.

### Step 3 — Dynamic validity

Generate and hash the state-pair evaluation set **before any representation is scored**. Pair membership cannot depend on the proposed geometry.

The frozen representation then identifies which of those fixed pairs are
near under the registered threshold.

For registered actions and horizons:

    one-step
    k in {1,4,16,32}

execute both states and compare:

- successor representation;
- relevant observable;
- task outcome;
- verifier result.

Every violating pair becomes an explicit counterexample artifact.

### Step 4 — Capability/computation

Only representations that pass the dynamic admission gate enter the
capability-vs-execution measurement.

Compare against raw-state and scalar controls at matched abstraction budgets.

### Step 5 — Repair, later phase

A failed representation is not silently modified.

Instead:

    Phi_t
      -> verifier counterexamples
      -> representation-repair proposal
      -> Phi_{t+1}

and the repaired representation receives a new version and a new evaluation
contract.

## Relation to the current TAC-OSM research

The repository's current evidence remains unchanged.

- C1 bounded temporal persistence remains a synthetic mechanism result.
- C2 bounded relational routing remains a mechanism result.
- C3 representability remains a compulsory gate.
- C4 model-state integrity remains a compulsory gate.
- C5 useful execution scaling remains untested.
- M1 established that the large-H top-1 problem has not been solved, while
  dense reward improved top-K ranking without materially improving top-1.
- M2 G-CASM-010 remains the current executable-graph selective-routing
  experiment.
- The G-CASM-010 identifiability correction remains important: public node-local
  summaries are not sufficient when different hidden wirings are behaviorally
  distinct.

VRS-001 therefore does not replace any existing result. It introduces a new
upstream question:

    Does the representation preserve exactly the distinctions that the
    downstream selective computation needs?

## Later joint-scaling experiment

This phase keeps operator population E fixed.

The later joint experiment should vary:

    M = persistent-state population
    E = executable operator population
    H = accumulated history/horizon

and evaluate:

    C_useful(M,E,H | Phi)

against matched capability.

The central hypothesis remains:

    capability approximately stable
    while selected state/operator work grows more slowly than
    exhaustive M + E + H processing.

Finite-range fits must report raw data and uncertainty and must not be promoted
to asymptotic laws without sufficient scale and model comparison.

## Scientific boundary

This phase does not claim:

- that LLM-discovered geometry is intrinsic;
- that Euclidean distance is universally correct;
- that static state similarity implies future equivalence;
- that semantic representation removes the need for exact evidence;
- that LLM feature scoring is correct because it is parseable;
- that representation synthesis solves scientific discovery;
- that selective execution has asymptotic sublinear complexity.

The intended contribution is narrower and testable:

    semantic representation
      -> dynamic validity
      -> selective computation
      -> external verification
      -> persistent repair

That is the bridge from Abstraction Agent to the unified PLM research program.

## Research addendum — representation refinement and behavioral anchors

Two additional 2026 results sharpen the post-VRS research path.

### LOTUS: temporal/behavioral representation constraints

LOTUS defines universal task representations from temporal-logic structure and
uses a bisimulation metric to provide guarantees around behavioral equivalence,
optimality fidelity, and trajectory robustness. The relevant lesson for PLM is
not the specific LTL encoder. It is that representation validity can be stated
as a **behavioral constraint**, rather than as geometric similarity alone.

This suggests a second kind of anchor for future PLM representations:

    point anchor:
        Phi(s_anchor) = z_anchor

    behavioral anchor:
        B(Phi, s_anchor, a, horizon) = declared_property

A behavioral anchor can assert, for example, that two states represented as
equivalent must retain the same admissible action/outcome relation over a
registered horizon.

Source: arXiv:2608.15509.

### RefineICL: representation refinement without parameter updates

RefineICL treats support examples as instructions for updating an episode's
representations while leaving model parameters fixed. Its interventions report
that removing an intermediate support update degrades later query behavior,
which makes the evolving representation itself a causal object of the model's
computation rather than merely a passive cache.

The PLM implication is a useful two-timescale separation:

    slow representation structure:
        Phi_shared

    fast contextual/state refinement:
        DeltaPhi_t

with persistent consolidation deciding when a fast refinement becomes part of
shared structure. This fits the PLM distinction between slow shared structure
and fast specialist adaptation without requiring global parameter updates at
every decision.

Source: arXiv:2609.27679.

### Error-driven feature synthesis

The September 2026 feature-engineering work strengthens the same principle from
a different direction: the generator LLM proposes features, a separate
extractor materializes them, and concrete downstream errors such as ranking
inversions are translated into natural-language feedback for the next proposal.
The reported framework therefore closes a practical loop:

    proposal -> extraction -> downstream error -> targeted feature repair

The PLM analogue should preserve the error itself as structured evidence before
converting it into any representation update. In particular, a failed task
should not directly rewrite Phi. It should produce a counterexample record:

    (Phi_version, state_pair, action, horizon, observed_divergence, verifier)

and only a later repair stage may propose Phi'.

Source: arXiv:2609.21894.

### Consequence for VRS

VRS-001 should therefore be treated as the **static-to-dynamic admission
experiment**. It does not yet test representation learning across time.

The next representation experiment should test:

    Phi_t
      -> verifier counterexamples
      -> targeted refinement DeltaPhi_t
      -> dynamic re-validation
      -> selective persistence/consolidation

with an explicit no-regression condition: a repaired representation must retain
previously verified behavioral invariants unless the experiment explicitly
registers their replacement.

This is the representation-level analogue of verifier-gated state writes in the
existing PLM substrate.


## Instrument amendment A2 — non-vacuous dynamic validation

The dynamic-validity gate cannot pass when the representation produces zero
applicable near pairs or zero evaluated transitions. Such a result means the
instrument tested nothing about future behavior, not that the representation
was valid.

The implementation therefore requires:

    near_pairs >= 1
    checked_transitions >= 1

and requires the fixed action schedule length to equal the declared horizon.
This is an instrument-integrity amendment made before any confirmatory VRS
measurement; no scientific outcome was observed under the superseded rule.
