# TACOSM-IDENTIFIABILITY-REP-002

**Status: PREREGISTERED — no capability result yet.**

## Purpose

Test the information boundary implicated by REP-001 without changing the
frozen REP-001 protocol.

REP-001's executable topology is carried downstream as a Program, while a
router-visible Candidate contains only a descriptor. REP-002 makes the
candidate's own executable edge set explicit and asks whether that removes the
candidate-observation collision.

This is an identifiability diagnostic, not a general capability experiment.

## Intervention

The intervention is at the candidate representation boundary:

Candidate = (descriptor, executable_edges)

The executable edge set is candidate-local. Every candidate has a valid
program from the same fixed substrate, with the same node types and the same
descriptor. The router sees the candidate's own program topology, not an
environment-side gold field, target index, outcome, or correct action.

The legacy descriptor-only representation is retained as the collision
control.

## Fixed task

The topology substrate has seven possible edge positions:

1. three possible inputs to unary node 3;
2. four possible inputs to unary node 4, including node 3.

A program chooses exactly one edge for node 3 and one for node 4, giving
12 valid topologies. Each task contains eight distinct candidate programs.

All candidates have descriptor (0, ..., 0). Candidate action indices do not
encode the target.

The public query contains the desired topology's seven-bit edge mask and no
candidate index. The desired relation is therefore explicit query-to-program
matching once the executable candidate structure is exposed.

## Protocol

- seeds: 0, 1, 2, 3, 4;
- training episodes: 128 per seed;
- held-out episodes: 128 per seed;
- candidates: 8;
- exact explicit-graph executor;
- Top-1 selection;
- input/latent dimension: 7/7;
- learning rate: 0.01;
- margin: 0.1.

These values are fixed before the run. No hyperparameter search is part of the
result.

## Conditions

A. Analytic identity witness: fixed identity embeddings; representability only.

B. Learned explicit-program router: random initialization, outcome-success
pairwise updates against the strongest in-pool competitor.

C. No-learning explicit-program router: random initialization, no parameter
updates.

D. Oracle: environment-side target selection, outside the router boundary.

## Primary diagnostics

### 1. Observation collision

The descriptor-only representation must collapse all eight candidates to the
same observable candidate score, while explicit executable-topology
observations must produce eight distinct topology masks.

This directly tests whether the intervention changes the information available
to the router.

### 2. Representability

Condition A must rank the target first on the held-out tasks. For distinct
two-edge topology masks in this task family, the identity witness has a
minimum hard-negative score margin of 4.

### 3. Learning

Compare B against C on held-out Top-1 recall. B is the only arm whose
parameters are updated.

### 4. Execution integrity

The selected candidate is materialised as a Program with that candidate's
declared executable edges. Exact execution must report those same edges as
active.

### 5. Permutation invariance

Changing Candidate.action values must not change the explicit topology
observation or its scores.

## Falsifiers

- distinct candidate programs still collide under explicit topology
  observation;
- the analytic identity witness fails;
- learned B does not exceed frozen C on held-out Top-1 recall;
- action-index permutation changes explicit topology observations or scores;
- executor gates do not match the candidate's declared edge set.

A learning failure does not erase an observation-boundary result. It means the
program topology is observable but the registered learner does not recover the
matching relation.

## What REP-002 can establish

At most, this experiment can establish a bounded synthetic result that the
explicit-program candidate boundary distinguishes structures that the
descriptor-only boundary cannot distinguish, and optionally that the registered
learner can exploit that information.

## Non-claims

REP-002 does not establish:

- sublinear retrieval;
- semantic addressing;
- general compositional generalization;
- learned sparse execution at hardware scale;
- PLM validity;
- hardware FLOP superiority;
- economic superiority;
- general intelligence.

C5-001/C5-002/C5-003 evidence remains frozen.

## Provenance and freeze

The branch is based on Successor Architecture v1. REP-001 remains unchanged.
No C5 evidence, historical result, or baseline experiment is recomputed into
this diagnostic.
