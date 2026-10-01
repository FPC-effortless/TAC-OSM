# TACOSM-SEMANTIC-PROGRAM-REP-003

Status: PREREGISTERED — result pending.

## Purpose

REP-002 established that explicit candidate executable topology removes the descriptor-only observation collision, but its query directly specified the target edge mask. REP-003 increases the difficulty without changing the candidate boundary:

- the candidate exposes its explicit executable topology;
- the query specifies an abstract semantic dependency signature;
- the learner must infer that signature from graph structure.

The target is therefore not selected by exact edge-mask equality.

## Semantic task

The fixed five-bit query signature is [source_0, source_1, source_2, depth_1, depth_2].

It describes which input reaches the program output and whether that dependency path is direct or passes through the intermediate node.

The candidate substrate is the same seven-edge graph family used in REP-002. There are 12 executable topologies. For each task, distractors are sampled so that exactly one candidate has the queried semantic signature. Other non-target semantic classes may occur more than once.

The query contains the semantic signature only; it does not contain the candidate edge mask, candidate index, action, target field, outcome, or oracle structure.

## Learner

Condition B uses a small two-layer tanh candidate encoder: 7 edge features -> 16 hidden -> 8 latent.

The query encoder is linear: 5 query features -> 8 latent.

Routing is dot-product similarity followed by deterministic Top-1 selection.

Training uses only verified success as the positive signal and the strongest in-pool competitor as the negative. No target index is passed to the learner.

Condition C uses the identical architecture and initialization protocol but performs no updates.

## Protocol

- seeds: 0, 1, 2, 3, 4;
- training episodes: 512 per seed;
- held-out episodes: 256 per seed;
- candidates: 8;
- query width: 5;
- candidate edge width: 7;
- hidden width: 16;
- latent width: 8;
- learning rate: 0.01;
- margin: 0.1;
- executor: exact explicit graph executor.

These values are fixed before the run. No tuning is part of the registered result.

## Conditions

A. Exact semantic structural matcher — representability witness only.

B. Learned graph-program router — outcome-success pairwise updates.

C. No-learning graph-program router — frozen random initialization.

D. Oracle selection — environment-side target, outside the router boundary.

## Primary diagnostics

1. Semantic uniqueness: exactly one candidate matches the queried semantic signature.
2. Representability: A selects the target at Top-1 on held-out tasks.
3. Learning: B versus C on held-out Top-1 and mean target rank.
4. Execution integrity: selected program executes exactly its declared edge set.
5. Permutation invariance: candidate action indices do not change scores.

## Falsifiers

- semantic target is not unique in the candidate pool;
- analytic A fails;
- B does not exceed C on held-out Top-1;
- action permutation changes scores;
- declared topology and exact execution disagree.

## Interpretation boundary

A positive B result would establish only that a small learned graph encoder can exploit explicit executable topology for this synthetic semantic dependency task.

It would not establish general graph reasoning, semantic retrieval at scale, sublinear addressing, selective execution scaling, PLM/PNDS validity, or real-world capability.

The experiment is intentionally a bridge between exact topology matching in REP-002 and later structural relevance experiments.

## Cost accounting

For the registered learned router, one routing decision is approximately:

C_route = 8*5 + H*(16*7 + 8*16 + 8) MACs

so at H=8, C_route = 2,024 MACs before Python overhead, nonlinear operations, candidate materialization, verification, and execution.

The cost remains O(H). REP-003 is a representation/selection experiment, not a sparse-compute claim.

## Provenance

Base: REP-002 explicit-program routing branch after the completed TACOSM-IDENTIFIABILITY-REP-002 run.

REP-001 and C5 evidence remain frozen and are not recomputed here.