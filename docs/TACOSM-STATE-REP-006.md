# TACOSM-STATE-REP-006

Status: PREREGISTERED — result pending.

## Purpose

REP-005 demonstrated `known address -> persistent state read -> program routing`.
REP-006 removes the supplied address. The query contains a semantic requirement,
the persistent store contains multiple state items at opaque randomized keys,
and the system must identify the relevant state item before program selection.

The primary boundary is therefore:

`query -> semantic state address -> state read -> program route`

## State pool

Each episode contains 8 persistent state items.

- Keys are opaque randomized `mem-<hex>` identifiers.
- Each value is a five-bit semantic dependency signature.
- Exactly one state item has the queried target signature.
- Other state items use one of the non-target semantic signatures.
- The target address is never placed in the query.

The query contains only the five-bit semantic requirement.

## Temporal protocol

All 8 state items are staged at step `t` and become readable at `t+1`.
The router is invoked only after the temporal boundary.

State addressing is intentionally implemented as an exhaustive semantic scan
over the currently exposed state addresses. This fixes the functional boundary
without claiming a sublinear index.

## State addressor

The learned addressor is a small bilinear matcher:

`5 query features -> 8 latent`
`5 state features -> 8 latent`
`dot-product score over state items`

Training uses the same verified-success / strongest-negative update pattern
used by the earlier routing experiments.

Condition C is the identical model with no parameter updates.

Condition A uses identity embeddings as the exact representability witness.

## Program-routing isolation

After the state item is selected, the candidate program is selected with the
exact semantic structural matcher from REP-003 rather than the learned graph
router. Therefore the primary end-to-end failure mode is state addressing, not
program-router optimization.

## Protocol

- seeds: 0, 1, 2, 3, 4;
- training episodes: 512 per seed;
- held-out episodes: 256 per seed;
- state items: 8;
- candidates: 8;
- delay: 1 decision boundary;
- query: five-bit semantic signature, no address;
- state addresses: opaque randomized keys;
- addressor: 5 -> 8 bilinear learned matcher;
- learning rate: 0.01;
- margin: 0.1;
- downstream program matcher: exact structural witness.

## Conditions

A. Analytic state-addressing witness.
B. Learned state addressor.
C. No-learning state addressor.
D. Value-shuffle intervention.
E. Reset intervention.

## Primary diagnostics

1. Held-out Top-1 state-address recall.
2. End-to-end Top-1 after exact program selection.
3. B versus C.
4. Mean state items inspected and state pool size.
5. Address-preserving value shuffle.
6. Reset fail-closed behavior.

## Falsifiers

- target state signature is not unique;
- analytic state matcher fails;
- learned B does not exceed C;
- changing values while preserving addresses fails the registered shuffle control;
- cleared state is accepted instead of failing closed;
- prior evidence is modified.

## Cost boundary

At state-pool size M and latent dimension d=8:

`C_state_address = 8*5 + M*(8*5 + 8)` MACs

At M=8 this is 424 MACs before dictionary reads, comparisons, and Python
overhead.

Crucially, this is an exhaustive semantic state scan: O(M). It is not an
indexed/sublinear addressing result.

Program routing remains the exact structural witness and is not included in
the learned state-addressing cost.

## Interpretation boundary

A positive result establishes only that the current architecture can identify
a relevant persistent state item from its stored semantic content when the
address is opaque, then use that item for an exact downstream program choice.

It does not establish large-scale semantic memory retrieval, learned long-
horizon memory, sublinear addressing, selective hardware computation, PLM/PNDS
validity, or general intelligence.

REP-005 and all earlier evidence remain frozen.