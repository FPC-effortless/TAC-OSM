# TACOSM-STATE-REP-007

Status: PREREGISTERED — result pending.

## Purpose

REP-006 showed that a learned addressor can identify an opaque persistent state item at M=8. REP-007 asks whether that effect survives larger state pools.

Only the state population M changes. The semantic task, candidate topology, learner, training schedule, and temporal boundary remain fixed.

## Registered levels

M = {2, 4, 8, 16, 32} persistent state items.

Each episode contains exactly one state item whose five-bit semantic signature matches the query. All state addresses are opaque randomized keys.

## Protocol

- seeds: 0, 1, 2, 3, 4;
- 512 training episodes/seed/level;
- 256 held-out episodes/seed/level;
- one-step causal write/read delay;
- 8 candidate programs;
- 5-bit semantic requirement;
- 5 -> 8 bilinear learned addressor;
- learning rate 0.01;
- margin 0.1;
- exact downstream structural matcher.

No hyperparameter search is permitted.

## Conditions

A. Analytic identity addressor.
B. Learned addressor.
C. No-learning addressor.

## Cost

The registered learned addressing arithmetic is:

`C_state_address = 8*5 + M*(8*5 + 8)`

= `40 + 48M` MACs per decision, before dictionary reads and software overhead.

The implementation scans the state pool, so addressing is O(M).

| M | Address MACs |
|---:|---:|
| 2 | 136 |
| 4 | 232 |
| 8 | 424 |
| 16 | 808 |
| 32 | 1,576 |

These are arithmetic counts, not measured hardware FLOPs.

## Primary endpoint

Held-out state Top-1 recall, comparing learned B with frozen C at each M.

Chance is 1/M for uniform random selection.

The experiment reports the full M curve rather than reducing the outcome to one single large-M number.

## Interpretation boundary

A positive scaling curve would show only that this small semantic addressor continues to exploit stored semantic content under the registered synthetic population range.

It would not show sublinear addressing, learned indexes, arbitrary-address semantic memory, long-horizon persistence, natural-language memory, selective computation, or C5 capability/computation parity.

C5 remains UNTESTED.