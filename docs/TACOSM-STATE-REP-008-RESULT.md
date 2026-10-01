# TACOSM-STATE-REP-008 RESULT

## Run provenance

- workflow: `36665595709`
- code head: `44479b4be2ec88262fd4a4d19aa1d480d6092561`
- precondition gate: passed;
- registered ceiling run: passed;
- artifact: `TACOSM-STATE-REP-008`;
- artifact id: `11075568222`;
- artifact zip SHA-256: `a5f047ca73131d559d91a0285d39cad3292d9b05014830e746110350a2673b89`.

## Result

An exact semantic inverted index was built once from each state pool and then
queried 256 times. The query contained the semantic signature and no address.

| M | Exact-index recall | Full-scan recall | Full-scan MACs/query | Index build reads | Indexed query ops | Build reads/query |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 0.400 | 136 | 2 | 1 | 0.0078 |
| 4 | 1.000 | 0.200 | 232 | 4 | 1 | 0.0156 |
| 8 | 1.000 | 0.000 | 424 | 8 | 1 | 0.0313 |
| 16 | 1.000 | 0.000 | 808 | 16 | 1 | 0.0625 |
| 32 | 1.000 | 0.000 | 1,576 | 32 | 1 | 0.1250 |

The exact index achieved 1.000 recall at every registered population size.
The query-time lookup performed exactly one dictionary operation per query.
Index construction required one state read per stored item.

## Important control limitation

The full-scan comparison arm in REP-008 intentionally uses the existing
untrained bilinear addressor. Its recall therefore falls to zero for M>=8.
This makes it a **cost/control comparison**, not a capability-matched baseline.

It would be invalid to conclude from these cells that indexing is more capable
than the learned addressor. REP-007 already measured the trained addressor's
capability curve; REP-008 measures what query-time cost looks like when an exact
index is allowed to do the retrieval work.

## What the experiment establishes

Within the registered synthetic representation, semantic state addressing can
be decomposed into:

`one-time O(M) index construction`
`constant-time query lookup`

while preserving exact target-state recall.

This is an upper-bound ceiling for an indexable representation. It demonstrates
that O(M) query-time scanning is not mathematically required by the state
representation itself.

## What it does not establish

The result does not establish a learned semantic index, robustness to noisy or
approximate queries, dynamic index maintenance, or sublinear end-to-end TAC-OSM
computation.

In particular, the exact hash lookup is unusually favorable because the query
contains the same discrete semantic signature used as the index key.

## Architectural consequence

The next nontrivial experiment should keep the exact-index cost boundary but
introduce approximate semantic queries so exact hash matching is insufficient.
The index must return a small shortlist and a downstream scorer must preserve
capability.

That is the first experiment in the ladder that can test:

`C_index(M) + C_route(K)` with `K << M`

without confusing an exact lookup ceiling with learned semantic retrieval.

C5 remains UNTESTED.