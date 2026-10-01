# TACOSM-C5-EXEC-001

Status: PREREGISTERED — result pending.

## Purpose

SELECTIVE-001 demonstrated that the hardened runtime can reduce the number of
candidates presented to a router while preserving capability. It did not test
whether actual execution work scales with the retained subset.

C5-EXEC-001 isolates that missing boundary.

## Workload

Each static candidate universe contains H candidates.

The first six descriptor bits define relevance. The population is constructed
so that every query has exactly four relevant candidates, regardless of H.

H = 64, 128, 256.

The query rotates across 16 public six-bit relevance classes.

The four relevant candidates in each class carry the four two-bit payload
variants. A fixed candidate program returns a contribution derived from its
descriptor; the correct task output is the aggregate contribution of the four
relevant programs.

Therefore the correct answer requires all four relevant programs to be executed.

## Arms

Exhaustive:

`execute H candidate programs -> aggregate`.

Selective:

`exact content-address index -> retain R=4 -> execute 4 programs -> aggregate`.

The same `SelectiveProgramExecutor` is used in both arms.

## Registered execution work

Each candidate execution is assigned exactly 9 fixed work units.

`W_exhaustive = 9H`

`W_selective = 9R = 36`

Expected execution work:

| H | Exhaustive | Selective | Reduction |
|---:|---:|---:|---:|
| 64 | 576 | 36 | 93.75% |
| 128 | 1,152 | 36 | 96.875% |
| 256 | 2,304 | 36 | 98.4375% |

These are controlled operation units, not hardware FLOPs.

## Addressing cost

The selective arm builds its exact index once per static population.
Query-time addressing inspects six marked positions and returns exactly four
candidate indices.

Index construction is O(H), but it is amortized across 100 queries and is
reported separately from execution work.

## Protocol

- seeds: 0,1,2,3,4;
- H: 64,128,256;
- K: 4;
- R: 4;
- 100 repeated queries per seed/H cell;
- same candidate universe reused within each cell;
- learning disabled;
- repair disabled;
- exact public equality index;
- exact aggregate evaluator.

## Primary endpoint

Indexed success rate must equal exhaustive success rate at every H.

Secondary endpoints must show:

- exhaustive execution work = 9H;
- selective execution work = 36;
- retained candidate count = 4;
- retained fraction = 4/H.

## Interpretation

A passing result establishes a bounded synthetic mechanism result:

`C_execute(R) < C_execute(H)`

while preserving capability.

It does not establish the full C5 claim because this is still a synthetic
control with an exact public content-address index and a fixed executor.

## Next step

After this boundary passes, the next experiment must replace the exact equality
index with the semantic/noisy state index already validated in REP-009, then
measure the complete:

`C_address + C_route(K) + C_execute(R) + C_verify`

versus:

`C_full(H)`

on a capability-matched workload.