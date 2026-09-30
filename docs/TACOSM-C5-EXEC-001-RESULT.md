# TACOSM-C5-EXEC-001 RESULT

## Run provenance

- workflow: `36666872579`
- clean registered run: #3
- code head: `625ee3311b200ed7eae22879bd3c34f97f4a39b8`
- full test gate: **866 passed**
- measurement step: **passed**
- artifact upload: **passed**

## Workload

Each candidate universe contains H={64,128,256} programs.

The first six descriptor bits define the relevance class. The generator keeps
exactly four relevant programs for every query at every H. The remaining
programs are distractors.

The correct output is the aggregate contribution of all four relevant programs.
Therefore executing only a single candidate cannot solve the task.

## Capability parity

| H | Exhaustive success | Selective success | R |
|---:|---:|---:|---:|
| 64 | 1.0000 | 1.0000 | 4 |
| 128 | 1.0000 | 1.0000 | 4 |
| 256 | 1.0000 | 1.0000 | 4 |

All 15 seed/H cells in both arms solved the task exactly.

## Execution work

Each candidate program has exactly 9 registered execution work units.

| H | Exhaustive executor calls/query | Selective executor calls/query | Exhaustive work/query | Selective work/query | Work reduction | R/H |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 4 | 576 | 36 | 93.75% | 0.0625 |
| 128 | 128 | 4 | 1,152 | 36 | 96.875% | 0.03125 |
| 256 | 256 | 4 | 2,304 | 36 | 98.4375% | 0.015625 |

The final R/H value is a ratio; the execution-work reduction is 98.4375%.

## Addressing boundary

The selective arm builds an exact content-address index once per static
candidate population and reuses it for 100 queries.

At each query:

- six marked positions are inspected;
- exactly four candidate indices are retained;
- exactly four candidate programs are executed.

The index build is O(H) but amortized across repeated queries and reported
separately from execution work.

## What this establishes

This experiment provides bounded direct evidence for the execution-side
selective-computation relation:

`C_execute(selective) = O(R)`

while the exhaustive control executes:

`C_execute(exhaustive) = O(H)`.

Because R=4 is held fixed while H increases from 64 to 256, the selective
execution work remains constant while the exhaustive work increases linearly.

Capability parity is exact in this registered workload.

## What this does not establish

The result is not the full L4 C5 claim.

It does not establish:

- learned semantic indexing;
- natural-language relevance;
- long-horizon persistent computation;
- hardware FLOP/latency superiority;
- an end-to-end cost curve including realistic verifier and index-maintenance costs;
- general capability at large H.

The relevance relation is synthetic and publicly indexable, and the executor
is a fixed control program.

## Research consequence

The next required bridge is to combine the already-measured semantic state
addressing from REP-009 with this execution-side boundary:

`query -> semantic state index -> retained R -> execute R -> verify`

against a full-history control:

`query -> inspect H -> execute H -> verify`.

The final C5 measurement must preserve capability parity while recording:

`C_address + C_route + C_execute + C_verify`

for both paths.

C5 remains **UNTESTED as a broad program claim**, while C15 records the
execution-side mechanism as supported and bounded.