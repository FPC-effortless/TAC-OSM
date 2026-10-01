# TACOSM-SELECTIVE-001 RESULT

## Current registered run

- workflow: `36666455184`
- measurement commit: `0ef94bed1ffda74e4a10c46cfc34aeb50c311ba7`
- full test gate: **858 passed**
- measurement gate: **PASS**
- artifact: `TACOSM-SELECTIVE-001`
- artifact id: `11075954570`
- artifact zip SHA-256: `c0cf905f7d5c874654d13d9a599cb9f099ae4ffe216e26b7720c67e4abdf6f00`
- configuration: H={8,64,256}, K={2,4}, seeds 0–4, 100 queries per cell.

## Primary endpoint

| H | K | Exhaustive success | Indexed success | Indexed router candidates/query | Indexed amortized candidate work |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 1.0000 | 1.0000 | 2.00 | 4.08 |
| 8 | 4 | 1.0000 | 1.0000 | 2.00 | 4.08 |
| 64 | 2 | 1.0000 | 1.0000 | 2.00 | 4.64 |
| 64 | 4 | 1.0000 | 1.0000 | 4.00 | 6.64 |
| 256 | 2 | 1.0000 | 1.0000 | 2.00 | 6.56 |
| 256 | 4 | 1.0000 | 1.0000 | 4.00 | 8.56 |

All six indexed cells preserved the exhaustive 1.0000 task capability.

## Runtime boundary

The exhaustive arm routes over H candidates each query. The indexed arm builds
an exact public equality index once for the static candidate population and
then presents only the retained bucket to the router.

At H=256 the router therefore receives only 2 or 4 candidates instead of 256.

The one-time index build remains included in `amortized_candidate_work` over
the declared 100 queries. Query-time addressing inspects the two marked
positions.

## Interpretation

This establishes a bounded **selective routing boundary** in the hardened
runtime for the registered equality control.

It does not establish the full C5 claim because execution still performs one
selected-candidate invocation per task. The executor's own work therefore does
not yet scale with the retained subset R.

It also does not establish semantic learned retrieval or hardware latency
improvement. The indexed control uses exact content addressing, and the
synthetic executor is deliberately tiny.

## Relation to C5

SELECTIVE-001 establishes:

`H -> retain R -> route(R)`.

It does not yet establish:

`H -> retain R -> execute(R)`.

That distinction is the next experimental boundary. A C5-facing experiment
must require the correct result to depend on aggregating over all relevant
candidate programs, so the exhaustive arm executes H programs and the indexed
arm executes only R programs under matched capability.

## Frozen evidence

REP-001 through REP-009 and prior C5 measurements remain separate historical
records. This document supersedes the earlier SELECTIVE-001 result only as the
latest recorded run; it does not rewrite the frozen baseline.