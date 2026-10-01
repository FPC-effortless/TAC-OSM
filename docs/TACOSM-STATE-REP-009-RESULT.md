# TACOSM-STATE-REP-009 RESULT

## Run provenance

- workflow: `36666210591`
- run: #7
- code head: `8eee75f85f280fd6911c7569b71253be91d1a3b8`
- precondition gate: passed
- full test suite: **858 passed**
- experiment step: passed
- artifact: `TACOSM-STATE-REP-009`
- artifact id: `11075409253`
- artifact digest: `sha256:1e92da09dfd1f524a4c9c529b91b9c2e95852d8241889b6c84c63f8a09bdcd00`

## Raw result

One-bit-noisy 10-bit state queries were evaluated against a fixed opaque state
pool. The approximate index used Hamming radius 2 and retained at most K=4
state items. The same learned 10->8 bilinear addressor was evaluated over the
full pool and over the indexed shortlist.

| M | Learned full Top-1 | Learned indexed Top-1 | Index target recall | Mean K | Full scorer MACs | Indexed scorer MACs |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.2914 | 0.8594 | 1.0000 | 1.29 | 784 | 193.9 |
| 16 | 0.1094 | 0.7461 | 1.0000 | 1.67 | 1,488 | 226.6 |
| 32 | 0.0758 | 0.5078 | 1.0000 | 2.46 | 2,896 | 296.9 |
| 64 | 0.0297 | 0.2320 | 1.0000 | 3.69 | 5,712 | 404.9 |

Chance declines from 1/8 to 1/64 across the registered levels.

## Cost reduction

The learned scorer's arithmetic reduction, using the measured MAC accounting,
is:

| M | Reduction in scorer MACs | Full / indexed |
|---:|---:|---:|
| 8 | 75.3% | 4.04x |
| 16 | 84.8% | 6.57x |
| 32 | 89.7% | 9.75x |
| 64 | 92.9% | 14.11x |

The index itself performs 56 Hamming bucket probes per query. Those dictionary
operations are reported separately rather than converted into fake FLOPs.
Index construction costs M state reads and is amortized across 256 queries on a
fixed evaluation pool.

## Capability effect

The intervention did more than preserve capability: the same learned scorer
performed substantially better after candidate-state pruning.

Indexed minus full learned Top-1:

- M=8: +0.56797
- M=16: +0.63672
- M=32: +0.43203
- M=64: +0.20234

This is an important diagnostic. The current learned scorer is not merely
compute-heavy; it is also sensitive to the number of distractor state items.
The cheap index acts as both a computational filter and a hard-negative filter.

## Preregistration interpretation

The preregistered primary criterion required indexed performance to remain
within an absolute 0.05 of the full learned arm. That criterion did not fire
because the intervention produced a much larger positive change.

This is **not** evidence that the index failed. It means the registered rule was
written as absolute closeness rather than one-sided non-inferiority, so it did
not specify what to do with a large beneficial capability shift.

The raw data remain valid and are not reclassified as confirmatory support under
a modified rule.

Future selective-index preregistrations should use a one-sided preservation
criterion such as:

`indexed_top1 >= full_top1 - 0.05`

while separately reporting any material positive capability delta.

## What the experiment establishes

Under this synthetic noisy-code task, a deterministic approximate state index can
reduce the number of state items presented to the learned scorer from M to at
most 4 while retaining the target in 100% of tested queries.

The retained set becomes increasingly small relative to the state pool:

- M=8: mean K/M = 0.162
- M=16: mean K/M = 0.104
- M=32: mean K/M = 0.077
- M=64: mean K/M = 0.058

That is a bounded selective-addressing result.

## What remains open

The index is hand-designed over a 10-bit Hamming code and therefore does not
establish a learned semantic index.

The task also does not measure executable work as a function of a real relevant
subset R. Therefore C5 remains **UNTESTED**.

Still open:

- dynamic index maintenance under writes;
- natural-language or continuous semantic representations;
- learned approximate indexing;
- capability-matched long-history computation;
- `C_execute(|R|)` versus full-history capability parity.

## Research consequence

The next implementation should move this filter into the actual hardened runtime
boundary and expose two separate costs:

`C_state_index/query + C_candidate_route(K) + C_execute(|R|) + C_verify`

versus:

`C_full_history`.

That is the measurement needed to connect the successful state-addressing work
to the repository's actual C5 claim without conflating retrieval cost with
execution cost.

REP-008, REP-007, REP-006 and all earlier REP/C5 evidence remain frozen.