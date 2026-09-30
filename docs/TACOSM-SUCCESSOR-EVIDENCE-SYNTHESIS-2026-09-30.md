# TAC-OSM successor evidence synthesis — 2026-09-30

This document consolidates completed successor-architecture diagnostics. It does
not replace individual preregistrations or frozen result artifacts.

## Evidence ladder

| Experiment | Result | Interpretation boundary |
|---|---|---|
| REP-001 | Analytic=1.000; learned≈no-learning≈0.18; execution=1.000 | Candidate observations were insufficient for the tested topology distinction. |
| REP-002 | 8/8 topology observations distinct; analytic=1.000; learned=0.2219; no-learning=0.1266 | Explicit executable topology makes the tested relation identifiable and partially learnable. |
| REP-003 | learned=0.2008; no-learning=0.0977; analytic=1.000 | Some semantic structural signal is learned, but seed stability is limited. |
| REP-004 | exploration=0.2094 vs baseline=0.2008 pooled | Simple epsilon-greedy exploration is not the main repair. |
| REP-005 | learned=0.1633; no-learning=0.1023; analytic=1.000; reset=160/160 fail-closed | The tested semantic requirement survives one causal persistent-state boundary. |
| REP-006 | learned state Top-1 pooled=0.7047 vs no-learning=0.1859; reset 5/5 fail-closed | Bounded learned semantic addressing over an 8-item opaque state pool. |
| REP-007 | learned > no-learning at M=2,4,8,16,32, but recall falls from .992 to .334 | Learned state addressing survives bounded population scaling but is not scale-invariant. |
| REP-008 | exact discrete state index recall=1.000 at M=2..32; query lookup is O(1) after build | Demonstrates an exact indexing ceiling/control, not learned semantic retrieval. |
| REP-009 | noisy Hamming index recall=1.000; learned scorer retained in a bounded shortlist with 75.3–92.9% scorer arithmetic reduction | Hand-designed approximate indexing can create a runtime retention boundary. |
| SELECTIVE-001 | indexed success=1.000 across H=8,64,256; router input bounded at K=2/4 | Routing-side retention boundary is supported in the registered runtime. |
| C5-EXEC-001 | selective executor work=36 vs exhaustive 576/1152/2304 for H=64/128/256; capability=1.000 | Direct bounded evidence that execution work can depend on R rather than H in the synthetic executor. |
| C5-END-TO-END-001 | capability=1.000 for both arms; selective work=48 vs 768/1536/3072 | Bounded end-to-end relation \x60persistent state -> relevant subset -> execute(R)\x60 with exact indexes. |
| C5-LEARNED-STATE-001 | direct target-state retention=0.170 pooled; capability rule fails in 12/15 cells | Current learned binary state index does not replace the hand-designed state index. |

## Architecture decomposition

The evidence now separates:

1. candidate observability;
2. semantic structural representability;
3. training dynamics;
4. temporal state transport;
5. semantic state addressing;
6. approximate indexing;
7. selective routing;
8. selective execution;
9. combined end-to-end execution;
10. learned-vs-index quantization as a new failure boundary.

The important distinction is between **semantic representation quality** and
**retrieval/indexing quality**. C5-LEARNED-STATE-001 failed at the latter under
its registered binary quantization, while earlier REP-006 showed that learned
continuous semantic addressing can work over a smaller pool.

## Current compute model

For the learned state index:

- query encoder: 80 MACs/query;
- state-index construction: 5,120 MACs/build for M=64;
- amortized state-build arithmetic over a 100-query cell: 51.2 MACs/query;
- binary lookup: 9 Hamming probes at radius 1 over an 8-bit code.

These quantities are not interchangeable with the synthetic executor work
units.

For the bounded C5 execution path:

\x60C_total ≈ C_address + C_candidate_retrieval + C_execute(R) + C_verify\x60.

The current positive C5 measurements establish only the synthetic execution
term and exact/selective addressing controls. They do not establish that a
learned semantic addressor provides the needed sublinear boundary.

## Current failure boundary

C5-LEARNED-STATE-001 is informative because the continuous learned model and
the binary indexed model can now be separated experimentally.

The confirmatory run used 48 training codes and 16 disjoint evaluation target
codes. The learned state index produced only 0.06--0.26 target-address
retention by seed. The same retention was observed across H because the
amended task stream keeps query/state generation fixed across history levels.

The downstream aggregate capability rate rises with H in this synthetic task,
despite stable state-retention rates. That indicates the executor's aggregate
output can alias some wrong retrieved states. Consequently, **target-state
retention is the load-bearing endpoint for this experiment**, not the raw
downstream capability curve.

## C5 status

Broad C5 remains **ungraded**.

Supported bounded components:

\x60persistent state -> selective retention -> execute(R)\x60.

Not established:

- learned semantic sublinear state addressing;
- learned sublinear candidate retrieval;
- capability/computation parity on realistic workloads;
- hardware latency/FLOP superiority;
- long-horizon persistent learning;
- natural-language semantic program selection.

## Next measurement boundary

The immediate diagnostic is to take the **same trained encoder** from
C5-LEARNED-STATE-001 and compare:

\x60continuous semantic top-1 over M=64\x60  
versus  
\x608-bit binary index + radius-1 probe + K=1\x60.

This isolates whether the loss occurs in the learned representation itself or
at the continuous-to-discrete retrieval boundary. The result should precede
any learned candidate-index experiment.

## Provenance

The following evidence remains frozen and independently reproducible:

REP-001 through REP-009, SELECTIVE-001, C5-EXEC-001, C5-END-TO-END-001, and
C5-LEARNED-STATE-001.
