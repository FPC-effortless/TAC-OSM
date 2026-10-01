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
| REP-008 | exact discrete state index recall=1.000 at M=2..32; query lookup is O(1) after build | Exact discrete indexing control, not learned semantic retrieval. |
| REP-009 | noisy Hamming index recall=1.000; learned scorer retained in bounded shortlist with 75.3–92.9% scorer arithmetic reduction | Hand-designed approximate indexing can create a runtime retention boundary. |
| SELECTIVE-001 | indexed success=1.000 across H=8,64,256; router input bounded at K=2/4 | Routing-side retention boundary is supported in the registered runtime. |
| C5-EXEC-001 | selective executor work=36 vs exhaustive 576/1152/2304 for H=64/128/256; capability=1.000 | Direct bounded evidence that execution work can depend on R rather than H in the synthetic executor. |
| C5-END-TO-END-001 | capability=1.000 for both arms; selective work=48 vs 768/1536/3072 | Bounded end-to-end relation \`persistent state -> relevant subset -> execute(R)\` with exact indexes. |
| C5-LEARNED-STATE-001 | target-state retention=0.170 pooled; capability rule failed in 12/15 cells | Current learned binary state index does not replace the hand-designed state index. |
| C5-LEARNED-STATE-DIAG-001 | continuous learned recall=0.214 vs random=0.004; binary recall=0.170; continuous-binary gap=0.044 | Learned encoder contains real held-out signal; binary quantization is not the dominant current failure. |
| C5-LEARNED-STATE-BUDGET-001 | mean continuous recall 0.214 -> 0.314 -> 0.272 at 32/128/512 epochs | Extra training budget does not reliably remove the low-recall boundary; 128 epochs is a local peak, not a demonstrated optimum. |
| C5-NEGATIVE-COVERAGE-001 | continuous recall 0.272 -> 0.402 with mean-8 negatives; hardest-8 = 0.204 | Negative coverage materially affects learning, but the improvement remains below reliable selective-retrieval levels and costs 8x negative evaluations. |
| C5-NEGATIVE-SWEEP-001 | mean recall 0.272 -> 0.392 -> 0.402 -> 0.402 for 1/4/8/16 negatives | Mean-negative coverage has a strong early gain, then plateaus after eight; 16 negatives doubles training negative evaluations without improving mean Top-1. |
| C5-POSITIVE-VIEWS-001 | mean recall 0.402 -> 0.238 -> 0.286 for 1/2/4 views | Multi-view positive averaging reduces retrieval in every seed at 2 and 4 views; the current intervention is not a useful repair. |
| C5-LATENT-WIDTH-001 | mean recall 0.402 -> 0.460 -> 0.436 for latent width 8/16/32 | Width 16 gives a modest local improvement, but width 32 does not extend it; the registered capacity test enters saturation. |

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
10. learned-vs-index retrieval failure;
11. training-budget response.

The central remaining issue is not whether the encoder can learn any signal.
It clearly can. The question is whether the representation/objective can
produce sufficiently reliable state retrieval to make the selective boundary
safe.

## Current compute model

For the learned continuous state retriever:

- query encoder = 80 MACs/query;
- 64 cached state embeddings x 8-dimensional dot products = 512 MACs/query;
- continuous inference = **592 MACs/query**;
- state embedding build = **5,120 MACs/build**.

For the learned binary state index:

- query encoder = 80 MACs/query;
- Hamming radius-1 lookup = 9 probes;
- build arithmetic = 5,120 MACs/build.

For bounded C5 execution:

\`C_total ≈ C_address + C_candidate_retrieval + C_execute(R) + C_verify\`.

The current positive C5 result establishes only the synthetic execution and
exact/selective addressing controls. A learned state addressor has not yet
demonstrated the required high-recall boundary.

## Learned-state failure boundary

C5-LEARNED-STATE-DIAG-001 shows:

- no-learning continuous recall = 0.004;
- learned continuous recall = 0.214;
- learned binary recall = 0.170.

Thus the model learns real held-out semantic information. The binary
quantization gap is only 0.044 absolute on the same queries, below the
registered 0.20 diagnostic threshold.

C5-LEARNED-STATE-BUDGET-001 then varied training budget without changing the
representation, task stream, or evaluation split:

- 32 epochs: 0.214 mean recall;
- 128 epochs: 0.314;
- 512 epochs: 0.272.

The 32 -> 512 change is only +0.058. More updates improve some seeds and
degrade others, while target rank generally improves at the largest budget.
The result therefore does not support a simple “train longer” explanation for
the low Top-1 boundary.

## Current C5 status

Broad C5 remains **ungraded**.

Supported bounded components:

\`persistent state -> selective retention -> execute(R)\`.

Also supported:

- learned continuous semantic signal above a random control;
- direct identification of binary indexing as a non-dominant loss source;
- direct evidence that training budget alone is insufficient to eliminate the
  low-recall boundary in the current learner.

Not established:

- reliable learned selective state retrieval;
- learned sublinear semantic state addressing;
- learned sublinear candidate retrieval;
- capability/computation parity on realistic workloads;
- hardware latency/FLOP superiority;
- long-horizon persistent learning;
- natural-language semantic program selection.

## Next measurement boundary

The positive-view and latent-width interventions do not remove the retrieval
boundary. Positive-view averaging is harmful, while width scaling gives a small
local peak at 16 dimensions followed by a decline at 32.

The next intervention should therefore target the **similarity/objective
geometry** rather than more scalar capacity. A controlled experiment should
keep latent width 16, one positive view, and eight mean negatives fixed while
varying normalization and/or margin/temperature. Continuous target-state
retrieval remains the primary endpoint.

Binary indexing should remain excluded until continuous retrieval reaches a
materially stronger level.

## Provenance

The following evidence remains frozen and independently reproducible:

REP-001 through REP-009, SELECTIVE-001, C5-EXEC-001, C5-END-TO-END-001,
C5-LEARNED-STATE-001, C5-LEARNED-STATE-DIAG-001, C5-LEARNED-STATE-BUDGET-001,
C5-NEGATIVE-SWEEP-001, C5-POSITIVE-VIEWS-001, and C5-LATENT-WIDTH-001.
