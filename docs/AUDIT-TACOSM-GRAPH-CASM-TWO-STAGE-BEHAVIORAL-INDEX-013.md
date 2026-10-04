# AUDIT — TACOSM-GRAPH-CASM-TWO-STAGE-BEHAVIORAL-INDEX-013

**Status: PREREGISTERED / IMPLEMENTATION UNDER AUDIT

Confirmatory trigger: [run-graph-casm-two-stage-behavioral-index-013-full] / NO CONFIRMATORY RESULT.**

## Scientific purpose

G-CASM-012 changed the training target from exact target identity to
candidate/support behavioral compatibility, but its deployment scorer is still
a dense query-conditioned cross-encoder: candidate encodings are recomputed and
pair-scored across the population.

013 tests the next architectural decomposition rather than tuning 012:

```
static candidate library
        |
        +--> exact behavioral index ----> compatible set R
        |                                  |
        +--> cached candidate latents      +--> late-interaction reranker
                                                   |
                                                   +--> exact verifier/executor
```

The exact index is deliberately a **finite-domain ceiling**. It uses public
candidate truth-table metadata and observed support rows. It is not claimed as a
learned semantic mechanism.

## Why this is the right bottleneck isolation

The G-CASM workload has four binary inputs, hence sixteen possible input rows.
That makes exact behavior indexing unusually strong. This is scientifically
useful: before inventing another learned router, we can determine whether the
present bottleneck is inability to access behavior cheaply or inability to rank
the behavior-compatible candidates.

A positive 013 index result therefore does not establish generic sparse
retrieval. It establishes that an access path exists in the registered finite
domain.

## Late interaction

The learned candidate encoder from 012 is moved to an offline cache. Query-time
work consists of support-row encoding plus candidate-row pair interaction only
for candidates returned by the exact index.

The cached scorer is required to match the uncached 012 scorer numerically.
This makes the cache a deployment transformation rather than a new model.

## Static-library integrity

For each seed a single 512-program evaluation library is constructed after
training/evaluation structure and truth-table disjointness is asserted. The
behavioral index and candidate-latent cache are each built once before any
evaluation query. M=32/64/128/256/512 are prefix masks over that same library.

This is stronger than rebuilding a fresh index per query and models the intended
tool-library setting.

## Leakage boundary

The exact index receives:
- candidate public truth-table metadata;
- observed support rows.

The learned reranker receives:
- public candidate graph features;
- observed support rows.

Neither component receives:
- evaluator target identity;
- target-index metadata;
- complementary verifier rows.

Target identity appears only in post-routing metrics.

## Fixed/adaptive execution accounting

Fixed-budget evaluation checks every selected candidate against every verifier
row. Therefore, for every query and every registered budget,

```
fixed_rows_evaluated = budget * verifier_row_count
```

must hold exactly. Adaptive evaluation may short-circuit a candidate after a
mismatch and stops after the first verifier-passing candidate.

This separation prevents early stopping from making the fixed-budget
denominator incomparable to exhaustive execution.

## Cost accounting

No single artificial scalar combines bitmap operations, neural MACs, and
executor operations.

Reported separately:
- one-time candidate-cache MACs;
- one-time index membership operations;
- query-time row-encoding MACs;
- query-time pair-head MACs;
- posting lookups and bitmap-word operations;
- exact graph execution work;
- measured wall-clock, if added later.

The registered arithmetic is:

- candidate encoder: 20,608 MACs/candidate;
- row encoder: 2,368 MACs/support row;
- pair head: 8,256 MACs/candidate-row pair.

## Statistical integrity

Five seeds are the independent resampling unit. Task-level IID is not assumed.
Raw per-seed trial data is retained. The primary budget-selection rule is fixed
at support size 4 and the complete budget eligibility rule is applied before
seed aggregation.

## Non-claims

013 must not be interpreted as evidence of:
- asymptotic O(log M), O(sqrt M), or O(1) search;
- hardware speedup;
- general molecular, language, image, or audio semantics;
- universal superiority over other retrieval systems.

Those require new benchmark domains and separately preregistered experiments.

## Evidence inheritance

- 010: public executable graph representation and exact graph execution.
- 011: support-level identifiability / ambiguity ceiling.
- 012: compatibility-conditioned learned scorer.
- 013: access-cost decomposition using exact behavior retrieval + cached
  compatibility ranking.

If 013 fails, it does not erase 010–012. It narrows the failure boundary.


Implementation repair trigger: [run-graph-casm-two-stage-behavioral-index-013-full]
