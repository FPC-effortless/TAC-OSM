# G-CASM-017B — Corrected exact evidence index

**Status: PREREGISTERED / IMPLEMENTATION UNDER AUDIT**

## Why 017B exists

017A is retained as an invalid predecessor. Its registered efficiency endpoint
mixed candidate scans, index posting writes, and bitmap-word operations as if
they
were interchangeable units. Its indexed selector also recomputed expected
environment cost by scanning all M candidates, leaving an unaccounted O(M)
term inside the purported indexed query.

No confirmatory scientific result is inferred from 017A.

## Scientific question

Can an exact prefix evidence histogram preserve the G-CASM-015 activation-trace
selector while removing candidate-record scanning from repeated one-step
selection queries?

The intended claim is deliberately narrower than C5. The index is a static
candidate-library optimization. It does not change the fact that constructing
the candidate evidence cache or the index is linear in the library size.

## Intervention

trace_exhaustive reproduces the G-CASM-015 exhaustive selector.

trace_indexed uses a prefix histogram of public candidate-predicted activation
traces plus cumulative candidate execution cost. For any requested prefix M,
the query reads the already-built histogram entries and cumulative cost instead
of iterating candidate records.

The evidence alphabet is fixed at the six-bit activation trace used by G-CASM-015,
so the number of histogram bins per action is bounded by 2^6.

## Exactness requirements

The indexed selector must reproduce exhaustive:

- selected action;
- information gain;
- expected remaining candidates;
- expected environment cost.

Target identity and realized target evidence are read only after the selector
has chosen an action. Exact downstream action equivalence is therefore the
capability-preservation bridge: both selectors feed the same downstream probe.

## Leakage boundaries

The index is built only from candidate-side evidence already present in the
public cache.

It cannot access:

- target identity or target index;
- realized target evidence before selection;
- verifier labels;
- test outcomes;
- any measured result used to tune the index or select M.

Train/evaluation structure and truth sets remain disjoint exactly as required by
G-CASM-015.

## Cost boundaries

The experiment records four separate quantities rather than pretending that all
operations share one hardware-independent unit:

1. exhaustive candidate-record reads per query;
2. indexed histogram-bin reads per query;
3. one-time index-build candidate-record reads per seed;
4. measured query latency on the same GitHub Actions runner.

Query latency is explicitly hardware-bound.

The index-build cost is never folded into the query-only endpoint. The workflow
also records the timing needed for lifecycle/amortization analysis, preventing a
one-time build from being silently omitted.

## Scientific interpretation

A positive result licenses only:

exact candidate-side indexing reduces repeated selector-query latency on the
pinned workload while preserving selector behavior.

It does not license:

- sublinear total history acquisition;
- hardware-independent asymptotic speedup;
- learned probing;
- semantic generalization;
- multimodal capability;
- general intelligence.

## Relationship to C5

C5 was a claim about total computation and was measured negative because
O(M) acquisition dominated the complete computation.

017B tests a separate, narrower question: whether one repeated candidate-side
acquisition/selection component can be amortized and made independent of M at
query time under a fixed finite evidence alphabet.

Even a positive 017B result would therefore not overturn the C5 result.


Confirmatory trigger: [run-graph-casm-evidence-index-017B-full]


Clean rerun authorization marker [run-graph-casm-evidence-index-017B-full]


Final smoke authorization after test repair: [run-graph-casm-evidence-index-017B-full]


Authorization retained in recent history: [run-graph-casm-evidence-index-017B-full]


Confirmatory authorization: [run-graph-casm-evidence-index-017B-full]


Final confirmatory authorization: [run-graph-casm-evidence-index-017B-full]
