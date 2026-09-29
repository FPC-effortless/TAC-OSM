# TACOSM-SELECTIVE-001 — Selective computation boundary

**Status:** pre-registered; no confirmatory result yet.

## Question

Can the hardened runtime preserve capability while changing actual router work
from the full persistent candidate universe H to a bounded retained subset R?

This is the first direct measurement of the repaired call graph:

    persistent candidate universe
        -> address
        -> retain R
        -> route over R
        -> execute
        -> observe
        -> verify

## Why the candidate universe is static

The v0 retrieval implementation rebuilt an O(H) index whenever the synthetic
candidate list changed. Measuring that path as 'cheap retrieval' would merely
move the H-scaling work from the scorer into the index.

This experiment therefore creates one candidate universe per (seed, H) and
reuses it for exactly 100 queries. The one-time index build cost is reported
separately and amortized over those declared queries.

## Registered design

* H = 8, 64, 256.
* K = 2, 4 for the indexed arm.
* Seeds = 0, 1, 2, 3, 4.
* 100 queries per (seed, H, K) cell.
* 'exhaustive' scores all H candidates.
* 'indexed' builds a content-addressed equality index once and passes only the
  retained bucket to the router.
* The candidate universe is deliberately multi-valid: the two marked
  positions define four equality buckets, and any candidate in the queried
  bucket is acceptable. This makes R a meaningful retained set rather than a
  hidden single-gold shortcut.
* The router is a deterministic public equality control. No hidden task truth
  enters the routing or addressing path.
* Repair and learning writes are disabled.

## Endpoints

Primary:

    success_rate

Secondary:

    router_candidates_per_step
    index_build_candidates
    address_query_positions
    amortized_candidate_work
    executor_invocations
    wall_clock_seconds

'amortized_candidate_work' is:

    (one_time_index_build + query-time address work + router candidate work)
    / declared query count

The one-time build term is not discarded. The experiment explicitly states
the number of queries over which it is amortized.

## Decision rule

An indexed cell that preserves the exhaustive capability while keeping
router input bounded and building the index once establishes that the
selective runtime boundary is actually present for this exact synthetic
control.

It does not establish:

* semantic retrieval quality;
* learned or neural addressing;
* that a PLM's total compute scales with R;
* the L4 C5 economic claim.

Those require later experiments with the actual relevance representation and
structural computation substrate.

## Leakage and truth separation

The index receives only the public query and any persistent representation
explicitly returned by the state layer. Hidden reference_bits remain runner
truth and are used only to grade/execute the task.

This separation is enforced in HardenedLoop by distinct
visible_reference and truth_reference values.

## Reproduction

Contract:

    contracts/TACOSM-SELECTIVE-001.json

Instrument:

    python scripts/measure_selective_scaling.py

Use '--smoke' only for implementation checks; smoke output is not a result.