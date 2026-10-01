# TACOSM-C5-PRODUCT-KEY-001 RESULT

## Run provenance

- workflow run: 36699321108
- head: 69302a4d88e910bb6872e5f1f425de241cb9244e
- artifact: TACOSM-C5-PRODUCT-KEY-001-full
- artifact id: 11089197442
- artifact digest: sha256:f0ed3bea00e2ef71b17234db16244f1d475e7567079a2af30b036c185c07c0d1
- preconditions: passed
- repository test suite: 994 passed
- protocol: M=64, H={64,128,256}, seeds={0,1,2,3,4}, K={4,8,16}

## Pooled result

| K | Factor beam | Proposal target retention | Selective target recall | Exhaustive target recall | Selective end-to-end success | Exhaustive end-to-end success | Mean shortlist | Mean states scored | Mean query MACs | Arithmetic reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.2093 | 0.1860 | 0.7000 | 0.4740 | 0.8333 | 1.102 | 1.102 | 305.6 | 74.19% |
| 8 | 2 | 0.4680 | 0.3733 | 0.7000 | 0.6813 | 0.8333 | 4.181 | 4.190 | 355.0 | 70.01% |
| 16 | 4 | 0.8713 | 0.6240 | 0.7000 | 0.7980 | 0.8333 | 14.903 | 16.115 | 545.8 | 53.90% |

The K=16 target-recall gap from exhaustive cosine is 0.0760. The end-to-end
gap is 0.0353. Thus the mechanism approaches the downstream capability
reference more closely than the sparse prototype funnel, but target-state
recall remains below the 0.05 parity margin used in the earlier prototype
experiment.

## H decomposition

| H | K=4 target recall | K=8 target recall | K=16 target recall | K=16 proposal retention | K=16 end-to-end success |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.18 | 0.34 | 0.58 | 0.80 | 0.58 |
| 128 | 0.18 | 0.39 | 0.60 | 0.78 | 0.72 |
| 256 | 0.13 | 0.40 | 0.64 | 0.84 | 1.00 |

The retrieval endpoint is comparatively stable across H, while the synthetic
executor again exhibits output aliasing at H=256; target-state recall remains
the primary retrieval measure.

## Build and runtime arithmetic

The exhaustive reference costs 1,184 MACs/query.

Product-key query arithmetic includes the teacher query projection, factor-code
scoring, and teacher cosine scoring over every over-fetched candidate actually
scored before truncation.

At K=16, the mean runtime cost is 545.8 MACs/query with 16.1 states scored on
average and 14.9 states retained. This is a 53.90% arithmetic reduction
relative to exhaustive scoring.

The product-key build uses training-only codebook fitting and full-state cell
assignment. With 48 training states, factor size 8, latent width 16, and eight
k-means iterations, the registered build ledger is 49,152 MACs for codebook
fitting plus 8,192 MACs for runtime state assignment, for 57,344 MACs total.

No hardware wall-clock speedup is established.

## Comparison with the preceding sparse funnel

At K=16 the distilled prototype funnel reported 64.1% proposal retention,
45.4% actual target recall, 72.1% end-to-end success, and 832 MACs/query.
The product-key mechanism reported 87.1%, 62.4%, 79.8%, and 545.8 MACs/query
under the same M/H/K workload and the same fixed teacher/executor.

This is evidence that factorized product-key geometry changes the proposal
coverage regime substantially. It does not, by itself, establish capability
parity or general learned semantic addressing.

## Scientific status

The product-key mechanism is the strongest sparse-addressing signal measured
in this branch so far, but it remains below the previously used 0.05 target
recall parity margin at K=16.

The result isolates a narrower problem: the factorized address geometry captures
more of the teacher's target state with substantially less arithmetic, but the
remaining misses are still large enough to prevent a capability-parity claim.

## Next diagnostic

The next controlled intervention is K=32 with a factor beam large enough to
stress coverage while remaining below the 64-state exhaustive reference.
The teacher representation, codebooks, state layout, and downstream executor
remain fixed.

## Reproduction

Full measurement: python scripts/run_c5_product_key_001.py

Workflow: https://github.com/FPC-effortless/TAC-OSM/actions/runs/36699321108
