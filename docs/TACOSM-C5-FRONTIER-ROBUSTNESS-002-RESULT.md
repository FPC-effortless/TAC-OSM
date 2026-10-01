# TACOSM-C5-FRONTIER-ROBUSTNESS-002 RESULT

Status: MEASURED.

## Provenance

- GitHub Actions run: 36793799316
- measurement commit: 4b011641e6e5887d5190279150636dad0b327d9c
- full artifact: TACOSM-C5-FRONTIER-ROBUSTNESS-002-full
- artifact id: 11133237411
- artifact digest: sha256:c0cc6ae55bd5900a4a206e5c6ee409c4d66159ff418729f135fa3786a2d8c97e
- protocol: H=256, M={128,256,512}, K=32, 14 fixed product-key configurations, seeds 10-19, 100 evaluation steps per seed/configuration/M
- repository preconditions: passed
- test suite at measurement head: 999 passed

## Primary robustness result

The full 14-configuration surface was evaluated with exact pooled success-count aggregation across seeds 10-19.

The global pooled frontier is:

- factor_size = 16
- factor_beam = 6
- minimum capability retention across M = 0.8113207547
- mean states_scored_over_M = 0.1542864583
- mean total query arithmetic = 1155.1893 MACs

Therefore the registered robust sparse-frontier criterion passes on this seed set:
an eligible fixed configuration exists and its mean states_scored_over_M is below 0.20.

## Pooled frontier by M

| M | frontier factor size | frontier beam | capability retention | states scored/M | mean query MACs |
|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 6 | 0.870466 | 0.154414 | 732.24 |
| 256 | 8 | 2 | 0.843373 | 0.066199 | 559.152 |
| 512 | 32 | 4 | 0.886792 | 0.025729 | 882.768 |

These per-M minima differ from the global fixed frontier. The registered global decision is therefore based on one fixed configuration eligible at all three M levels, not on selecting a different configuration for each M.

## Reference configuration (16,6)

| M | Exhaustive successes | Selective successes | Capability retention | States scored/M |
|---:|---:|---:|---:|---:|
| 128 | 386 | 336 | 0.870466 | 0.154414 |
| 256 | 166 | 152 | 0.915663 | 0.152699 |
| 512 | 53 | 43 | 0.811321 | 0.155746 |

The M=512 regime remains the limiting pooled capability cell, but it remains above the preregistered 0.80 floor for the ten-seed robustness set.

## Seed-level limitation

The pooled result should not be read as every seed individually satisfying the 0.80 threshold at every M. The ten-seed cell artifact records substantial seed variation, particularly at M=512. For (16,6) at M=512, individual-seed retention ranges from 0.0 to 1.0 because exhaustive successes are sparse at that population size.

This is an aggregation-level limitation, not a missing data point. The preregistered robustness decision explicitly uses exact pooled counts across the ten independent seeds.

## Relation to the earlier five-seed replication

The earlier independent replication on seeds 5-9 failed to reproduce (16,6) at M=512 and selected (32,8) as its own global frontier. The present ten-seed robustness run does not erase that result. It shows that (16,6) is recoverable as the global fixed frontier when evaluated on a larger independent seed set with exact pooled count conservation.

This should be interpreted as evidence of configuration/seed sensitivity plus a robust pooled frontier on seeds 10-19, not as proof that (16,6) is invariant to every seed distribution.

## Computation

Across the ten-seed robustness run, (16,6) scores about 15.4% of the state population while preserving at least 81.1% of the exhaustive downstream capability at every registered M.

The compute advantage is therefore a bounded fraction of population on this workload, not an absolute O(1) candidate-count result. As M increases, the absolute reranking set still grows.

## Scientific status

Supported on this workload and seed set:

- a fixed product-key configuration can satisfy the registered 0.80 capability floor across M={128,256,512};
- that fixed frontier can operate at approximately 0.154 M states scored on average;
- exact pooled count-conserving analysis reproduces a sparse frontier on an independent ten-seed set.

Still unresolved:

- robustness across arbitrary seeds;
- absolute constant candidate computation;
- asymptotic sublinear state computation;
- semantic/general retrieval;
- hardware wall-clock speedup.

## Next research target

The important remaining question is no longer whether the two-factor index can produce a sparse pooled frontier. It can, on this workload.

The next experiment should measure whether the sparse frontier survives larger state populations and whether its fractional compute law improves or degrades. The clean extension is M={1024,2048} with the fixed (16,6) configuration plus exhaustive reference, using new independent seeds and the same state-distinct executor.

A separate seed/generalization axis should not be mixed into that population scaling run; otherwise we cannot distinguish a scaling failure from seed sensitivity.
