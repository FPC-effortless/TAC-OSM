# TACOSM-C5-COVERAGE-FRONTIER-004 RESULT

Status: MEASURED — VALID.

## Provenance

- GitHub Actions run: 36790982734
- Measurement head: 50d56eef583e9764bdb38423de2611f831949bf4
- Full artifact: TACOSM-C5-COVERAGE-FRONTIER-004-full
- Artifact id: 11131529242
- Artifact digest: sha256:7f1b8c20aae757c9daf6a552fe559b0984ccb28ff92bf74ffd32f04bc6950af2
- 999 tests passed in preconditions.
- Smoke passed.
- Full measurement passed.

## Validity

The state-distinct executor audit passed for all 210 seed × M × configuration cells.

Pooled aggregation passed exact count conservation for all 42 M × configuration cells. Each pooled cell contains 500 evaluations: five seeds × 100 evaluation steps.

The stored `capability_retention` is computed from summed selective and exhaustive success counts. The same scalar is used for the eligibility flag, and the runner asserts that those quantities agree.

No truncation, per-cell ratio averaging, or non-discriminating executor is used in the accepted result.

## Primary result

A single product-key configuration remained above the preregistered 0.80 capability-retention floor at every M level:

- factor size = 16
- factor beam = 6
- mean `states_scored_over_M` = 0.1508177
- minimum capability retention across M = 0.8518519
- mean per-query total arithmetic = 1,138.08 MACs

Thus, on this registered synthetic workload, a fixed product-key configuration preserved at least 85.2% of the exhaustive reference capability while scoring about 15.1% of the state population on average.

## Per-M capability/computation frontier

| M | factor size | factor beam | capability retention | states scored / M | total MACs | exhaustive E2E | selective E2E | selective target recall | proposal retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 6 | 0.8785 | 0.1498 | 722.72 | 0.362 | 0.318 | 0.318 | 0.790 |
| 256 | 8 | 2 | 0.8710 | 0.0668 | 561.568 | 0.186 | 0.162 | 0.162 | 0.510 |
| 512 | 32 | 2 | 0.8148 | 0.0086 | 742.368 | 0.054 | 0.044 | 0.044 | 0.160 |

The per-M frontier changes configuration with M. It therefore demonstrates a capability/computation surface, not a single universal parameter setting.

## Fixed-configuration frontier

The fixed configuration `(factor_size=16, factor_beam=6)` is eligible at all three M levels and has the lowest mean `states_scored_over_M` among fixed configurations satisfying the 0.80 floor at every M.

Its per-M values are approximately 14.98%, 15.14%, and 15.13% of M for M=128, 256, and 512 respectively.

That is an approximately constant fraction of the population. It is therefore a constant-factor sparse-compute result, not evidence of sublinear asymptotic scaling.

## Arithmetic effect

For the fixed `(16,6)` configuration, mean per-query total MACs are:

| M | exhaustive query MACs | selective total MACs | arithmetic reduction |
|---:|---:|---:|---:|
| 128 | 2,208 | 722.72 | 67.27% |
| 256 | 4,256 | 1,036.096 | 75.66% |
| 512 | 8,352 | 1,655.424 | 80.18% |

The savings are from reducing the state reranking workload after factor routing; factor scoring remains part of the selective computation and is reported in total arithmetic.

## Capability limitation

The exhaustive reference itself weakens as M grows: exhaustive end-to-end success is 0.362 at M=128, 0.186 at M=256, and 0.054 at M=512.

Therefore the 0.80 retention floor measures preservation of the reference condition, not high absolute capability. At M=512, for example, the frontier configuration achieves 0.044 absolute end-to-end success while retaining 81.5% of the exhaustive reference.

## Scientific status

This result advances C5 from untested to **PARTIALLY SUPPORTED, bounded**.

It supports the narrower statement that the registered product-key sparse retrieval mechanism can preserve a fixed capability fraction while reducing the state reranking fraction substantially on this synthetic state-addressing workload.

It does not establish:

- sublinear asymptotic scaling in M;
- universal semantic retrieval;
- a hardware wall-clock speedup;
- a universal product-key optimum;
- or the complete L4 claim that all relevant execution cost scales with the routed relevant subset.

The execution-side `C_execute = O(R)` observation remains separately bounded by `TACOSM-C5-EXEC-001`. Together, the evidence now covers both a retrieval-side sparse frontier and a controlled execution-side relevant-subset relation, but not a generalized end-to-end theorem.

## Next technical step

The next C5 experiment should test whether the approximately 15% fixed fraction can be reduced while maintaining the same 0.80 relative-capability floor using a registered multi-stage or hierarchical router, rather than increasing factor granularity alone.
