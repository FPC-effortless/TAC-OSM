# TACOSM-C5-PRODUCT-KEY-K32-001 RESULT

## Run provenance

- workflow run: 36699821631
- head: 77883f874aa872553411eda306a7969a62f9e086
- artifact: TACOSM-C5-PRODUCT-KEY-K32-001-full
- artifact id: 11089124114
- artifact digest: sha256:87058740c4bb26a278f07613ae5547b456383d41883f125fb032435e4716d849
- preconditions: passed
- repository test suite: 996 passed

## Registered result

| H | Proposal retention | Actual target recall | Exhaustive target recall | Selective end-to-end success | Exhaustive end-to-end success | Mean shortlist | Mean states scored | Mean query MACs | Arithmetic reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.000 | 0.700 | 0.700 | 0.700 | 0.700 | 31.93 | 35.82 | 861.1 | 27.27% |
| 128 | 0.970 | 0.670 | 0.700 | 0.790 | 0.800 | 31.92 | 36.00 | 864.0 | 27.03% |
| 256 | 0.990 | 0.690 | 0.700 | 1.000 | 1.000 | 31.92 | 35.88 | 862.1 | 27.19% |

Pooled across the 15 H/seed cells:

- proposal target retention: 0.9847;
- actual target recall: 0.6920;
- exhaustive target recall: 0.7000;
- actual end-to-end success: 0.8280;
- exhaustive end-to-end success: 0.8333;
- mean shortlist size: 31.87;
- mean states scored before truncation: 35.77;
- mean query arithmetic: 860.28 MACs;
- arithmetic reduction versus 1,184-MAC exhaustive cosine: 27.34%.

## Interpretation

The K=32 intervention closes the target-recall gap to 0.008 absolute and the
end-to-end gap to 0.0053 relative to the exhaustive reference on this workload.
Under the previously used 0.05 parity margin, both downstream endpoints are
within the margin.

However, this is not a strong sparse-addressing result by itself. K=32 retains
roughly half of the M=64 state population and the factorized proposal scores
roughly 35.8 states on average before truncation. The remaining arithmetic
reduction is only 27.34% because much of the sparse-routing budget is spent on
the over-fetched candidate set.

Therefore the result supports a bounded product-key mechanism with near-reference
capability at K=32 on this synthetic workload, but it does not establish a
meaningful sublinear state-compute law.

## Relation to K=16

The K=16 result had pooled target recall 0.6240, proposal retention 0.8713,
end-to-end success 0.7980, and mean query arithmetic 545.85 MACs.
K=32 raises these to 0.6920, 0.9847, 0.8280, and 860.28 MACs respectively.

This separates the tradeoff cleanly: additional candidate budget recovers most
of the remaining target states, but it costs substantially more arithmetic.

## Scientific status

The product-key mechanism now has bounded support for near-reference retrieval
capability at K=32 under M=64. The claim remains conditional on this workload
and on the fixed representation/teacher.

Not established:

- state-compute scaling as M grows;
- fixed-K performance as state population increases;
- hardware speedup;
- general semantic retrieval.

## Next diagnostic

Scale the persistent-state population while holding the K budget and teacher
representation fixed. The important quantity is not absolute K=32 performance
at M=64, but whether the same bounded candidate budget retains capability as M
increases.

## Reproduction

Full measurement: python scripts/run_c5_product_key_k32_001.py

Workflow: https://github.com/FPC-effortless/TAC-OSM/actions/runs/36699821631
