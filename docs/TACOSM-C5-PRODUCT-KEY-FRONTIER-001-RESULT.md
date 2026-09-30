# TACOSM-C5-PRODUCT-KEY-FRONTIER-001 RESULT

Status: MEASURED.

## Provenance

- GitHub Actions run: 36793158179
- measurement commit: 1213671de27da123b3b7ae4bc0cbcc257b98bb4d
- full artifact: TACOSM-C5-PRODUCT-KEY-FRONTIER-001-full
- artifact id: 11132317893
- artifact digest: sha256:0ea1b37fad3bd8f04957661536a981501f55652fed9d805111a227f4d16d73f4
- protocol: H=256, M={128,256,512}, K=32, five seeds, 100 evaluation steps, factor sizes 16/32 with registered beam grids

## Pooled frontier

| factor size | beam | M | exhaustive recall | selective recall | recall retention | proposal retention | states scored/M | total MACs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 128 | 0.362 | 0.256 | 0.711 | 0.560 | 0.0728 | 565.1 |
| 16 | 4 | 256 | 0.186 | 0.148 | 0.795 | 0.554 | 0.0725 | 712.9 |
| 16 | 4 | 512 | 0.054 | 0.050 | 0.927 | 0.596 | 0.0723 | 1008.0 |
| 16 | 6 | 128 | 0.362 | 0.318 | 0.882 | 0.790 | 0.1498 | 722.7 |
| 16 | 6 | 256 | 0.186 | 0.176 | 0.943 | 0.802 | 0.1514 | 1036.1 |
| 16 | 6 | 512 | 0.054 | 0.046 | 0.683 | 0.830 | 0.1513 | 1655.4 |
| 16 | 8 | 128 | 0.362 | 0.330 | 0.910 | 0.910 | 0.2606 | 949.7 |
| 16 | 8 | 256 | 0.186 | 0.178 | 0.955 | 0.914 | 0.2574 | 1470.4 |
| 16 | 8 | 512 | 0.054 | 0.052 | 0.800 | 0.938 | 0.2590 | 2538.0 |
| 16 | 12 | 128 | 0.362 | 0.358 | 0.989 | 0.994 | 0.5589 | 1560.6 |
| 16 | 12 | 256 | 0.186 | 0.190 | 1.025 | 0.998 | 0.5575 | 2699.5 |
| 16 | 12 | 512 | 0.054 | 0.054 | 1.000 | 0.996 | 0.5619 | 5019.5 |
| 16 | 16 | 128 | 0.362 | 0.362 | 1.000 | 1.000 | 1.0000 | 2464.0 |
| 16 | 16 | 256 | 0.186 | 0.186 | 1.000 | 1.000 | 1.0000 | 4512.0 |
| 16 | 16 | 512 | 0.054 | 0.054 | 1.000 | 1.000 | 1.0000 | 8608.0 |
| 32 | 4 | 128 | 0.362 | 0.196 | 0.547 | 0.374 | 0.0264 | 726.0 |
| 32 | 4 | 256 | 0.186 | 0.116 | 0.607 | 0.350 | 0.0252 | 775.4 |
| 32 | 4 | 512 | 0.054 | 0.042 | 0.841 | 0.370 | 0.0249 | 875.6 |
| 32 | 6 | 128 | 0.362 | 0.246 | 0.679 | 0.538 | 0.0490 | 772.4 |
| 32 | 6 | 256 | 0.186 | 0.124 | 0.662 | 0.526 | 0.0477 | 867.4 |
| 32 | 6 | 512 | 0.054 | 0.046 | 0.898 | 0.560 | 0.0472 | 1059.0 |
| 32 | 8 | 128 | 0.362 | 0.270 | 0.744 | 0.678 | 0.0805 | 836.9 |
| 32 | 8 | 256 | 0.186 | 0.142 | 0.760 | 0.684 | 0.0783 | 992.5 |
| 32 | 8 | 512 | 0.054 | 0.046 | 0.854 | 0.722 | 0.0766 | 1299.8 |
| 32 | 12 | 128 | 0.362 | 0.334 | 0.925 | 0.884 | 0.1594 | 998.4 |
| 32 | 12 | 256 | 0.186 | 0.172 | 0.927 | 0.880 | 0.1578 | 1318.2 |
| 32 | 12 | 512 | 0.054 | 0.048 | 0.743 | 0.882 | 0.1572 | 1959.9 |

## Primary finding

The tested beam/granularity grid does not contain a configuration that is
simultaneously high-capability and strongly sparse.

Under the preregistered 0.90 target-recall-retention floor, only:

- factor16 / beam12;
- factor16 / beam16

satisfy the floor at all three M levels.

Their maximum reranking fractions are approximately 0.562M and 1.000M,
respectively.

All configurations with states scored/M <= 0.20 fail the 0.90 capability floor
at at least one M level.

Therefore the current two-factor product-key construction has a measured
compute-capability frontier, but the registered grid does not establish the
desired combination of high capability and low state-computation fraction.

## Mechanistic interpretation

The result is consistent with a beam/coverage tradeoff rather than a missing
final shortlist budget. Increasing beam admits more product cells and recovers
target coverage, but candidate-state computation rises with the selected cell
union.

At factor size 16, beam6 gives approximately 0.15M state scoring but falls to
0.683 recall retention at M=512. Beam12 recovers the recall floor but moves
back to approximately 0.56M state scoring.

At factor size 32, beam12 remains sparse at approximately 0.16M states scored,
but recall retention at M=512 falls to 0.743.

This rules out simply increasing the beam as a solution to the C5 bottleneck.

## Arithmetic

Factor16/beam6 uses roughly 0.72x–1.66k MACs/query across M=128–512.
Factor32/beam12 uses roughly 1.00x–1.96k MACs/query.

These are arithmetic estimates from the registered cost model, not hardware
timings.

## Boundary of the claim

Supported on this workload:

- beam size controls a reproducible capability/computation tradeoff;
- high product-key granularity can reduce the reranking fraction substantially;
- recovering high teacher-recall retention at the tested scale presently requires
  admitting enough cells to give back much of that sparsity.

Not supported:

- a configuration simultaneously achieving >=0.90 recall retention and
  <=0.20 states_scored/M across M={128,256,512};
- universal sublinear retrieval;
- hardware speedup;
- general semantic memory capability.

## Next experiment

Do not continue a one-dimensional beam sweep.

The remaining bottleneck is factorization itself: two-factor addressing forces a
fixed rectangular Cartesian cell decomposition. The next controlled experiment
should test a **three-factor product-key index** at matched per-factor codebook
size and small beam, measuring whether additional factorization provides finer
address resolution without requiring a large beam.

The comparison should hold the teacher, training-only codebook regime, M, H,
K, query stream and executor fixed. The new intervention is factorization depth.
