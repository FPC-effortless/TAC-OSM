# TACOSM-C5-PRODUCT-KEY-SCALING-001 RESULT

## Run provenance

- workflow run: 36751997528
- measurement head: e6da5b3085460edb73da2ca2b01c68a6531b8779
- artifact: TACOSM-C5-PRODUCT-KEY-SCALING-001-full
- artifact id: 11115885250
- artifact digest: sha256:8728b98abe41dec02a48249e3c0e2e1abdfee6e0c21f7a4a1653c1b69208cb8d
- repository preconditions: passed
- protocol: H=256, M={64,128,256,512}, K=32, factor beam=6, five seeds, 100 evaluation steps per seed/M cell
- test suite at measurement head: 999 passed

## Pooled result

| M | K/M | Proposal target retention | Actual target recall | Exhaustive target recall | Selective end-to-end success | Exhaustive end-to-end success | Mean states scored | States scored / M | Mean query MACs | Arithmetic reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.5000 | 0.9960 | 0.7000 | 0.7000 | 1.0000 | 1.0000 | 35.37 | 0.5527 | 853.98 | 27.87% |
| 128 | 0.2500 | 0.9660 | 0.3540 | 0.3620 | 0.5800 | 0.5880 | 71.27 | 0.5568 | 1428.26 | 35.31% |
| 256 | 0.1250 | 0.9820 | 0.1860 | 0.1860 | 0.3160 | 0.3180 | 143.17 | 0.5593 | 2578.69 | 39.41% |
| 512 | 0.0625 | 0.9800 | 0.0540 | 0.0540 | 0.1060 | 0.1060 | 287.18 | 0.5609 | 4882.94 | 41.54% |

## Primary finding

The fixed-K scaling hypothesis is not supported by the registered measurement.

Target-state proposal retention stays high across the population sweep, but the
number of states subjected to teacher cosine scoring increases approximately
linearly with the persistent-state population:

- M=64 -> 35.37 scored states;
- M=128 -> 71.27;
- M=256 -> 143.17;
- M=512 -> 287.18.

The scored fraction remains almost constant at 55.3%–56.1% of M.

This occurs despite K remaining fixed at 32. Consequently the present product-key
construction does not produce a bounded state-computation regime as M grows.

## Capability result

From M=128 onward, the fixed teacher itself has weak target-state recall on this
scaled workload (0.362, 0.186, 0.054 at M=128,256,512). The selective arm tracks
the exhaustive reference closely:

- target recall differences: -0.008, 0.000, 0.000 at M=128,256,512;
- end-to-end differences: -0.008, -0.002, 0.000.

Thus the scaling failure is not primarily caused by selective routing losing
additional targets. The addressing layer preserves nearly the same retrieval
behavior as the fixed exhaustive reference while paying for an approximately
constant fraction of the entire state pool.

## Arithmetic

Exhaustive query arithmetic grows from 1,184 MACs at M=64 to 8,352 at M=512.

Selective arithmetic grows from 853.98 to 4,882.94 MACs/query. Arithmetic
reduction improves from 27.87% to 41.54%, but this is not evidence of bounded
state computation because the selective cost also grows substantially with M.

No hardware wall-clock speedup is established.

## Mechanistic interpretation

The current implementation has a fixed factor size of 8 in each of two dimensions,
giving 64 possible product cells. With factor beam 6, up to 36 cells are
considered. As M grows while the cell topology remains fixed, average cell
occupancy grows, so the over-fetched candidate union grows with M.

The measured scored-state fractions around 0.56 are consistent with this fixed
cell-granularity geometry rather than with a fixed-K sparse addressing law.

This localizes the next problem: **cell granularity must scale, or the routing
mechanism needs a second sparse addressing stage.**

## Scientific status

Supported on this workload:

- the product-key mechanism can retain near-all target states while using a
  bounded final shortlist K=32;
- selective retrieval can closely track the exhaustive cosine reference.

Not supported:

- fixed-K bounded state computation as M grows;
- a sublinear state-compute law for the current fixed factor-size construction;
- hardware speedup;
- general semantic retrieval.

## Next diagnostic

Hold K=32 and H=256 fixed at M={128,256,512}, and vary factor size across
training-fitted codebooks (8,16,32). Measure whether increasing cell granularity
reduces states scored/M without materially degrading target recall. Keep the
factor beam fixed so the intervention isolates cell topology.

Reproduction:

python scripts/run_c5_product_key_scaling_001.py
