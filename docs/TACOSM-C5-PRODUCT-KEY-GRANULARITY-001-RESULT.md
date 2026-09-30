# TACOSM-C5-PRODUCT-KEY-GRANULARITY-001 RESULT

Status: MEASURED.

## Provenance

- GitHub Actions run: 36756106525
- Measurement commit: caa29eaade75cb5b6b26aafcaae0dcc1bfddb006
- Full artifact: TACOSM-C5-PRODUCT-KEY-GRANULARITY-001-full
- Artifact id: 11116179310
- Artifact digest: sha256:cb4c30a3a38609e5fa38d8c2eed9d1e75e653de612510da9169573e7c0746c0c
- Protocol: H=256, M={128,256,512}, K=32, factor beam=6, factor sizes={8,16,32}, five seeds.

## Pooled results

| factor size | M | exhaustive recall | selective recall | proposal retention | states scored/M | total MACs |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 0.362 | 0.354 | 0.966 | 0.5568 | 1428.256 |
| 8 | 256 | 0.186 | 0.184 | 0.982 | 0.5593 | 2578.688 |
| 8 | 512 | 0.054 | 0.054 | 0.980 | 0.5609 | 4882.944 |
| 16 | 128 | 0.362 | 0.318 | 0.790 | 0.1498 | 722.720 |
| 16 | 256 | 0.186 | 0.176 | 0.802 | 0.1514 | 1036.096 |
| 16 | 512 | 0.054 | 0.046 | 0.830 | 0.1513 | 1655.424 |
| 32 | 128 | 0.362 | 0.246 | 0.538 | 0.0490 | 772.448 |
| 32 | 256 | 0.186 | 0.124 | 0.526 | 0.0477 | 867.424 |
| 32 | 512 | 0.054 | 0.046 | 0.560 | 0.0472 | 1059.008 |

## Findings

### 1. The suspected scaling mechanism is supported

Factor size 8 keeps the reranking fraction essentially fixed at 55.7%–56.1% of M.

Increasing factor size changes that directly:

- 8 -> approximately 55.9% of M;
- 16 -> approximately 15.1% of M;
- 32 -> approximately 4.8% of M.

The scaling failure therefore was not caused by K=32 itself. It was caused by the fixed 64-cell product-key topology over-fetching increasingly populated cells as M grows.

This is the first controlled evidence that the product-key index can make the executed candidate set materially smaller than the persistent-state population by increasing address-space granularity.

### 2. The capability tradeoff is real

Factor size 8 is close to the exhaustive reference on this workload: pooled selective recall is 0.354/0.184/0.054 against exhaustive 0.362/0.186/0.054.

Factor size 16 reduces computation substantially, but selective recall falls to 0.318/0.176/0.046 and proposal retention is about 0.79/0.80/0.83.

Factor size 32 reduces the candidate fraction further, but proposal retention falls to about 0.54/0.53/0.56 and selective recall to 0.246/0.124/0.046.

This is not a free sparsification mechanism. Granularity controls a coverage/computation frontier.

### 3. Arithmetic cost also improves, but not monotonically with factor size

Mean total MACs at M=128/256/512 are:

- factor 8: 1428 / 2579 / 4883;
- factor 16: 723 / 1036 / 1655;
- factor 32: 772 / 867 / 1059.

Factor size 16 is lower-cost than factor size 32 at M=128, but factor size 32 becomes lower-cost at M=256 and 512. This reflects the tradeoff between factor-scoring/codebook cost and reranking cost.

These are arithmetic estimates, not hardware wall-clock measurements.

### 4. What this does and does not establish

**Established on the registered workload:** factor granularity is a causal control on candidate-set scaling in this product-key construction, and the fixed-size-8 occupancy mechanism explains the earlier approximately linear candidate growth.

**Not established:** universal sublinear retrieval, universal semantic addressing, or hardware speedup.

The C5 claim remains broader than this experiment. The evidence should therefore be promoted as a mechanism-level result, not as a completed C5 proof.

## Next research step

Do not immediately increase factor size further.

The useful next experiment is a **coverage-constrained granularity frontier**: vary factor size and factor beam jointly, with a pre-registered capability-retention floor, and measure the minimum states-scored/M satisfying that floor across M.

That separates two questions that are currently conflated:

1. whether the index can make computation sparse; and
2. whether it can do so while preserving enough task capability.

The resulting frontier is the appropriate next bridge toward the C5 capability-vs-computation claim.
