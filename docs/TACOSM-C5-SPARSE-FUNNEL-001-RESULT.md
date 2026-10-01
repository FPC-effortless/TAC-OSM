# TACOSM-C5-SPARSE-FUNNEL-001 RESULT

## Run provenance

- workflow run: 36698358062
- head: 5aa8e856d19a60ddae8d20d3c704633891040e36
- artifact: TACOSM-C5-SPARSE-FUNNEL-001-full
- artifact id: 11089146615
- artifact digest: sha256:441829017da558e1c8147745bb387897b433d193feb08a786bac6fa965589f2c
- preconditions: passed
- repository test suite: 991 passed
- registered protocol: M=64, H={64,128,256}, seeds={0,1,2,3,4}, K={4,8,16}

## Distillation

The proposal distillation used 48 training-state items and 48 deterministic noisy
training queries. The full-candidate teacher representation remained fixed.

- initial KL: 2.805035776537395
- final KL: 0.44752718094967375
- optimizer updates: 12,288
- teacher candidate-score arithmetic: 9,437,184 MACs
- student prototype-score arithmetic: 3,145,728 MACs

The reduction in KL demonstrates that the student fitted the registered
prototype-distribution objective, but objective fit did not translate into
capability parity.

## Pooled result

| K | Prototype beam | Proposal target retention | Selective target recall | Exhaustive target recall | Selective end-to-end success | Exhaustive end-to-end success | Mean query MACs | Arithmetic reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.2587 | 0.1713 | 0.7000 | 0.5773 | 0.8333 | 640 | 45.95% |
| 8 | 2 | 0.4307 | 0.2827 | 0.7000 | 0.6227 | 0.8333 | 704 | 40.54% |
| 16 | 4 | 0.6413 | 0.4540 | 0.7000 | 0.7213 | 0.8333 | 832 | 29.73% |

The K=16 target-recall gap is 0.2460 and the end-to-end gap is 0.1120.
Both exceed the registered 0.05 capability margin, so the registered
capability boundary is not preserved at the largest tested budget.

## H decomposition

| H | K=4 target recall | K=8 target recall | K=16 target recall | K=16 proposal retention |
|---:|---:|---:|---:|---:|
| 64 | 0.192 | 0.286 | 0.430 | 0.622 |
| 128 | 0.176 | 0.274 | 0.428 | 0.606 |
| 256 | 0.148 | 0.276 | 0.416 | 0.596 |

The retrieval degradation persists across H. The H=256 end-to-end output can
mask retrieval errors because the synthetic executor has output aliasing;
actual target-state recall is therefore the primary retrieval endpoint.

## Compute

The exhaustive reference costs 1,184 MACs/query.

The selective funnel costs 640, 704, and 832 MACs/query at K=4,8,16.
Arithmetic reduction is therefore 45.95%, 40.54%, and 29.73%, respectively.
These are operation-count statements only; no hardware wall-clock speedup is
established.

## Interpretation

The intervention separates two effects.

First, increasing the proposal budget materially increases coarse target
retention: 0.2587 -> 0.4307 -> 0.6413. That makes proposal budget a real
limitation in this regime.

Second, the final target recall remains substantially below the exhaustive
cosine reference even at K=16. The teacher reranker is not the main source of
the loss: once the target is retained, the observed conditional recovery is
close to the teacher's full-pool behavior. The dominant failure remains proposal
coverage.

Relative to the earlier one-prototype continuous experiment, the new funnel
improves bounded proposal retention at the larger K budgets, but it does not
cross the registered capability boundary.

## Scientific status

This is negative evidence for the current distilled-prototype proposal as a
capability-preserving sparse state-addressing mechanism on this workload.

It does not establish that product-key addressing, different prototype
geometry, larger K, or a different student architecture cannot work.

## Next diagnostic

The next controlled intervention is factorized product-key addressing with
training-only codebook fitting and the same fixed teacher/reranker. This isolates
prototype-cell geometry from the student's query-to-prototype distillation
while keeping the downstream execution path unchanged.

## Reproduction

Full measurement: python scripts/run_c5_sparse_funnel_001.py

Workflow: https://github.com/FPC-effortless/TAC-OSM/actions/runs/36698358062
