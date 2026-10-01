# TACOSM-C5-NEGATIVE-SWEEP-001

Status: PREREGISTERED — result pending.

## Purpose

C5-NEGATIVE-COVERAGE-001 found that eight mean-gradient negatives improved
continuous target-state recall from 0.272 to 0.402, while hardest-negative
selection reduced recall to 0.204.

This sweep removes the hardest-negative intervention and varies only the size
of the mean-gradient negative pool.

The measured path remains:

`noisy query -> learned continuous semantic encoder -> exhaustive state score`.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor.
- 5 seeds (0..4).
- 512 epochs for every arm.
- 100 held-out noisy queries per seed/arm.
- dual linear 10 -> 8 encoder.
- learning rate 0.02; margin 0.25.
- training codes = CODEBOOK[16:64].
- evaluation codes = CODEBOOK[:16].
- A1 H-invariant query/state stream.
- continuous state scoring only.

Negative-pool sizes:

`1, 4, 8, 16`.

All arms use mean-gradient aggregation. Every arm makes exactly one optimizer
update per positive code per epoch, so every arm performs:

`512 * 48 = 24,576 optimizer updates`.

Only negative coverage changes.

## Primary endpoint

Target-state Top-1 recall under continuous exhaustive state scoring.

Secondary:

- mean target rank;
- fixed optimizer update count;
- total negative evaluations;
- inference arithmetic.

## Decision rule

The preregistered comparison focuses on the 8 -> 16 transition:

- mean-16 >= mean-8 + 0.10: broader coverage continues to materially help;
- mean-16 within [-0.05, +0.10] of mean-8: saturation/plateau after eight;
- mean-16 < mean-8 - 0.05: excessive coverage reverses the observed gain.

These are workload-specific diagnostics, not claims of an optimal negative pool
size.

## Training overhead

Negative evaluations per optimizer update are exactly the registered pool
size:

| Arm | Negatives/update | Total negative evaluations |
|---|---:|---:|
| mean-1 | 1 | 24,576 |
| mean-4 | 4 | 98,304 |
| mean-8 | 8 | 196,608 |
| mean-16 | 16 | 393,216 |

Thus widening negative coverage has a directly measured training cost.

## Inference cost

Inference is unchanged across arms:

- query encoding = 80 MACs/query;
- 64 cached state embeddings x 8 dimensions = 512 MACs/query;
- total continuous inference = 592 MACs/query;
- state embedding build = 5,120 MACs/build.

## Interpretation boundary

The sweep identifies the response curve of the current learning objective. It
does not establish learned selective retrieval, sublinear addressing, natural-
language semantic memory, or broad C5.

Binary indexing remains excluded. It should only be reintroduced after
continuous target-state retrieval is sufficiently reliable.

## Reproduction

Measurement command:

`python scripts/run_c5_negative_sweep_001.py`

Contract:

`contracts/TACOSM-C5-NEGATIVE-SWEEP-001.json`
