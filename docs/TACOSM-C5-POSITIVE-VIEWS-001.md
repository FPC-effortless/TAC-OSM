# TACOSM-C5-POSITIVE-VIEWS-001

Status: PREREGISTERED — result pending.

## Purpose

The negative-pool sweep found that mean-gradient coverage improves retrieval
strongly from one to four negatives, weakly from four to eight, and then
plateaus at sixteen.

This experiment keeps the eight-negative mean-gradient objective fixed and
changes the positive-query view coverage. The test asks whether the learner is
limited by seeing only one noisy realization of each positive per update.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor.
- 5 seeds (0..4).
- 512 epochs for every arm.
- 100 held-out noisy queries per seed/arm.
- dual linear 10 -> 8 encoder.
- learning rate 0.02; margin 0.25.
- eight mean-gradient negatives per positive view.
- training codes = CODEBOOK[16:64].
- evaluation codes = CODEBOOK[:16].
- A1 H-invariant query/state stream.
- continuous state scoring only.

Positive-view counts:

1, 2, 4.

Every arm makes exactly 24,576 optimizer updates. The views are averaged
within the same optimizer update, so view count changes training example
coverage and cost, not update count.

## Positive-view construction

Each view is a deterministic noisy version of the same positive semantic code.
The view index changes the corruption pattern. No stochastic stream is used, so
the result remains reproducible.

The same view-generation rule is used for every arm; an arm with fewer views is
a strict prefix of the registered view set.

## Primary endpoint

Target-state Top-1 recall under continuous exhaustive state scoring.

Secondary:

- mean target rank;
- optimizer update count;
- negative evaluations;
- positive-view evaluations;
- continuous inference arithmetic.

## Decision rule

- views_4 >= views_1 + 0.15 and views_4 >= 0.50: material positive-view effect;
- views_4 within 0.05 of views_1: saturation;
- otherwise: partial evidence that positive-view coverage contributes.

These rules are workload-specific diagnostics, not optimality claims.

## Training overhead

Negative evaluations are fixed at:

24,576 updates x 8 negatives x positive_views.

Thus:

| Arm | Positive views/update | Negative evaluations | Positive-view evaluations |
|---|---:|---:|---:|
| views-1 | 1 | 196,608 | 24,576 |
| views-2 | 2 | 393,216 | 49,152 |
| views-4 | 4 | 786,432 | 98,304 |

The multi-view arms therefore pay directly measured training-time overhead.

## Inference cost

Inference remains:

- query encoding = 80 MACs/query;
- 64 cached state embeddings x 8 dimensions = 512 MACs/query;
- total continuous inference = 592 MACs/query;
- state embedding build = 5,120 MACs/build.

## Interpretation boundary

This experiment tests training-view coverage in the current learned semantic
retriever. It does not establish semantic memory, learned selective routing,
sublinear addressing, or broad C5.

The binary index remains excluded.

## Reproduction

Measurement command:

python scripts/run_c5_positive_views_001.py

Contract:

contracts/TACOSM-C5-POSITIVE-VIEWS-001.json
