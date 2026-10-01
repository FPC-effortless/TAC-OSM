# TACOSM-C5-NEGATIVE-COVERAGE-001 RESULT

## Run provenance

- workflow: `36668977830`
- clean run: #2
- clean measurement head: `68a05acb923bda4c2c34d882a535383448602786`
- precondition gate: passed
- full test suite: **909 passed**
- measurement step: passed

## Registered comparison

All arms used:

- M=64 persistent state items;
- H=64 task anchor;
- A1 H-invariant query/state stream;
- 48 training codes and 16 disjoint evaluation codes;
- the same dual linear 10 -> 8 encoder;
- learning rate 0.02 and margin 0.25;
- **512 epochs**;
- 24,576 optimizer updates per arm;
- continuous exhaustive state scoring.

Only the negative objective changed.

## Target-state Top-1 recall

| Seed | Single negative | Mean 8 negatives | Hardest 8 negatives |
|---:|---:|---:|---:|
| 0 | 0.16 | 0.41 | 0.21 |
| 1 | 0.30 | 0.36 | 0.29 |
| 2 | 0.38 | 0.49 | 0.16 |
| 3 | 0.25 | 0.43 | 0.15 |
| 4 | 0.27 | 0.32 | 0.21 |
| **Mean** | **0.272** | **0.402** | **0.204** |

Mean-8 negatives improve mean recall by:

`0.402 - 0.272 = +0.130`.

Hardest-8 negatives change mean recall by:

`0.204 - 0.272 = -0.068`.

## Target rank

| Seed | Single negative | Mean 8 negatives | Hardest 8 negatives |
|---:|---:|---:|---:|
| 0 | 3.73 | 2.48 | 2.96 |
| 1 | 2.81 | 2.31 | 3.12 |
| 2 | 2.31 | 2.31 | 3.35 |
| 3 | 2.99 | 2.42 | 3.61 |
| 4 | 2.71 | 2.90 | 3.12 |
| **Mean** | **2.91** | **2.484** | **3.232** |

Mean-8 training improves average target rank from 2.91 to 2.484.
Hardest-8 worsens it to 3.232.

## Decision

The registered result enters the **partial-evidence** branch.

- mean-8 improvement = **+0.130**, above the +0.10 threshold;
- neither alternative reaches the stronger requirement of +0.20 and mean
  recall >=0.50;
- hardest-8 is negative.

Therefore negative coverage/objective design is a **material contributing
factor** under this synthetic continuous-retrieval workload, but it is not
sufficient to produce a reliably selective retriever.

The result does not establish that mean-8 negatives are an optimal objective,
and no post-hoc optimization claim is made.

## Training cost

All arms use exactly 24,576 optimizer updates.

Negative evaluations differ:

| Arm | Negatives/update | Total negative evaluations |
|---|---:|---:|
| Single negative | 1 | 24,576 |
| Mean 8 negatives | 8 | 196,608 |
| Hardest 8 negatives | 8 | 196,608 |

The multi-negative arms therefore spend 8x the negative-scoring evaluations
while leaving the optimizer-update count unchanged.

This overhead is part of the result and must not be silently omitted from any
future end-to-end cost comparison.

## Inference cost

The inference path is unchanged:

- query encoding = **80 MACs/query**;
- 64 cached state embeddings x 8 dimensions = **512 MACs/query**;
- continuous total = **592 MACs/query**;
- state embedding build = **5,120 MACs/build**.

## Interpretation

The evidence now separates the current representation/objective failure into
two effects:

1. More training steps alone do not reliably solve the problem:
   32 -> 512 epochs changed mean recall only from 0.214 to 0.272.
2. More negative coverage does matter:
   single-negative -> mean-8 changed recall from 0.272 to 0.402.

However, hardest-negative selection did not help in this configuration. That
means “hard-negative mining” cannot be treated as synonymous with
“better negative coverage”; the aggregation and sampling geometry matters.

The mean-8 result is therefore useful evidence for the learning objective, not
a completed retrieval solution.

## Current C5 status

Broad C5 remains **ungraded**.

Supported bounded findings now include:

- persistent state transport;
- learned semantic signal above random;
- selective execution work scaling with R in the synthetic executor;
- negative-coverage sensitivity of the semantic state learner.

Not established:

- reliable learned selective state retrieval;
- learned sublinear state addressing;
- learned candidate retrieval;
- realistic end-to-end capability/cost parity;
- hardware-level speedup;
- broad L4 C5.

## Next diagnostic

The next controlled step should increase semantic negative coverage without
introducing hardest-negative selection. A registered sweep over negative pool
sizes such as 1, 4, 8, 16 at the same 512-epoch budget would test whether the
mean-8 improvement continues, saturates, or reverses before the binary index is
reintroduced.
