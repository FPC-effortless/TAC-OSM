# TACOSM-C5-LEARNED-STATE-BUDGET-001 RESULT

## Run provenance

- workflow: \`36668690764\`
- clean run: #3
- clean measurement head: \`dc6fe0cff89dab27dbd49510728a9600624cfbc5\`
- precondition gate: passed
- full test suite: **901 passed**
- measurement step: passed
- artifact: \`TACOSM-C5-LEARNED-STATE-BUDGET-001\`
- artifact id: \`11077138334\`
- artifact digest: \`sha256:332f79bd9b48a0ffa7fcbcd236b4aecf25c8e62be2fef269171f71a1d202c196\`

The first two workflow runs were harness failures only:

- run #1 started before the final test-count synchronization and failed the
  live-count gate;
- run #2 passed the 901-test gate but still required a removed K dimension.

Neither produced a scientific result. The third run is the clean measurement.

## Registered comparison

The experiment held constant:

- M=64 persistent state items;
- H=64 task anchor;
- 48 training codes and 16 disjoint evaluation codes;
- one-bit query noise;
- the A1 H-invariant task stream;
- dual linear 10 -> 8 encoder;
- learning rate 0.02 and margin 0.25;
- continuous exhaustive state scoring.

Only the training budget changed.

Training budgets:

- 32 epochs = 1,536 training pairs;
- 128 epochs = 6,144 training pairs;
- 512 epochs = 24,576 training pairs.

## Target-state Top-1 recall

| Seed | No-learning | 32 epochs | 128 epochs | 512 epochs |
|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.20 | 0.24 | 0.16 |
| 1 | 0.00 | 0.25 | 0.37 | 0.30 |
| 2 | 0.01 | 0.21 | 0.36 | 0.38 |
| 3 | 0.01 | 0.23 | 0.35 | 0.25 |
| 4 | 0.00 | 0.18 | 0.25 | 0.27 |
| **Mean** | **0.004** | **0.214** | **0.314** | **0.272** |

The learned models remain clearly above the random continuous control.

The highest mean recall occurs at 128 epochs (0.314). Increasing the budget
from 128 to 512 reduces mean recall to 0.272.

The registered 32 -> 512 improvement is only:

\`0.272 - 0.214 = +0.058\`

which is below the preregistered 0.20 materiality threshold.

## Target rank

Mean target ranks by budget:

| Seed | 32 epochs | 128 epochs | 512 epochs |
|---:|---:|---:|---:|
| 0 | 3.84 | 3.63 | 3.73 |
| 1 | 3.08 | 2.49 | 2.81 |
| 2 | 4.76 | 2.46 | 2.31 |
| 3 | 6.41 | 5.63 | 2.99 |
| 4 | 5.69 | 3.18 | 2.71 |

The 512-epoch run improves mean rank relative to 32 epochs even though Top-1
recall does not improve proportionally. The model is therefore often placing
the correct state near the top without consistently making it the top-ranked
item.

## Decision

The registered diagnostic enters the second branch:

- 32 -> 512 recall gain = **+0.058**, below the +0.20 materiality threshold;
- 512-epoch mean recall = **0.272**, below 0.50.

Therefore, additional training budget does **not** remove the current
low-recall boundary under this architecture/objective.

This is evidence against treating the current failure primarily as a lack of
training steps.

It is not evidence that a different objective, representation, negative
sampling strategy, or architecture cannot improve the retriever.

## Important non-monotonicity

The result is not a monotonic learning curve:

\`32: 0.214 -> 128: 0.314 -> 512: 0.272\`.

The 128-epoch improvement is real across the five-seed mean, but the 512-epoch
extension does not preserve it. This is compatible with optimization
instability, objective saturation, or over-specialization, but the present
experiment does not distinguish those mechanisms.

No post-hoc budget selection is made for the C5 claim.

## Compute

Continuous scoring remains fixed across all budgets:

- query encoding = **80 MACs/query**;
- 64 cached state embeddings x 8 dimensions = **512 MACs/query**;
- total continuous query arithmetic = **592 MACs/query**;
- state embedding build = **5,120 MACs/build**.

Training cost changes with budget and is represented by the registered pair
count rather than silently mixed into inference cost.

## Interpretation

Combined with C5-LEARNED-STATE-DIAG-001, the current evidence now separates
three facts:

1. the encoder learns a genuine held-out semantic signal above random;
2. binary quantization is not the dominant loss boundary under the diagnostic;
3. increasing training budget from 32 to 512 epochs does not reliably lift
   retrieval into a high-recall regime.

The remaining bottleneck should therefore be investigated at the
representation/objective level.

The most direct next experiment is negative-coverage/objective design: replace
the current single deterministic negative with controlled hard-negative
coverage while holding training budget, model width, task stream and
evaluation split fixed.

## C5 status

Broad C5 remains **ungraded**.

This experiment does not establish learned selective retrieval, sublinear
learned addressing, learned candidate retrieval, hardware speedup, realistic
workload parity, or the broad L4 C5 claim.

## Reproduction

Measurement command:

\`python scripts/run_c5_learned_state_budget_001.py\`

Contract:

\`contracts/TACOSM-C5-LEARNED-STATE-BUDGET-001.json\`
