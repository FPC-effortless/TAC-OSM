# TACOSM-C5-NEGATIVE-SWEEP-001 RESULT

## Run provenance

- workflow: 36669460013
- clean run: #4
- clean measurement head: a2f6672620cc681f06888d5e889972902ccb704f
- precondition gate: passed
- full test suite: 916 passed
- measurement step: passed
- artifact: TACOSM-C5-NEGATIVE-SWEEP-001
- artifact id: 11077239762
- artifact digest: sha256:13c59663df6e192d41c85524c803e75786e8500b6a9a8cdfcb86db9a7d70f3fd

The first three runs did not produce a scientific result:

- run #1 failed the stale live-count gate before measurement;
- run #2 passed the count gate but exposed a structural-test mismatch;
- run #3 repeated that structural-test failure.

Those runs are retained as harness provenance. Run #4 is the clean sweep.

## Registered protocol

All arms held constant:

- M=64 persistent state items;
- H=64 task anchor;
- A1 H-invariant query/state stream;
- one-step causal write/read boundary;
- dual linear 10 -> 8 semantic encoder;
- learning rate 0.02;
- margin 0.25;
- 512 epochs;
- 24,576 optimizer updates;
- 48 training codes;
- 16 disjoint held-out evaluation codes;
- continuous exhaustive state scoring.

Only the mean-gradient negative-pool size changed: 1, 4, 8, 16.

## Target-state Top-1 recall

| Seed | Mean-1 | Mean-4 | Mean-8 | Mean-16 |
|---:|---:|---:|---:|---:|
| 0 | 0.16 | 0.39 | 0.41 | 0.41 |
| 1 | 0.30 | 0.38 | 0.36 | 0.39 |
| 2 | 0.38 | 0.43 | 0.49 | 0.46 |
| 3 | 0.25 | 0.39 | 0.43 | 0.38 |
| 4 | 0.27 | 0.37 | 0.32 | 0.37 |
| **Mean** | **0.272** | **0.392** | **0.402** | **0.402** |

The response is therefore:

1 -> 4: +0.120

4 -> 8: +0.010

8 -> 16: +0.000.

## Target rank

| Seed | Mean-1 | Mean-4 | Mean-8 | Mean-16 |
|---:|---:|---:|---:|---:|
| 0 | 3.73 | 2.45 | 2.48 | 2.43 |
| 1 | 2.81 | 2.25 | 2.31 | 2.35 |
| 2 | 2.31 | 2.27 | 2.31 | 2.75 |
| 3 | 2.99 | 3.36 | 2.42 | 3.09 |
| 4 | 2.71 | 2.46 | 2.90 | 2.51 |
| **Mean** | **2.91** | **2.558** | **2.484** | **2.626** |

The best mean Top-1 recall occurs at both 8 and 16 negatives, but the rank metric does not improve monotonically: mean-16 rank is worse than mean-8.

## Preregistered decision

The 8 -> 16 recall change is:

0.402 - 0.402 = 0.000.

The registered interval for a plateau was:

-0.05 <= mean_16 - mean_8 < +0.10.

Therefore the result enters the **plateau-after-eight** branch.

The evidence does not support claiming that increasing mean-gradient negative coverage beyond eight continues to improve retrieval on this workload.

It also does not establish that eight negatives is globally optimal. It only shows that the registered 16-negative extension produced no mean Top-1 gain.

## Training overhead

All arms use 24,576 optimizer updates.

| Arm | Negatives/update | Total negative evaluations |
|---|---:|---:|
| Mean-1 | 1 | 24,576 |
| Mean-4 | 4 | 98,304 |
| Mean-8 | 8 | 196,608 |
| Mean-16 | 16 | 393,216 |

Moving from 8 to 16 negatives therefore doubles negative-scoring work without improving mean Top-1 recall.

## Inference cost

Inference is identical across all arms:

- query encoding = 80 MACs/query;
- 64 cached state embeddings x 8 dimensions = 512 MACs/query;
- continuous inference = 592 MACs/query;
- state embedding build = 5,120 MACs/build.

Training overhead is separate from inference arithmetic.

## Scientific interpretation

The objective diagnosis is now sharper:

1. Single-negative training is materially weaker than broader coverage: 0.272 mean recall.
2. Four negatives gives a substantial increase: 0.392.
3. Eight negatives produces a small additional gain: 0.402.
4. Sixteen negatives produces no additional mean Top-1 gain: 0.402.

Thus negative coverage is a genuine contributor, but the current learner reaches a plateau around the 8-negative regime.

The target rank remains around 2.5, meaning the representation often places the correct item near the top while failing to make it reliably top-1.

This points away from another brute-force increase in negative count. The next learning intervention should alter the representation/objective geometry rather than simply increasing the number of negatives.

A controlled next step is to preserve mean-8 coverage and add a richer positive view—multiple independent noise realizations of each positive query per update—while keeping the total optimizer-update budget explicit.

## C5 status

Broad C5 remains **ungraded**.

Supported bounded findings now include:

- persistent-state transport;
- learned semantic signal above random;
- sensitivity to negative coverage;
- saturation of the current mean-negative objective after approximately eight negatives;
- selective execution work depending on retained R in the synthetic executor.

Not established:

- reliable learned selective state retrieval;
- learned sublinear state addressing;
- learned candidate retrieval;
- realistic end-to-end capability/cost parity;
- hardware-level speedup;
- broad L4 C5.

## Reproduction

Measurement command: python scripts/run_c5_negative_sweep_001.py

Contract: contracts/TACOSM-C5-NEGATIVE-SWEEP-001.json