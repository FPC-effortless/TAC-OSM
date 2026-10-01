# TACOSM-C5-POSITIVE-VIEWS-001 RESULT

## Run provenance

- workflow: 36669933898
- clean run: #3
- clean measurement head: c62ddbd81e24a2dc7da7b1c2d2ca697496d0a320
- precondition gate: passed
- full test suite: 923 passed
- measurement step: passed
- artifact: TACOSM-C5-POSITIVE-VIEWS-001
- artifact id: 11077423458
- artifact digest: sha256:581f42e01b346613ea333f3e7da9e7fd40295847c78fc535991039dea2c5af58

The first two workflow runs were harness failures:

- run #1 failed the live test-count gate because it started before the 923-test synchronization;
- run #2 passed the count gate but exposed a structural-test defect in the eight-negative smoke test.

Run #3 is the first clean scientific measurement.

## Registered protocol

All arms held constant:

- M=64 persistent state items;
- H=64 task anchor and A1 H-invariant query/state stream;
- one-step causal write/read boundary;
- dual linear 10 -> 8 semantic encoder;
- learning rate 0.02;
- margin 0.25;
- 512 epochs;
- eight mean-gradient negatives per positive view;
- 24,576 optimizer updates;
- 48 training codes;
- 16 disjoint held-out evaluation codes;
- continuous exhaustive state scoring.

Only positive-view count changed: 1, 2, or 4 deterministic noisy views per update.

## Target-state Top-1 recall

| Seed | 1 view | 2 views | 4 views |
|---:|---:|---:|---:|
| 0 | 0.41 | 0.29 | 0.34 |
| 1 | 0.36 | 0.21 | 0.26 |
| 2 | 0.49 | 0.22 | 0.28 |
| 3 | 0.43 | 0.23 | 0.24 |
| 4 | 0.32 | 0.24 | 0.31 |
| **Mean** | **0.402** | **0.238** | **0.286** |

The observed effects are:

- 1 -> 2 views: -0.164 mean recall;
- 1 -> 4 views: -0.116 mean recall;
- 2 -> 4 views: +0.048 mean recall.

Every seed is lower at two views than one view, and every seed is lower at four views than one view.

## Target rank

| Seed | 1 view | 2 views | 4 views |
|---:|---:|---:|---:|
| 0 | 2.48 | 2.94 | 2.66 |
| 1 | 2.31 | 3.10 | 2.88 |
| 2 | 2.31 | 2.97 | 2.78 |
| 3 | 2.42 | 2.93 | 2.81 |
| 4 | 2.90 | 2.84 | 2.75 |
| **Mean** | **2.484** | **2.956** | **2.776** |

Rank also worsens on average when moving from one to multiple views.

## Preregistered decision handling

The contract's intended decision branches covered:

- a strong positive 1 -> 4 effect;
- a near-zero plateau;
- an intermediate positive effect.

The observed 1 -> 4 effect is negative (-0.116). That case was not explicitly represented in the preregistered branch text.

The runner therefore emitted the label partial_effect through its catch-all else clause. That label is **not a valid scientific decision for this run** and is quarantined.

The raw endpoint and its direction are still valid because they were fixed by the registered measurement itself.

Descriptive conclusion: richer deterministic positive-view averaging is **harmful in this configuration**. It should not be promoted as the next repair for the current learner.

This does not establish that all multi-view training is harmful. It isolates the specific intervention used here: averaging gradients from 2 or 4 deterministic noisy query views while holding eight mean negatives and the rest of the learner fixed.

## Training overhead

Optimizer updates remain fixed at 24,576 per arm.

| Arm | Positive views/update | Negative evaluations | Positive-view evaluations |
|---|---:|---:|---:|
| 1 view | 1 | 196,608 | 24,576 |
| 2 views | 2 | 393,216 | 49,152 |
| 4 views | 4 | 786,432 | 98,304 |

The multi-view arms therefore add training compute while reducing recall.

## Inference cost

Inference is unchanged:

- query encoding = 80 MACs/query;
- 64 cached state embeddings x 8 dimensions = 512 MACs/query;
- continuous inference = 592 MACs/query;
- state embedding build = 5,120 MACs/build.

## Scientific interpretation

The result argues against the hypothesis that the learner is mainly limited by seeing only one noisy positive realization per update.

With the current mean-8 negative objective:

- one view is the strongest of the registered view counts by the measured Top-1 endpoint;
- two views substantially degrade recall;
- four views recover slightly from the two-view degradation but remain below one-view recall.

The likely mechanism is not identified. Candidate explanations include gradient interference between transformed views or insufficient representational capacity for the averaged invariance objective, but these are hypotheses for future tests rather than findings of this experiment.

## Current C5 status

Broad C5 remains **ungraded**.

Supported bounded findings now include:

- persistent-state transport;
- learned semantic signal above random;
- negative-coverage sensitivity and plateau after eight negatives;
- the present multi-view positive averaging intervention does not improve retrieval;
- selective execution work depends on retained R in the synthetic executor.

Not established:

- reliable learned selective state retrieval;
- learned sublinear state addressing;
- learned candidate retrieval;
- realistic end-to-end capability/cost parity;
- hardware-level speedup;
- broad L4 C5.

## Next measurement boundary

The current results now point toward **encoder capacity / representation geometry** rather than more examples per update.

A clean next experiment is latent-width scaling at fixed mean-8 negatives and one positive view: compare 8, 16, and 32 latent dimensions while holding task generation, training budget, objective, and continuous evaluation fixed.

## Reproduction

Measurement command: python scripts/run_c5_positive_views_001.py

Contract: contracts/TACOSM-C5-POSITIVE-VIEWS-001.json