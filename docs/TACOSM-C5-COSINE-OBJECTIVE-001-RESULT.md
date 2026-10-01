# TACOSM-C5-COSINE-OBJECTIVE-001 RESULT

## Run provenance

- workflow: 36671027453
- clean run: #3
- clean measurement head: 0f7b7c0d040b7baf45ebe271205c4fc035c4006b
- precondition gate: passed
- full test suite: 938 passed
- measurement step: passed
- artifact: TACOSM-C5-COSINE-OBJECTIVE-001
- artifact id: 11077997461
- artifact digest: sha256:03d59e9e91c5f0ee34c0010c855a60d261b06dca3bd52a030275f476aa75a99b

The first two runs were harness failures only: the first started before the
938-test synchronization, and the second exposed a Python-list normalization
bug in a numerical-gradient test. Neither produced a scientific result.

## Registered comparison

Both arms used:

- M=64 persistent state items;
- H=64 task anchor and A1 H-invariant query/state stream;
- latent width 16;
- 512 epochs;
- 24,576 optimizer updates;
- one positive view;
- eight mean-gradient negatives;
- 48 disjoint training codes;
- 16 held-out evaluation codes;
- learning rate 0.02;
- margin 0.25.

The only training-objective difference was raw dot-product margin versus
cosine-normalized margin. Both arms were evaluated with the same cosine
similarity metric.

## Target-state Top-1 recall

| Seed | Raw objective | Cosine objective |
|---:|---:|---:|
| 0 | 0.70 | 0.70 |
| 1 | 0.70 | 0.70 |
| 2 | 0.70 | 0.70 |
| 3 | 0.70 | 0.70 |
| 4 | 0.70 | 0.70 |
| **Mean** | **0.700** | **0.700** |

## Target rank

| Seed | Raw objective | Cosine objective |
|---:|---:|---:|
| 0 | 1.30 | 1.30 |
| 1 | 1.30 | 1.30 |
| 2 | 1.30 | 1.30 |
| 3 | 1.30 | 1.30 |
| 4 | 1.30 | 1.30 |
| **Mean** | **1.30** | **1.30** |

The seed-level equality is exact in the recorded artifact.

## Preregistered decision

The cosine-minus-raw recall difference is:

0.700 - 0.700 = 0.000.

This is within the preregistered ±0.05 no-material-change interval.

Therefore the result enters the **no-material-change** branch.

The current evidence does not support cosine-normalized training as a material
improvement over the existing raw objective when both are evaluated under the
same cosine metric on this workload.

This is not evidence that score normalization is irrelevant generally.

## Important diagnostic observation

The absolute recall of 0.700 is higher than the approximately 0.402 recall in
the earlier latent-width experiment, which evaluated the raw-trained model with
raw dot-product scoring.

That difference cannot be attributed to the training objective comparison in
this experiment because the inference metric was intentionally fixed to cosine
for both arms.

It creates a new diagnostic boundary: inference-time similarity geometry itself
may be responsible for part of the apparent retrieval improvement.

The next test should therefore hold the raw training objective fixed and compare
raw versus cosine inference on the same trained model.

## Cost

At latent width 16:

- query projection = 160 MACs/query;
- 64-state score products = 1,024 MACs/query;
- projection + score arithmetic = 1,184 MACs/query;
- state embedding build = 10,240 MACs/build.

Cosine inference additionally uses vector normalization. The current ledger
reports 130 normalization operations per query rather than converting them into
MACs.

Training negative evaluations remain 196,608 per arm.

## Scientific interpretation

This experiment rules out a simple interpretation that the raw objective is
materially inferior to cosine-normalized training under a fixed cosine
retrieval metric.

The more important observation is that the evaluation metric differs from the
earlier raw-dot retrieval experiments. The next experiment must isolate that
effect before further changing the training objective.

## Current C5 status

Broad C5 remains **ungraded**.

Supported bounded findings now include:

- persistent-state transport;
- genuine learned semantic signal above random;
- strong sensitivity to negative coverage, with a plateau after eight negatives;
- multi-view positive averaging is harmful in this configuration;
- latent-width scaling through 32 dimensions does not provide continuing gain;
- raw versus cosine training is indistinguishable under fixed cosine evaluation.

Not established:

- reliable learned selective state retrieval;
- learned sublinear state addressing;
- learned candidate retrieval;
- realistic end-to-end capability/cost parity;
- hardware-level speedup;
- broad L4 C5.

## Reproduction

Measurement command: python scripts/run_c5_cosine_objective_001.py

Contract: contracts/TACOSM-C5-COSINE-OBJECTIVE-001.json