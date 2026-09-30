# TACOSM-C5-LATENT-WIDTH-001 RESULT

## Run provenance

- workflow: 36670555698
- clean run: #4
- clean measurement head: 291f24fe42a995669ef6a55c5517b800ba558535
- precondition gate: passed
- full test suite: 930 passed
- measurement step: passed
- artifact: TACOSM-C5-LATENT-WIDTH-001
- artifact id: 11077204371
- artifact digest: sha256:1231f28a5702a8b91c540f02fda45d60b3a5cd68e87013f6aba6842cce9b8428

Runs #1-3 were harness failures involving the latent-width structural tests. No scientific width result was taken from them. Run #4 is the clean measurement.

## Registered protocol

All arms held constant:

- M=64 persistent state items;
- H=64 task anchor and A1 H-invariant query/state stream;
- one-step causal write/read boundary;
- 512 epochs;
- one deterministic positive view per update;
- eight mean-gradient negatives per positive;
- 24,576 optimizer updates;
- 48 training codes;
- 16 disjoint held-out evaluation codes;
- learning rate 0.02;
- margin 0.25;
- continuous exhaustive state scoring.

Only latent width changed: 8, 16, or 32.

## Target-state Top-1 recall

| Seed | Width 8 | Width 16 | Width 32 |
|---:|---:|---:|---:|
| 0 | 0.41 | 0.45 | 0.42 |
| 1 | 0.36 | 0.52 | 0.44 |
| 2 | 0.49 | 0.45 | 0.43 |
| 3 | 0.43 | 0.45 | 0.43 |
| 4 | 0.32 | 0.43 | 0.46 |
| **Mean** | **0.402** | **0.460** | **0.436** |

Mean width response:

- 8 -> 16: +0.058;
- 16 -> 32: -0.024;
- 8 -> 32: +0.034.

Width 16 is the highest mean-recall arm, but the wider 32-dimensional encoder does not preserve or extend that gain.

## Target rank

| Seed | Width 8 | Width 16 | Width 32 |
|---:|---:|---:|---:|
| 0 | 2.48 | 2.19 | 2.32 |
| 1 | 2.31 | 2.00 | 2.09 |
| 2 | 2.31 | 1.98 | 2.23 |
| 3 | 2.42 | 2.08 | 2.11 |
| 4 | 2.90 | 2.29 | 2.19 |

Width 16 improves mean target rank versus width 8; width 32 is similar but does not convert that into a larger Top-1 gain.

## Preregistered decision

The registered primary comparison was width 32 versus width 8.

Observed:

0.436 - 0.402 = +0.034.

This is below the preregistered +0.05 saturation boundary and far below the +0.15 strong-effect criterion.

Therefore the experiment enters the **saturation** branch.

The result does not support treating wider latent dimension as a material solution to the current retrieval boundary.

It also does not establish that width 16 is globally optimal; the observed width-16 peak is a workload-specific local maximum.

## Inference cost

| Width | Query encode | 64-state scoring | Total query MACs | State build MACs |
|---:|---:|---:|---:|---:|
| 8 | 80 | 512 | 592 | 5,120 |
| 16 | 160 | 1,024 | 1,184 | 10,240 |
| 32 | 320 | 2,048 | 2,368 | 20,480 |

Thus the width-16 mean-recall improvement over width 8 is +0.058 while doubling continuous inference arithmetic, and width 32 increases arithmetic fourfold relative to width 8 without a corresponding Top-1 gain.

These MAC counts are arithmetic proxies, not hardware latency measurements.

## Scientific interpretation

The current learner has now been tested along several orthogonal axes:

1. More training budget: no reliable monotonic gain.
2. More negative coverage: strong early gain, then plateau after eight.
3. Multiple noisy positive views: harmful under gradient averaging.
4. Greater latent width: small width-16 improvement, no continuing gain at 32.

Taken together, the evidence suggests the remaining retrieval boundary is not removed by simply increasing optimization steps, negative count, positive-view count, or latent width within this model family.

The next intervention should therefore target the **objective geometry itself** rather than scalar capacity. A natural next test is a temperature/margin or normalized similarity objective that reduces score-scale instability while preserving the same encoder width, mean-8 negatives, and one positive view.

## C5 status

Broad C5 remains **ungraded**.

Supported bounded findings now include:

- persistent-state transport;
- genuine learned semantic signal above random;
- selective execution work depending on retained R in the synthetic executor;
- sensitivity to negative coverage, with plateau after eight negatives;
- negative effect of multi-view gradient averaging in this configuration;
- saturation of latent-width scaling through 32 dimensions.

Not established:

- reliable learned selective state retrieval;
- learned sublinear state addressing;
- learned candidate retrieval;
- realistic end-to-end capability/cost parity;
- hardware-level speedup;
- broad L4 C5.

## Reproduction

Measurement command: python scripts/run_c5_latent_width_001.py

Contract: contracts/TACOSM-C5-LATENT-WIDTH-001.json