# TACOSM-C5-COSINE-SELECTIVE-001 RESULT

## Run provenance

- workflow: 36671969372
- clean run: #7
- clean measurement head: dcab70a4f9c291a7c22bf7e67d6a01f38161a9e1
- precondition gate: passed
- full test suite: 955 passed
- measurement step: passed
- artifact: TACOSM-C5-COSINE-SELECTIVE-001
- artifact id: 11078073948
- artifact digest: sha256:b765c9ff79b2504ce58696aa9e2fb1fa2282677560f9601a0de42f38367c912f

The earlier runs on this branch were harness-only failures. They are retained as provenance and are not used as scientific results.

## Registered protocol

All arms held constant:

- persistent state population M=64;
- H=64, 128, 256;
- A1 H-invariant query/state generation;
- one-step causal write/read boundary;
- dual linear 10 -> 16 encoder;
- raw dot-product training objective;
- 512 epochs;
- one deterministic positive view;
- eight mean-gradient negatives;
- 48 training codes and 16 held-out evaluation codes;
- exact downstream candidate index;
- fixed 12-work-unit candidate executor.

The selective arm used an 8-bit learned sign index, Hamming radius 2, K=4, and cosine re-ranking over the retained state shortlist.

## State retrieval

Exhaustive cosine target-state recall is 0.700 at every H level.

The selective arm retains the actual target state at 0.428 pooled:

| Seed | Exhaustive target recall | Proposal target retention | Selective target recall |
|---:|---:|---:|---:|
| 0 | 0.70 | 0.55 | 0.50 |
| 1 | 0.70 | 0.50 | 0.40 |
| 2 | 0.70 | 0.46 | 0.44 |
| 3 | 0.70 | 0.53 | 0.42 |
| 4 | 0.70 | 0.48 | 0.38 |
| **Mean** | **0.700** | **0.504** | **0.428** |

The proposal itself contains the true target only 0.504 of the time. Cosine reranking does not repair that loss; it reduces actual target recall further to 0.428.

The selective target-retention gap versus exhaustive cosine is:

0.700 - 0.428 = 0.272.

That exceeds the registered 0.05 tolerance by a wide margin.

## End-to-end capability

Actual end-to-end success is compared with hidden environment truth, not the exhaustive model output:

| H | Exhaustive success | Selective success | Gap |
|---:|---:|---:|---:|
| 64 | 0.700 | 0.474 | 0.226 |
| 128 | 0.800 | 0.648 | 0.152 |
| 256 | 1.000 | 1.000 | 0.000 |
| **Mean** | **0.833** | **0.707** | **0.126** |

The H=256 equality is an output-aliasing effect: state-target recall remains 0.428 while the downstream synthetic task still produces the correct aggregate output. This is why state retrieval remains the load-bearing endpoint.

## Preregistered decision

The registered rule required selective target-state recall and selective end-to-end success to remain within 0.05 of their exhaustive-cosine references while the shortlist stayed at K=4.

Observed pooled values:

- target-state recall gap = 0.272;
- end-to-end success gap = 0.126;
- mean shortlist size = 4.0.

Therefore the preregistered capability rule **fails**.

The registered selective boundary does not preserve the exhaustive learned-cosine reference sufficiently on this workload.

This does not establish that learned selective retrieval is impossible. It establishes that this specific learned binary proposal + K=4 cosine-reranking mechanism is insufficient.

## Compute

State retrieval arithmetic:

- exhaustive state scoring = 1,024 MACs/query after query projection;
- selective cosine re-ranking = 64 MACs/query at K=4;
- query projection = 160 MACs/query for both;
- selective projection + reranking = 224 MACs/query;
- arithmetic reduction in state scoring = 93.75%;
- total projection+score arithmetic reduction = 1 - 224/1184 = 81.1%;
- binary proposal = 37 radius-2 bucket probes;
- state-index build = 10,240 MACs/build;
- mean shortlist size = 4.0.

The selective path therefore reduces continuous state-score arithmetic substantially, but the retained subset is not sufficiently correct.

These are arithmetic operation counts, not hardware-latency measurements.

## Scientific interpretation

This experiment is the strongest current test of whether the learned state retriever can actually cross into a bounded selective-compute regime.

The answer under the registered mechanism is negative.

The learned continuous model can reach 0.700 target-state recall at M=64 with cosine similarity. However, the learned binary proposal retains the true target only about half the time, and cosine re-ranking inside K=4 cannot recover the missing target.

Thus the failure is now localized more sharply than before:

1. The learned representation has useful continuous semantic signal.
2. Cosine inference materially improves that representation.
3. The current learned binary proposal cannot preserve enough of that signal to support a K=4 selective boundary.
4. The resulting 81.1% reduction in projection+score arithmetic is therefore not a valid end-to-end compute advantage because capability is not preserved.

## C5 status

Broad C5 remains **ungraded**.

Supported bounded findings:

- persistent-state transport;
- learned semantic signal above random;
- inference-time cosine geometry materially improves retrieval;
- negative coverage materially affects learning and plateaus after eight negatives;
- multi-view positive averaging is harmful in this configuration;
- latent width through 32 does not produce continuing gains;
- selective execution work can depend on R in the synthetic executor.

Not established:

- reliable learned selective state retrieval;
- learned sublinear state addressing with capability parity;
- learned candidate retrieval;
- realistic end-to-end compute advantage;
- hardware speedup;
- broad L4 C5.

## Next measurement boundary

The current failure is now specifically at the **discrete proposal boundary**: continuous cosine retrieval is 0.700, but the learned binary proposal only retains the true target 0.504 of the time.

The next controlled test should therefore avoid sign-code bucketing and evaluate a learned continuous coarse-to-fine proposal, such as a learned centroid/codebook routing layer, while keeping the cosine representation fixed.

The proposal stage must be trained without evaluation addresses or gold labels, and its shortlist cost must be explicit.

## Reproduction

Measurement command: python scripts/run_c5_cosine_selective_001.py

Contract: contracts/TACOSM-C5-COSINE-SELECTIVE-001.json