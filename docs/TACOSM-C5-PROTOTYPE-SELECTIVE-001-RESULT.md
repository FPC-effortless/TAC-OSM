# TACOSM-C5-PROTOTYPE-SELECTIVE-001 RESULT

## Run provenance

- workflow: 36672513478
- clean run: #2
- clean measurement head: 9f0b9e0c97f8d6516fe402e4041fe3ba4e612c8b
- precondition gate: passed
- full test suite: 963 passed
- measurement step: passed
- artifact: TACOSM-C5-PROTOTYPE-SELECTIVE-001
- artifact id: 11078481728
- artifact digest: sha256:7ba9ebf5a2696d747f303d5b4c792752933d6f6fbf28e4d80f5e87497065b0cc

## Registered protocol

All arms held constant:

- persistent state population M=64;
- H=64, 128, 256;
- A1 H-invariant query/state stream;
- one-step causal write/read boundary;
- width-16 dual linear semantic encoder;
- raw dot-product training objective;
- 512 epochs;
- one positive view;
- eight mean-gradient negatives;
- 48 training codes and 16 held-out evaluation codes;
- exact downstream candidate index;
- fixed 12-work-unit executor.

The selective mechanism learned 16 continuous prototypes using only training-code
embeddings, assigned evaluation states to capacity-four buckets, selected one
prototype per query, and cosine-reranked the resulting four states.

## State retrieval

The exhaustive cosine reference has actual target-state recall 0.700 in all H cells.

The prototype-selective arm has actual target-state recall 0.204 pooled and a
prototype-bucket target retention of 0.314.

| Seed | Exhaustive target recall | Prototype proposal retention | Prototype selective target recall |
|---:|---:|---:|---:|
| pooled | 0.700 | 0.314 | 0.204 |

The coarse prototype router therefore loses the true target before the final
cosine re-ranking stage. The four-state cosine reranker cannot recover a state
that was not assigned to the selected prototype bucket.

## End-to-end capability

| H | Exhaustive success | Prototype selective success | Gap |
|---:|---:|---:|---:|
| 64 | 0.800 | 0.572 | 0.228 |
| 128 | 1.000 | 1.000 | 0.000 |
| 256 | 0.700 | 0.206 | 0.494 |
| **Mean** | **0.833** | **0.593** | **0.241** |

The H=128 equality is another output-aliasing case; target-state recall remains
0.204, so downstream equality cannot substitute for retrieval correctness.

## Preregistered decision

The registered rule required selective actual target recall and actual end-to-end
success to remain within 0.05 of the exhaustive-cosine references while retaining
at most four states.

Observed pooled gaps:

- target-state recall gap = 0.700 - 0.204 = 0.496;
- end-to-end success gap = 0.833 - 0.593 = 0.241;
- shortlist size = 4.0.

Therefore the capability rule **fails**.

This is negative evidence for one-prototype continuous coarse routing with
capacity-four buckets.

## Compute

Exhaustive state retrieval:

- query projection = 160 MACs;
- 64-state cosine scoring = 1,024 MACs;
- total = 1,184 MACs/query.

Prototype-selective retrieval:

- query projection = 160 MACs;
- 16-prototype scoring = 256 MACs;
- four-state cosine reranking = 64 MACs;
- total = 480 MACs/query;
- arithmetic reduction = 1 - 480/1184 = 59.5%.

Prototype/index construction:

- 48 training embeddings = 7,680 MACs;
- eight k-means iterations = 98,304 MACs;
- 64 evaluation-state embeddings = 10,240 MACs;
- state-to-prototype assignment = 16,384 MACs;
- total build arithmetic = 132,608 MACs.

The arithmetic reduction is therefore real as an operation-count statement,
but it is not an end-to-end compute advantage because capability is not preserved.

## Scientific interpretation

Replacing sign-code bucketing with learned continuous prototypes did not solve
the selective boundary.

The failure is specifically at coarse routing: the selected prototype contains
the true target only 31.4% of the time. This is worse than the earlier learned
binary proposal's 50.4% target retention on the same general regime.

The result also reinforces the importance of reporting direct target-state
retention rather than relying on downstream task success, because H-dependent
output aliasing can hide severe retrieval errors.

## C5 status

Broad C5 remains **ungraded**.

Supported bounded findings:

- persistent-state transport;
- genuine learned semantic signal above random;
- cosine inference materially improves continuous retrieval;
- negative coverage materially affects learning and plateaus after eight;
- multi-view positive averaging is harmful in this configuration;
- latent-width scaling through 32 does not provide continuing gain;
- binary sign selective retrieval fails capability parity;
- one-prototype continuous selective retrieval also fails capability parity;
- selective execution work depends on R in the synthetic executor.

Not established:

- reliable learned selective state retrieval;
- learned sublinear state addressing with capability parity;
- learned candidate retrieval;
- realistic end-to-end compute advantage;
- hardware speedup;
- broad L4 C5.

## Next diagnostic

The next controlled intervention should not change the trained representation.

Test a small prototype beam: retain the top two learned prototypes and a bounded
K=8 state shortlist, then cosine-rerank those eight states. The purpose is to
measure whether the current failure is primarily one-prototype coarse-routing
recall rather than prototype representation quality.

Any such beam must report proposal retention, final target recall, shortlist size,
prototype-score cost, reranking cost, and total build arithmetic separately.

## Reproduction

Measurement command: python scripts/run_c5_prototype_selective_001.py

Contract: contracts/TACOSM-C5-PROTOTYPE-SELECTIVE-001.json