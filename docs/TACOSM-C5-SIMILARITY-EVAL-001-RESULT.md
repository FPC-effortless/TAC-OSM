# TACOSM-C5-SIMILARITY-EVAL-001 RESULT

## Run provenance

- workflow: 36671378156
- clean run: #2
- clean measurement head: c3a80aa9ac9d553320c4a3b83582403a7aebedd7
- precondition gate: passed
- full test suite: 945 passed
- measurement step: passed
- artifact: TACOSM-C5-SIMILARITY-EVAL-001
- artifact id: 11078295955
- artifact digest: sha256:fcbc63a364226bcd95c4b7d7386f286ca3eea3428acf84098e79574720dc7041

The first workflow run started before the 945-test synchronization and is quarantined. Run #2 is the clean measurement.

## Registered comparison

The same trained encoder was used for both inference arms:

- M=64 persistent state items;
- latent width 16;
- 512 epochs;
- one positive view;
- eight mean-gradient negatives;
- raw dot-product training objective;
- 48 training codes and 16 disjoint held-out evaluation codes;
- A1 H-invariant query/state stream;
- 100 held-out queries per seed.

Only inference similarity geometry changed:

- raw dot product;
- cosine similarity.

No model weights differ between arms.

## Target-state Top-1 recall

| Seed | Raw dot | Cosine | Delta |
|---:|---:|---:|---:|
| 0 | 0.45 | 0.70 | +0.25 |
| 1 | 0.52 | 0.70 | +0.18 |
| 2 | 0.45 | 0.70 | +0.25 |
| 3 | 0.45 | 0.70 | +0.25 |
| 4 | 0.43 | 0.70 | +0.27 |
| **Mean** | **0.460** | **0.700** | **+0.240** |

Cosine inference improves target-state recall in every registered seed.

## Target rank

| Seed | Raw dot | Cosine |
|---:|---:|---:|
| 0 | 2.19 | 1.30 |
| 1 | 2.00 | 1.30 |
| 2 | 1.98 | 1.30 |
| 3 | 2.08 | 1.30 |
| 4 | 2.29 | 1.30 |
| **Mean** | **2.108** | **1.300** |

Cosine inference moves the correct state to mean rank 1.30 from 2.108.

## Preregistered decision

The cosine-minus-raw recall difference is:

0.700 - 0.460 = +0.240.

The preregistered strong threshold required at least +0.10 and cosine recall >=0.50.

Both conditions are satisfied.

Therefore the experiment enters the **material_cosine_effect** branch.

This establishes that inference-time similarity geometry materially affects retrieval for the current learned state representation on this workload.

It does not establish general cosine superiority, nor does it by itself establish selective retrieval or broad C5.

## Inference cost

At latent width 16 both arms perform:

- query projection = 160 MACs/query;
- 64-state score products = 1,024 MACs/query;
- projection + dot-product arithmetic = 1,184 MACs/query.

Cosine adds normalization operations:

- 130 recorded normalization operations/query;
- normalization is not converted into MACs in this ledger.

Thus the recall gain comes with an additional normalization operation count, while the linear projection/dot-product arithmetic remains unchanged.

These arithmetic counts are proxies, not hardware latency measurements.

## Scientific interpretation

This resolves the earlier inference-metric confound.

The raw objective is not inherently producing only ~0.46 recall. When the same trained weights are normalized at inference, the exact same model reaches 0.70 recall on the registered held-out state-retrieval task.

Combined with the cosine-objective null, this indicates that the dominant difference is introduced at **inference-time similarity geometry**, not by changing the training objective from raw dot product to cosine.

The current best continuous state-retrieval measurement is therefore 0.700 with mean rank 1.30 at M=64.

That is materially stronger than the earlier 0.402 raw-dot width-8 baseline and 0.460 raw-dot width-16 result, but it is still below a safe high-recall selective boundary.

## Implication for C5

The learned state representation now appears strong enough to justify testing an actual bounded retrieval boundary.

The next experiment should therefore revisit learned selective state retrieval using cosine similarity as the scoring geometry, while preserving strict target-state retention as the primary endpoint.

The first discrete/index test should use cosine similarity to re-rank a bounded approximate shortlist rather than relying only on binary sign-code identity. The shortlist size and index probe/build costs must be explicit.

Broad C5 remains **ungraded** until that learned selective path demonstrates capability parity and an auditable total-cost advantage over exhaustive retrieval.

## Reproduction

Measurement command: python scripts/run_c5_similarity_eval_001.py

Contract: contracts/TACOSM-C5-SIMILARITY-EVAL-001.json