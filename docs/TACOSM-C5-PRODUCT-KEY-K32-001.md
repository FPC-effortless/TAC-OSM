# TACOSM-C5-PRODUCT-KEY-K32-001

Status: PREREGISTERED — corrected 996-test head; K32 measurement trigger repeated.

## Purpose

The measured K=16 product-key arm retained 87.1% of target states and selected
62.4%. This diagnostic asks whether a K=32 shortlist closes the remaining
coverage gap without reverting to the 64-state exhaustive reference.

Only the factor beam and shortlist budget change. The teacher representation,
training-only codebooks, runtime cells, query stream, and downstream executor
are held fixed.

## Registered protocol

- M=64 persistent states.
- H=64, 128, 256.
- Five seeds.
- Width-16 raw-trained cosine teacher.
- 512 teacher-training epochs.
- Eight mean-gradient negatives.
- 48 training codes and 16 held-out query codes.
- Product-key factor size 8 with eight k-means iterations.
- factor beam 6, producing up to 36 factor cells before the K cap.
- K=32 retained shortlist.

## Measurement

The exhaustive reference scores all 64 states.
The product-key arm computes the teacher query embedding once, scores the two
factor codebooks, generates the Cartesian beam, scores every over-fetched state
candidate with the teacher cosine similarity, then retains at most 32 states.

The compute ledger charges all over-fetched state scoring, not just the final
retained shortlist.

## Interpretation boundary

This is a budget diagnostic. It does not establish a general retrieval
guarantee or hardware speedup. The result is interpreted against the fixed
exhaustive reference and the preceding K=16 product-key run.

## Reproduction

Smoke: python scripts/run_c5_product_key_k32_001.py --smoke

Full: python scripts/run_c5_product_key_k32_001.py

Artifact: artifacts/TACOSM-C5-PRODUCT-KEY-K32-001.json
