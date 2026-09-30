# TACOSM-C5-PRODUCT-KEY-001

Status: PREREGISTERED — corrected 994-test head; full measurement triggered.

## Purpose

The sparse-funnel result shows a clear proposal-coverage limitation even at
K=16. This diagnostic changes the addressing mechanism while holding the
width-16 cosine teacher and downstream executor fixed.

The mechanism factorizes each state embedding into two subvectors. Separate
training-only cosine codebooks are fitted for the two factors. Runtime states
are assigned to the Cartesian product cells, and a small factor beam proposes
candidate cells. The fixed cosine teacher then selects within the retained
state set.

## Registered protocol

- M=64 persistent states.
- H=64, 128, 256.
- Five seeds.
- Width-16 raw-trained cosine teacher.
- 512 teacher-training epochs with eight mean-gradient negatives.
- 48 training semantic codes and 16 held-out query codes.
- Two factor codebooks with factor size 8.
- Eight cosine k-means iterations.
- K=4, 8, 16 with factor beams 1, 2, 4.
- Training-only codebook fitting; all 64 runtime states are assigned to cells.
- Fixed cosine reranking and fixed downstream executor.

## Addressing

For an embedding z=[z_1,z_2], the two factor codebooks define

cell(z) = (argmax_i <z_1,p_i>, argmax_j <z_2,q_j>).

At query time the top-B factor codes for each factor form B^2 cells.
The union is scored by the fixed cosine teacher. The implementation records
the number of states actually scored before the K cap, so over-fetch arithmetic
is not hidden by the retained-shortlist size.

## Leakage boundary

Factor codebooks are fitted only from the 48 training-state embeddings.
The 16 held-out evaluation-state embeddings are not used to fit factor
centers. They are only assigned to the fixed cells at runtime.

## Compute

The exhaustive teacher reference is 1,184 MACs/query.
Product-key query cost is:

teacher query projection + factor-code scoring + teacher cosine scoring over
the actual over-fetched state candidates.

Build arithmetic includes factor-codebook fitting and runtime state assignment.
Arithmetic is not treated as hardware wall-clock speedup.

## Interpretation boundary

This is a mechanism diagnostic, not a claim of universal product-key
retrieval guarantees. A passing result would support only the registered
workload and budget. A failure would localize the limitation away from the
current distilled prototype addressing toward representation or query geometry.

## Reproduction

Smoke: python scripts/run_c5_product_key_001.py --smoke

Full: python scripts/run_c5_product_key_001.py

Artifact: artifacts/TACOSM-C5-PRODUCT-KEY-001.json
