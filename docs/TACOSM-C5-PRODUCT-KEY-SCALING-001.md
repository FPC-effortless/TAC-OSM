# TACOSM-C5-PRODUCT-KEY-SCALING-001

Status: PREREGISTERED — corrected M-level contract; full scaling measurement triggered.

## Purpose

The K=32 product-key experiment reaches near-reference retrieval capability at
M=64, but K=32 retains about half of the state pool. This experiment tests the
actual scaling boundary by holding K=32 fixed while M grows.

H is held at 256 so persistent-state population is the only intervention.
The teacher representation, training-only codebook fitting, factor size, factor
beam, query construction, and downstream executor remain fixed.

## Registered protocol

- M=64, 128, 256, 512 persistent states.
- H=256.
- Five seeds.
- K=32 and factor beam 6.
- Factor size 8 with eight k-means iterations.
- Width-16 raw-trained cosine teacher.
- 512 teacher-training epochs and eight mean-gradient negatives.
- 48 training codes versus 16 held-out target codes.

Each task contains the 48 training-code states, exactly one held-out target-code
state, the remaining held-out evaluation codes when available, and then unique
deterministic binary decoys. That state population and its opaque addresses are fixed across evaluation steps. Every task contains exactly one state whose value
equals the selected target code.

## Primary scaling quantities

The primary retrieval endpoint is actual target-state Top-1 recall.
Supporting scaling quantities are:

- proposal target retention;
- K/M;
- states scored/M before shortlist truncation;
- total query arithmetic;
- end-to-end success.

Exhaustive reference arithmetic is query projection plus M state cosine scores,
so its state-scoring component grows linearly with M. Product-key arithmetic
includes the factor scores and every over-fetched state scored before the K cap.

## Interpretation boundary

A fixed-K result at M=64 is not sufficient evidence for sparse scaling.
The intended question is whether the same bounded computation remains useful
as M grows and K/M falls.

The experiment does not claim a general sublinear-compute law, universal ANN
guarantees, or hardware speedup.

## Reproduction

Smoke: python scripts/run_c5_product_key_scaling_001.py --smoke

Full: python scripts/run_c5_product_key_scaling_001.py

Artifact: artifacts/TACOSM-C5-PRODUCT-KEY-SCALING-001.json

Run trigger marker: fixed-K scaling harness fixed; final registered measurement trigger repeated.
