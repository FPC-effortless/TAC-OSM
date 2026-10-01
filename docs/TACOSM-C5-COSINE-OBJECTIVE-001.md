# TACOSM-C5-COSINE-OBJECTIVE-001

Status: PREREGISTERED — result pending.

## Purpose

The learned-state evidence now shows:

- genuine learned signal above random;
- no reliable improvement from simply increasing training budget;
- a strong early benefit from broader negative coverage, then a plateau at
  eight negatives;
- degraded retrieval from averaging multiple positive views;
- only a modest width effect through latent dimension 32.

The next isolated intervention is similarity/objective geometry.

This experiment compares the existing raw dot-product margin objective with a
cosine-normalized margin objective.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor.
- 5 seeds (0..4).
- latent width = 16.
- 512 epochs.
- 24,576 optimizer updates per arm.
- one positive view per update.
- eight mean-gradient negatives per update.
- learning rate 0.02; margin 0.25.
- train codes = CODEBOOK[16:64].
- evaluation codes = CODEBOOK[:16].
- A1 H-invariant query/state stream.
- continuous state scoring only.

## Arms

### Raw objective

Existing raw dot-product margin training.

### Cosine objective

L2-normalize query, positive-state, and negative-state embeddings inside the
training objective and backpropagate through the normalization.

## Fixed evaluation metric

Both arms are evaluated with the same cosine similarity over the 64 cached
state embeddings.

This fixes the inference scoring geometry and isolates the effect of training
objective.

The primary endpoint is target-state Top-1 recall.

## Decision rule

- cosine >= raw +0.10 and cosine >=0.50: material normalized-objective effect;
- cosine within 0.05 of raw: no material change;
- cosine < raw -0.05: normalized objective is harmful in this configuration;
- otherwise: partial objective-geometry effect.

The negative branch is explicit so an adverse result cannot be silently routed
through a positive-only catch-all.

## Cost

At latent width 16:

- query linear projection = 160 MACs/query;
- 64-state dot products = 1,024 MACs/query;
- fixed projection/dot-product arithmetic = 1,184 MACs/query;
- state build = 10,240 MACs/build.

Cosine evaluation adds vector norm/division operations. Those are reported as
normalization operations rather than being converted into MACs.

Training cost remains 24,576 optimizer updates for each arm, but the cosine arm
adds normalization operations inside every positive/negative scoring pair.

## Interpretation boundary

A positive result identifies score geometry as a contributing factor. It does
not establish an optimal objective, learned selective retrieval, sublinear
addressing, or broad C5.

## Reproduction

Measurement command:

python scripts/run_c5_cosine_objective_001.py

Contract:

contracts/TACOSM-C5-COSINE-OBJECTIVE-001.json
