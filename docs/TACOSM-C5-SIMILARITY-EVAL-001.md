# TACOSM-C5-SIMILARITY-EVAL-001

Status: PREREGISTERED — result pending.

## Purpose

The cosine-objective experiment produced identical raw-objective and
cosine-objective training results when both were evaluated with cosine
similarity. The earlier latent-width experiment used raw dot-product inference.

This experiment isolates the remaining confound: inference metric geometry.

The exact same trained encoder is scored two ways:

raw dot product -> state

versus

cosine similarity -> state.

No model weights or training objective differ between arms.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor.
- 5 seeds.
- latent width = 16.
- 512 epochs.
- one positive view.
- eight mean-gradient negatives.
- 48 training codes; 16 held-out evaluation codes.
- A1 H-invariant query/state stream.
- 100 held-out noisy queries per seed.
- same raw-dot training objective in every seed.

## Arms

### Raw-dot evaluation

For each query, select the highest raw query/state dot product.

### Cosine evaluation

For the exact same query and cached state embeddings, L2-normalize both
vectors and select the highest cosine similarity.

The trained model is identical between the arms.

## Primary endpoint

Target-state Top-1 recall.

Secondary:

- target rank;
- query projection arithmetic;
- state score arithmetic;
- normalization operations.

## Decision rule

- cosine >= raw +0.10 and >=0.50: material inference-metric effect;
- cosine within 0.05 of raw: no material inference-metric change;
- cosine < raw -0.05: cosine harmful in this configuration;
- otherwise: partial inference-metric effect.

## Cost

At latent width 16:

- query projection = 160 MACs/query;
- 64-state dot products = 1,024 MACs/query;
- raw total = 1,184 MACs/query;
- cosine uses the same projection/dot products plus vector normalization;
- normalization operations are reported separately.

## Interpretation boundary

This experiment answers only whether inference metric geometry explains the
difference between earlier raw-dot and cosine-evaluated retrieval results.

It does not establish learned indexing, sublinear addressing, natural-language
semantic memory, or broad C5.

## Reproduction

Measurement command:

python scripts/run_c5_similarity_eval_001.py

Contract:

contracts/TACOSM-C5-SIMILARITY-EVAL-001.json
