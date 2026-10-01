# TACOSM-C5-LATENT-WIDTH-001

Status: PREREGISTERED — result pending.

## Purpose

The learned-state diagnostics establish that the current encoder contains real
held-out semantic signal, while more training budget and more positive views do
not remove the low Top-1 boundary. The next isolated variable is representation
capacity.

This experiment varies latent width only.

Measured path:

noisy query -> learned continuous semantic encoder -> exhaustive state score.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor.
- 5 seeds (0..4).
- latent widths 8, 16, 32.
- 512 epochs.
- 100 held-out noisy queries per seed/width.
- eight mean-gradient negatives per positive.
- one deterministic positive view per update.
- dual linear input width = 10.
- learning rate 0.02; margin 0.25.
- train codes = CODEBOOK[16:64].
- evaluation codes = CODEBOOK[:16].
- same A1 H-invariant stream.
- continuous retrieval only.

Every arm has exactly 24,576 optimizer updates.

## Primary endpoint

Target-state Top-1 recall.

Secondary:

- target rank;
- training update count;
- negative-evaluation count;
- inference arithmetic.

## Decision rule

- latent-32 >= latent-8 + 0.15 and >= 0.50: material capacity effect;
- latent-32 within 0.05 of latent-8: saturation;
- latent-32 improves by >0.05 and <0.15: partial capacity effect;
- latent-32 < latent-8 - 0.05: wider representation reverses the current gain.

These are diagnostic, workload-specific rules.

## Cost

Inference arithmetic scales with latent width:

| Width | Query encode | 64-state scoring | Total query MACs | State build MACs |
|---:|---:|---:|---:|---:|
| 8 | 80 | 512 | 592 | 5,120 |
| 16 | 160 | 1,024 | 1,184 | 10,240 |
| 32 | 320 | 2,048 | 2,368 | 20,480 |

Training also scales with width for the same number of optimizer updates and
negative/view evaluations.

## Interpretation boundary

A width response identifies capacity as a factor in this synthetic semantic
retriever. It does not establish learned indexing, sublinear addressing,
natural-language semantic memory, or broad C5.

Binary indexing remains excluded.

## Reproduction

Measurement command:

python scripts/run_c5_latent_width_001.py

Contract:

contracts/TACOSM-C5-LATENT-WIDTH-001.json
