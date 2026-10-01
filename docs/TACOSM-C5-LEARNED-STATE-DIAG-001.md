# TACOSM-C5-LEARNED-STATE-DIAG-001

Status: PREREGISTERED — result pending.

## Purpose

C5-LEARNED-STATE-001 failed at the learned binary state-index boundary.
This diagnostic holds the learned encoder fixed and removes quantization from
one arm.

It compares:

`continuous learned scoring -> state`

against

`learned encoder -> 8-bit binary code -> radius-1 index -> state`.

A random frozen encoder provides the learning control.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor; candidate history is irrelevant to this state-only test.
- 5 seeds (0..4).
- 32 training epochs.
- 100 evaluation queries per seed.
- input width 10; latent width 8.
- binary width 8; Hamming radius 1; K=1.
- training codes = CODEBOOK[16:64].
- evaluation target codes = CODEBOOK[:16].
- same A1 H-invariant query/state stream as C5-LEARNED-STATE-001.

## Arms

### No-learning continuous

Use a random frozen dual encoder. Precompute state embeddings, then scan all
64 state items using continuous dot-product scores.

### Learned continuous

Train the same dual encoder for 32 epochs on the 48 disjoint training codes.
Precompute state embeddings, then scan all 64 items using continuous scores.

### Learned binary

Use the same trained encoder from the learned-continuous arm. Quantize its first
8 latent coordinates by sign, build the 8-bit bucket index, probe the radius-1
neighborhood, and retain at most one address.

## Diagnostic rule

- learned continuous > no-learning continuous, while binary trails continuous
  by at least 0.20: the main observed loss is at the discrete retrieval boundary;
- learned continuous <= no-learning continuous: the current representation is
  itself not demonstrating held-out semantic retrieval;
- learned continuous > no-learning continuous, while binary is within 0.20 of
  continuous: the prior C5-LEARNED-STATE-001 failure is not primarily explained
  by binary quantization.

These are diagnostic branches, not a broad C5 verdict.

## Cost accounting

Continuous query cost:

- 80 MACs to encode the query;
- 64 x 8 = 512 MACs to score against cached state embeddings;
- total 592 MACs/query, excluding state-build encoding.

Binary query cost:

- 80 MACs to encode the query;
- 9 dictionary probes;
- state build = 64 x 80 = 5,120 MACs, amortizable over repeated queries.

No claim of hardware speedup is made from these arithmetic counts.

## Interpretation boundary

The diagnostic isolates one representation/index boundary on the same synthetic
task. It does not establish general semantic memory, natural-language
retrieval, long-horizon persistence, or broad C5.

## Reproduction

Measurement command:

`python scripts/run_c5_learned_state_diag_001.py`

Contract:

`contracts/TACOSM-C5-LEARNED-STATE-DIAG-001.json`
