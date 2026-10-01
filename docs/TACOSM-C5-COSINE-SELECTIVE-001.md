# TACOSM-C5-COSINE-SELECTIVE-001

Status: PREREGISTERED — result pending.

## Purpose

C5-SIMILARITY-EVAL-001 showed that the same learned width-16 encoder reaches
0.700 target-state recall with cosine inference versus 0.460 with raw-dot
inference.

This experiment reintroduces a bounded learned retrieval boundary:

learned 8-bit proposal -> bounded shortlist -> cosine reranking.

The downstream candidate index and executor remain unchanged.

## Registered protocol

- M=64 persistent state items.
- H=64, 128, 256.
- 5 seeds.
- latent width=16.
- 512 training epochs.
- one positive view.
- eight mean-gradient negatives.
- 100 evaluation queries per seed/H.
- learned 8-bit sign index.
- Hamming radius=2.
- state shortlist K=4.
- exact candidate index unchanged.
- fixed 12-work-unit executor unchanged.

## Arms

### Exhaustive cosine

Encode the query, cosine-score all 64 cached state embeddings, select the
highest-scoring state, perform the exact candidate lookup, and execute its four
candidate programs.

### Selective cosine

Encode the query, use the learned binary state index to probe the radius-2
neighborhood, retain at most four addresses, cosine-rerank only those retained
states, then perform the exact candidate lookup and execute.

The same learned weights are used in both arms.

## Primary endpoint

Actual target-state recall is whether the selective arm selects the hidden
environment's target state for the same query.

A reference endpoint also records whether the selective proposal contains the
state selected by exhaustive cosine.

## Secondary endpoint

Actual end-to-end success is whether the selective downstream output matches
the hidden environment truth. The exhaustive-cosine arm is reported as a
capability reference, not as an oracle.

## Capability rule

The selective arm must remain within 0.05 of the exhaustive-cosine arm on both
actual target-state recall and actual end-to-end success, while retaining at
most K=4 states per query.

## Cost ledger

At width 16:

- query projection = 160 MACs/query;
- exhaustive cosine state scoring = 64 x 16 = 1,024 MACs/query;
- selective cosine reranking = at most 4 x 16 = 64 MACs/query;
- binary proposal = 37 radius-2 bucket probes;
- one-time state embedding/build = 10,240 MACs/build;
- vector normalization operations reported separately;
- downstream executor work reported unchanged.

## Interpretation boundary

A passing result supports a bounded learned selective retrieval mechanism on
this synthetic workload. It does not establish general semantic indexing,
universal retrieval guarantees, hardware speedup, or broad C5.

A failure identifies the registered learned index plus cosine-reranking
boundary as insufficient for the current exhaustive-cosine reference.

## Reproduction

Measurement command: python scripts/run_c5_cosine_selective_001.py

Contract: contracts/TACOSM-C5-COSINE-SELECTIVE-001.json