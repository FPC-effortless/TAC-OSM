# TACOSM-C5-PROTOTYPE-SELECTIVE-001

Status: PREREGISTERED — result pending.

## Purpose

C5-COSINE-SELECTIVE-001 showed that the current learned sign-code proposal loses
too much semantic recall even after cosine reranking. This experiment removes
binary sign bucketing and replaces it with learned continuous prototype routing.

The learned state representation remains width 16 with the raw training
objective, eight mean negatives, one positive view, and 512 epochs.

## Registered protocol

- M=64 persistent state items.
- H=64, 128, 256.
- 5 seeds.
- latent width=16.
- 512 epochs.
- one positive view.
- eight mean-gradient negatives.
- training codes CODEBOOK[16:64].
- evaluation codes CODEBOOK[:16].
- 16 learned prototypes.
- capacity-four state buckets.
- one prototype selected per query.
- cosine reranking over at most four states.
- exact candidate index and fixed executor unchanged.

## Prototype learning

Prototype centroids are learned only from the 48 training-code embeddings of
the trained state encoder. Evaluation-state embeddings are not used to fit the
centroids.

Centroids are initialized deterministically by farthest-point selection and
refined with eight cosine k-means iterations.

At state-index build time, the 64 evaluation state embeddings are assigned to
prototype buckets with a hard capacity of four. Assignment is performed without
query information or gold labels.

At query time, the query embedding is compared with the 16 prototypes. The
single best prototype is selected, its bucket is retrieved, and cosine scoring
selects the final state from at most four candidates.

## Primary endpoint

Actual target-state recall against the hidden environment target.

Secondary endpoints:

- proposal target retention before reranking;
- actual end-to-end success against hidden environment truth;
- shortlist size;
- build and query arithmetic.

## Capability rule

The selective arm must remain within 0.05 of exhaustive cosine on both actual
target-state recall and actual end-to-end success, while keeping shortlist size
at or below four.

## Cost ledger

At width 16:

- query projection = 160 MACs;
- query-to-16-prototype scoring = 16 x 16 = 256 MACs;
- four-state cosine reranking = at most 4 x 16 = 64 MACs;
- selective projection + reranking = at most 480 MACs/query;
- exhaustive projection + state scoring = 1,184 MACs/query;
- state-score arithmetic reduction = 93.75%;
- total projection+score arithmetic reduction = 1 - 480/1184 = 59.5%.

Prototype build arithmetic is reported separately:

- 48 training embeddings = 7,680 MACs;
- eight k-means iterations over 48 x 16 assignments = 98,304 MACs;
- 64 evaluation-state embeddings = 10,240 MACs;
- 64 x 16 prototype assignment scoring = 16,384 MACs;
- total registered prototype build arithmetic = 132,608 MACs.

These are arithmetic operation counts, not hardware latency measurements.

## Interpretation boundary

A pass supports only the bounded prototype-routing mechanism on the registered
synthetic workload. It does not establish general semantic memory, universal
ANN retrieval, hardware speedup, or broad C5.

## Reproduction

Measurement command: python scripts/run_c5_prototype_selective_001.py

Contract: contracts/TACOSM-C5-PROTOTYPE-SELECTIVE-001.json