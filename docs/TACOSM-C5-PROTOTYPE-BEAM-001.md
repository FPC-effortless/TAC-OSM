# TACOSM-C5-PROTOTYPE-BEAM-001

Status: PREREGISTERED — result pending.

## Purpose

C5-PROTOTYPE-SELECTIVE-001 failed because one learned prototype retained the
true target only about 31% of the time. This experiment asks whether that loss
is mainly a coarse-routing beam-width problem.

The encoder and 16 learned prototypes are held fixed from the prior protocol.
Only query-time beam width changes from one prototype to two, increasing the
bounded state shortlist from K=4 to K=8.

## Registered protocol

- M=64 persistent state items.
- H=64, 128, 256.
- 5 seeds.
- width-16 raw-trained encoder.
- 512 epochs, one positive view, eight mean negatives.
- 16 prototypes learned from the 48 training codes.
- capacity four per prototype bucket.
- top-two prototype beam.
- cosine reranking over at most eight states.
- exact candidate index and fixed executor unchanged.

## Primary endpoint

Actual target-state Top-1 recall against hidden environment truth.

Secondary:

- true-target proposal retention before cosine reranking;
- actual end-to-end success;
- shortlist size;
- prototype and reranking arithmetic.

## Capability rule

Selective target recall and selective end-to-end success must both remain within
0.05 of their exhaustive-cosine references, with shortlist size <=8.

## Cost ledger

At latent width 16:

- query projection = 160 MACs;
- query-to-16-prototype scoring = 256 MACs;
- eight-state cosine reranking = 128 MACs;
- bounded selective query arithmetic = 544 MACs;
- exhaustive query arithmetic = 1,184 MACs;
- total projection+score arithmetic reduction = 1 - 544/1184 = 54.1%;
- prototype build remains the prior 132,608 MACs.

## Interpretation boundary

A passing result would show that the one-prototype failure was substantially
beam-width related. A failure would move attention back toward prototype/state
representation or prototype assignment geometry.

Neither outcome establishes universal semantic retrieval or broad C5.

## Reproduction

Measurement command: python scripts/run_c5_prototype_beam_001.py

Contract: contracts/TACOSM-C5-PROTOTYPE-BEAM-001.json