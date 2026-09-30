# TACOSM-C5-PRODUCT-KEY-FRONTIER-001

Status: PRE-REGISTERED.

## Purpose

The completed cell-granularity experiment showed that increasing factor size
from 8 to 16 to 32 reduces the reranking fraction from roughly 0.56M to 0.15M
to 0.048M, but also reduces proposal coverage.

This experiment measures the capability/computation frontier by increasing the
factor beam at factor sizes 16 and 32. The factor size controls cell resolution;
the beam controls how many cells are admitted before continuous reranking.

## Registered protocol

- H=256.
- M={128,256,512}.
- K=32.
- factor sizes={16,32}.
- factor beams:
  - factor 16: {4,6,8,12,16}
  - factor 32: {4,6,8,12}
- Five seeds.
- 100 evaluation steps per seed/M/arm.
- Eight cosine k-means iterations.
- Width-16 raw-trained cosine teacher.
- 512 teacher-training epochs with eight mean-gradient negatives.
- 48 training codes versus 16 held-out target codes.
- Training-only codebook fitting.

The factor-8/beam-6 result is retained as the previously measured reference
rather than rerun in this experiment.

## Primary capability constraint

For each M, compute:

target_recall_retention_ratio =
selective_target_recall / exhaustive_target_recall.

An arm is reported as satisfying the capability constraint only when this ratio
is at least 0.90 at every tested M.

The raw recall, proposal retention and compute metrics remain visible so the
90% floor cannot conceal a coverage or arithmetic regression.

## Primary compute diagnostic

states_scored_over_M =
mean number of state embeddings subjected to the final continuous reranker / M.

This is distinct from K=32, which bounds only the retained shortlist.

## Why the beam sweep matters

At factor size 16, the prior beam-6 arm achieved approximately 0.15M state
scoring with proposal retention around 0.79–0.83.

At factor size 32, beam-6 achieved approximately 0.048M state scoring but only
about 0.53–0.56 proposal retention.

Increasing the beam tests whether those coverage losses are recoverable without
returning to exhaustive state scoring.

## Interpretation boundary

This is a controlled synthetic-workload frontier. It does not establish
semantic memory performance, universal index scaling, or hardware speedup.
The output is intended to determine whether a viable compute/capability region
exists before introducing another addressing mechanism.

## Reproduction

Smoke: python scripts/run_c5_product_key_frontier_001.py --smoke

Full: python scripts/run_c5_product_key_frontier_001.py

Artifact: artifacts/TACOSM-C5-PRODUCT-KEY-FRONTIER-001.json

Full measurement trigger: commit message [run-c5-frontier-full] on this branch is the registered measurement trigger.
