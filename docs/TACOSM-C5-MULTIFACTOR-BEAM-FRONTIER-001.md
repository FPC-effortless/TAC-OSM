# TACOSM-C5-MULTIFACTOR-BEAM-FRONTIER-001

Status: PRE-REGISTERED.

## Purpose

The ten-seed factorization-depth measurement showed that a three-factor
factor-size-8 index at beam 3 reduces state reranking to about 6.5% of M,
but target-recall retention is below the established 0.90 capability floor
at M=128 and M=256.

This experiment keeps factorization depth and granularity fixed and increases
the factor beam from 3 to 4, 5, and 6. The purpose is to test whether the
coverage loss can be recovered without returning to the roughly 15% candidate
fraction of the validated two-factor factor16/beam6 reference.

## Registered arms

- three_factor_8_beam4
- three_factor_8_beam5
- three_factor_8_beam6

All use three factors, factor size 8, K=32, H=256, M={128,256,512}, and
seeds 10-19.

## Primary capability constraint

For each arm and M:

target_recall_retention_ratio = selective target recall / exhaustive target recall

The pre-registered capability floor is 0.90 at every M.

## Primary computation diagnostic

states_scored_over_M is the mean number of persistent-state embeddings
subjected to continuous reranking divided by M.

The final K=32 shortlist is not substituted for this measure.

## Interpretation

An arm meeting the 0.90 floor at all M levels is capability-eligible. The
compute result is then interpreted relative to the previously measured
two-factor factor16/beam6 reference.

No configuration is privileged in advance, and no universal beam optimum,
semantic retrieval guarantee, asymptotic theorem, or hardware speedup is licensed.

## Provenance

The protocol inherits the exact ten-seed stream and state-distinct executor
from TACOSM-C5-MULTIFACTOR-001. The prior two-factor reference is reported
unchanged rather than retuned or recomputed.

## Reproduction

Smoke: python scripts/run_c5_multifactor_beam_frontier_001.py --smoke

Full: python scripts/run_c5_multifactor_beam-frontier_001.py

Artifact: artifacts/TACOSM-C5-MULTIFACTOR-BEAM-FRONTIER-001.json

Full-run trigger: [run-c5-multifactor-beam-frontier-full]

## Operational trigger

The commit carrying the registered trigger token only dispatches the already
registered measurement; this trigger revision contains the contract-schema
correction only and changes no arm, threshold, seed, or analysis parameter.
