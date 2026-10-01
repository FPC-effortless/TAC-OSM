# TACOSM-C5-FACTORIZATION-DEPTH-FIXED-RATIO-001

Status: PRE-REGISTERED.

## Purpose

Previous C5 measurements show that deeper product-key factorization can reduce
the candidate fraction, but the existing comparisons changed factor size and
beam together. This experiment isolates factorization depth at a fixed
per-factor beam ratio.

## Registered intervention

All arms use factor_size=8 and factor_beam=5, so B/F=0.625.

- d=2: two factors
- d=3: three factors
- d=4: four factors
- d=5: five factors

The same generalized MultiFactorProductKeyStateIndex implementation is used
for every depth, eliminating a separate implementation comparison.

## Geometric prediction

If factor occupancy is approximately uniform, the admitted product-cell
fraction is:

(B/F)^d = (5/8)^d

which gives expected scored fractions of:

- d=2: 39.06%
- d=3: 24.41%
- d=4: 15.26%
- d=5: 9.54%

These values are geometric reference points, not pass/fail thresholds.

## Registered protocol

- H=256
- M={128,256,512}
- K=32
- factor_size=8
- factor_beam=5
- factor_count={2,3,4,5}
- seeds=10-19
- 100 evaluation tasks per seed/M/arm
- width-16 raw-trained cosine teacher
- 48 fixed training codes and 16 held-out target codes
- one-bit noisy queries
- eight cosine k-means iterations
- training-only codebook fitting
- state-distinct downstream executor
- exact pooled count aggregation

## Primary endpoint

target_recall_retention_ratio = selective target recall / exhaustive target recall.

An arm meets the capability floor only when retention is at least 0.90 at all
three registered M levels.

## Secondary endpoints

Record proposal target retention, exhaustive target recall, states scored/M,
absolute states scored, pair-generation operations, total MACs, nonempty cell
count, and maximum cell occupancy.

## Interpretation

The key test is whether depth changes retention beyond what B/F predicts while
the reranking fraction follows the expected geometric reduction.

A depth result that reaches a low scored fraction but loses capability is a
capacity/geometry tradeoff, not a sparse-compute win. A result that preserves
capability while approaching the geometric reference would support depth as a
mechanism for lowering the compute constant on this workload.

No result establishes sublinear asymptotics because factor size and beam remain
fixed while M changes.

## Integrity

This experiment is separate from PR #43 and from the exhaustive-reference
calibration study. No outcome from either study changes this registration.

## Reproduction

Smoke:
python scripts/run_c5_factorization_depth_fixed_ratio_001.py --smoke

Full:
python scripts/run_c5_factorization_depth_fixed_ratio_001.py

Artifact:
artifacts/TACOSM-C5-FACTORIZATION-DEPTH-FIXED-RATIO-001.json

Full-run trigger:
[run-c5-factorization-depth-fixed-ratio-001-full]