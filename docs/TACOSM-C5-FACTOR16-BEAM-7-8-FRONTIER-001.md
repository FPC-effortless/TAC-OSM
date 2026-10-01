# TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001

Status: PRE-REGISTERED.

## Motivation

The measured factor-size-16, three-factor beam-4/5/6 experiment substantially
reduced reranked-state fraction, but no beam met the registered 0.90 retention
floor across all M levels. The artifact also shows that, at M=128, the maximum
proposal/admission retention among beams 4-6 is only 0.705. Therefore beams 4-6
cannot reach 0.90 final retention even with a perfect post-admission reranker.

## Registered intervention

- factor_count = 3
- factor_size = 16
- factor_beam = 7 or 8
- H = 256
- M = {128,256,512}
- K = 32
- seeds = 10-19
- 100 evaluation tasks per seed/M/arm
- width-16 cosine teacher
- training-only codebook fitting
- state-distinct downstream executor
- exact pooled aggregation

## Geometric reference

At fixed factor size and factor count, the approximate product-cell fractions are:

- beam 7: (7/16)^3 = 8.37%
- beam 8: (8/16)^3 = 12.50%

Both are below the prior robust two-factor reference fraction of 15.4286%.

These are geometric reference points, not pass/fail thresholds.

## Primary endpoint

target_recall_retention_ratio = selective target recall / exhaustive target recall.

A beam is capability-eligible only when retention >= 0.90 at M=128, 256, and 512.
An eligible beam is a registered frontier improvement only if its mean
states_scored_over_M is strictly below 0.1542864583.

## Decomposition diagnostic

Report proposal_target_retention and:

conditional_selection_given_admission = selective_target_recall / proposal_target_retention

when proposal retention is nonzero.

This distinguishes coarse product-cell admission loss from post-admission
cosine-selection loss.

## Interpretation boundary

If beam 7 or 8 reaches the capability floor while remaining below the reference
reranking fraction, this supports a narrow workload-specific frontier extension.
If neither does, then factor-size-16 admission remains insufficient even below
the 15% state-reranking reference, motivating a change to the addressing mechanism
rather than another small beam increase.

No result establishes sublinear asymptotic scaling, semantic retrieval guarantees,
universal beam optimality, or hardware speedup.

## Integrity

This is a new preregistered experiment. It does not modify the completed beam-4/5/6
measurement or its verdict. The exhaustive-reference calibration study remains
separate.

## Reproduction

Smoke:

python scripts/run_c5_factor16_beam_7_8_frontier_001.py --smoke

Full:

python scripts/run_c5_factor16_beam_7_8_frontier_001.py

Artifact:

artifacts/TACOSM-C5-FACTOR16-BEAM-7-8-FRONTIER-001.json

Full-run trigger:

[run-c5-factor16-beam-7-8-frontier-001-full]