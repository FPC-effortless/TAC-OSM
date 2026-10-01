# TACOSM-C5-MULTIFACTOR-BEAM-001

Status: PRE-REGISTERED.

## Purpose

The measured factorization-depth experiment established that three-factor factor-size-8 beam-3 addressing reduces state reranking from about 15% of M to about 6.5% of M, while losing capability at M=128 and M=256. This experiment holds factor count and factor size fixed and increases beam from 3 to 6 to test whether proposal coverage can be recovered before compute cost returns to the two-factor regime.

## Registered arms

- three_factor_8_beam3
- three_factor_8_beam4
- three_factor_8_beam5
- three_factor_8_beam6

Shared settings:
- H=256
- M={128,256,512}
- K=32
- factor_count=3
- factor_size=8
- seeds={10,...,19}
- 100 evaluation steps per seed/M/arm
- width-16 raw-trained cosine teacher
- training-only codebook fitting
- eight k-means iterations
- state-distinct downstream executor
- same task stream

## Capability floor

A configuration is capability-qualified only if exact pooled target recall retention is at least 0.85 at every registered M level. The ratio is reconstructed from pooled counts rather than averaging per-seed ratios.

## Primary computation metric

states_scored_over_M is the mean number of state embeddings passed to the final cosine reranker divided by M. It is recorded independently from the final K=32 shortlist.

## Interpretation boundary

The experiment tests whether additional 3-factor beam can recover the capability lost at beam 3 while remaining below the approximately 15% candidate fraction of the validated two-factor (16,6) frontier.

It does not establish universal optimality, semantic memory capability, asymptotic sublinear scaling, or hardware speedup.

## Reproduction

Smoke: python scripts/run_c5_multifactor_beam_001.py --smoke

Full: python scripts/run_c5_multifactor_beam_001.py

Artifact: artifacts/TACOSM-C5-MULTIFACTOR-BEAM-001.json

Full-run marker: [run-c5-multifactor-beam-001-full]

CI trigger verification complete.
