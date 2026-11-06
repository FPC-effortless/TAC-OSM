# E2E-005 result record

Experiment: `TACOSM-PLM-INTEGRATED-E2E-005`

Confirmatory workflow run: `37463899950` (run 50), event `push`

Measured checkout commit recorded by the runner:
`4137b3922aa0e34131c177751e48d5de67564573`

Artifact:
- GitHub artifact id: `11413733685`
- artifact SHA-256: `c8150cccd287962119594d94cc07eec52c26b547650e2dda3c72bba6898b1a90`
- contract SHA-256: `388d829745d445b6f75f206260c20401b098cd4e60d8165dfed1f5b78a8c82ba`
- benchmark generator SHA-256: `ca6147a3ad0cf503f5dadd51283482621e164080f2eb266777c4ae72748142c6`

## Integrity disposition

Smoke preflight: PASS.

Full measurement: PASS.

Independent result validation: PASS.

Oracle q2 accuracy: 1.000 across all seeds.

Training/evaluation semantic overlap: 0 for every seed.

Gradient surface: PASS.

The registered benchmark therefore produced an admissible measured result. The validator failure seen in the preceding run was a CI instrumentation defect and is not retained as scientific evidence.

## Primary result

Mean normal held-out q2 accuracy: **0.512**.

Seed values: **0.530, 0.4975, 0.4875, 0.520, 0.525**.

Minimum seed: **0.4875**.

Seed-level bootstrap 95% interval: **[0.496, 0.526]**.

Registered threshold: **0.80**.

Primary decision: **FAIL**.

The result licenses only the registered negative statement: this fixed functional E2E-005 configuration did not establish the synthetic multimodal capability criterion. It does not establish architecture impossibility.

## Causal-scope limitation

q1 targets `entities[0]` and q2 targets `entities[1]`. Consequently q1's post-action verifier/update cannot causally influence the state read for q2. E2E-005 is therefore a test of multimodal encoding, entity-addressed storage and later retrieval plus fixed CASM execution; it is not a test of feedback-conditioned persistence across q1 -> q2.

## Development consequence

The E2E-005 evaluation set is now **retired from future model selection or tuning** because its outcome is known. Any repaired model must be evaluated on a fresh, independently pre-registered holdout.

A training-only diagnostic of the same fixed model also remained near chance after the registered schedule, indicating that the failure is not explained solely by held-out-composition generalization. Development analysis isolated a differentiable Boolean-CASM gradient dead zone around uncertain latent bits. This finding is an engineering diagnosis, not an E2E-005 capability result.

E2E-006 is blocked until the functional multimodal learner can first pass a training-only functionality gate. A subsequent confirmatory lane will use a fresh holdout rather than reuse E2E-005's evaluation compositions.
