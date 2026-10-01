# TACOSM-LEARN-REP-004 RESULT

## Run provenance

- workflow run: `36653857286`
- job: `109693780260`
- code head: `08204320a41df64f8dd66f7689fd0e4164090ab5`
- branch: `research/rep004-exploration-semantic-routing`
- full test suite: **826 passed**
- baseline current test count: **826**
- artifact: `TACOSM-LEARN-REP-004`
- artifact id: `11071552629`
- artifact zip SHA-256: `9ac0aa4427f227658ccc3b57098b3b83eb16200e92650f45b1a4592a74d1d020`

## Main result

| Seed | Baseline B | Exploration D | D − B | Exploration decisions | Training successes D |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3438 | 0.3984 | +0.0547 | 55 | 97 |
| 1 | 0.0938 | 0.0938 | +0.0000 | 60 | 51 |
| 2 | 0.0000 | 0.0000 | +0.0000 | 83 | 13 |
| 3 | 0.5352 | 0.5234 | −0.0117 | 74 | 186 |
| 4 | 0.0313 | 0.0313 | +0.0000 | 69 | 33 |
| **Pooled** | **0.2008 (257/1280)** | **0.2094 (268/1280)** | **+0.0086** | **341** | **380** |

The no-learning control remained at 0.0977 pooled Top-1. Chance is 0.1250.
The analytic witness remained 1.000 across all five seeds.

Mean target rank:

- baseline B: 4.378
- exploration D: 4.410

Mean hard-negative margin:

- baseline B: approximately −0.0039 when averaged across seeds;
- exploration D: approximately −0.0040 when averaged across seeds.

Exact edge execution remained 1.000 for every evaluated selected program.

## What the intervention changed

The exploration schedule increased stochastic training actions as intended.
However, this did not produce a consistent learning improvement.

Most notably, seed 2 improved from zero training successes to 13 and still had
zero held-out Top-1 recall. Seed 3 had 186 exploration-arm training successes
but held-out recall decreased slightly relative to the baseline. Therefore the
number of successful updates alone does not explain generalization quality.

The pooled held-out improvement is only 0.86 percentage points and is driven
primarily by seed 0. The intervention does not reproduce a robust gain across
seeds.

## Research conclusion

REP-004 does **not** support the hypothesis that deterministic action selection
or insufficient exploration is the main remaining bottleneck in the current
graph-program router.

The more defensible decomposition is now:

`explicit program topology: sufficient`
`semantic relation: exactly representable`
`learned exploitation: unstable`
`simple epsilon-greedy exploration: insufficient as the sole repair`

This pushes the research target toward the learning objective/representation
interface rather than another exploration schedule.

## Next architectural test

The next experiment should make persistent state part of the routing input while
keeping the explicit executable-program candidate boundary fixed.

The proposed loop is:

`world write -> temporal delay -> addressed state read -> semantic router -> exact execution -> verification`

The state item should store the semantic requirement, while the public query
contains only its address. The router must therefore combine persistent state
content with explicit candidate topology.

That experiment is important because it tests the distinctive PNDS claim more
directly than REP-002 through REP-004, while preserving the current diagnostic
separation between information availability, routing, execution, and learning.

## Non-claims

REP-004 does not establish persistent memory capability, sublinear retrieval,
selective hardware computation, general graph reasoning, PLM/PNDS validity, or
real-world capability.

REP-003, REP-002, REP-001, and C5 evidence remain frozen historical results.