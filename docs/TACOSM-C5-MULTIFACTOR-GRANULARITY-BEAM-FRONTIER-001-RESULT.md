# TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001 — RESULT

Status: MEASURED — VALID; no registered arm satisfies the frontier criterion.

## Provenance

- GitHub Actions run: `36806032589`
- measurement head: `9f7e1e46def57db7d6d1f491c3aba756f334ddfc`
- smoke job: success
- full job: success
- full artifact: `TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001-full`
- artifact id: `11137229446`
- artifact size: `5962` bytes
- artifact digest: `sha256:7c80367df42bd7191c54953d64328b62fb710d5d741ebf32b20703c1dac91ac0`
- artifact contents: `TACOSM-C5-MULTIFACTOR-GRANULARITY-BEAM-FRONTIER-001.json`
- artifact internal JSON size: 88,531 bytes
- protocol: H=256, M={128,256,512}, K=32, factor_count=3, factor_size=16, beams={4,5,6}, seeds=10-19, 100 evaluation steps

## Primary pooled result

| M | beam | exhaustive target recall | selective target recall | retention | proposal retention | states scored/M | mean states scored | mean query MACs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 0.386 | 0.268 | 0.6943 | 0.470 | 0.030016 | 3.842 | 477.47 |
| 128 | 5 | 0.386 | 0.292 | 0.7565 | 0.585 | 0.048992 | 6.271 | 516.34 |
| 128 | 6 | 0.386 | 0.330 | 0.8549 | 0.705 | 0.074187 | 9.496 | 567.94 |
| 256 | 4 | 0.166 | 0.141 | 0.8494 | 0.439 | 0.029156 | 7.464 | 535.42 |
| 256 | 5 | 0.166 | 0.152 | 0.9157 | 0.587 | 0.049277 | 12.615 | 617.84 |
| 256 | 6 | 0.166 | 0.144 | 0.8675 | 0.688 | 0.075082 | 19.221 | 723.54 |
| 512 | 4 | 0.053 | 0.046 | 0.8679 | 0.479 | 0.029461 | 15.084 | 657.34 |
| 512 | 5 | 0.053 | 0.046 | 0.8679 | 0.630 | 0.049658 | 25.425 | 822.80 |
| 512 | 6 | 0.053 | 0.051 | 0.9623 | 0.725 | 0.075338 | 38.573 | 1033.17 |

These pooled retention values are calculated directly from the artifact's exact
pooled counts, not from averaging the per-seed retention ratios:

- M=128: 386 exhaustive successes; selective successes = 268 / 292 / 330.
- M=256: 166 exhaustive successes; selective successes = 141 / 152 / 144.
- M=512: 53 exhaustive successes; selective successes = 46 / 46 / 51.

Each pooled cell contains 1,000 evaluation tasks (10 seeds × 100 steps). The
artifact contains 90 cell records in total (10 seeds × 3 M levels × 3 beams).

## Registered verdict

The preregistered capability floor is retention >= 0.90 at every M, and the
frontier improvement criterion additionally requires mean states_scored_over_M
strictly below 0.1542864583 across all three M levels.

| Beam | Minimum retention across M | Maximum states scored/M | Registered eligibility |
|---:|---:|---:|---|
| 4 | 0.6943 | 0.030016 | NOT ELIGIBLE |
| 5 | 0.7565 | 0.049658 | NOT ELIGIBLE |
| 6 | 0.8549 | 0.075338 | NOT ELIGIBLE |

No arm reaches the 0.90 capability floor at all three M levels. Therefore no
arm can be a registered frontier improvement, regardless of its sparse
computation fraction.

Failure localization is not a single M=128 cell. All three beams fail the
capability floor at M=128; beams 4 and 6 also fall below the floor at M=256.
Beam 5 clears the floor at M=256 only. The M=512 ratios are reference-relative
to an exhaustive baseline with only 53/1000 target successes and therefore are
not strong independent capability evidence.

## Sparsity result

The factor-size-16 three-factor construction sharply reduces the reranked
fraction relative to the preregistered two-factor reference of 0.1542864583:

- beam 4 maximum observed fraction: 0.030016 (~5.14× lower than reference fraction);
- beam 5 maximum observed fraction: 0.049658 (~3.11× lower);
- beam 6 maximum observed fraction: 0.075338 (~2.05× lower).

The exact fractions are essentially flat in M within each beam:

- beam 4: 0.030016, 0.029156, 0.029461;
- beam 5: 0.048992, 0.049277, 0.049658;
- beam 6: 0.074187, 0.075082, 0.075338.

This supports the fixed-geometry scaling relationship that reranked state count
remains approximately proportional to M while factor size changes the constant.
It does not establish sublinear reranking.

## Arithmetic and routing-combination cost

Pair-generation operations are exactly 64, 125, and 216 for beams 4, 5, and 6.
Mean query MACs rise with beam at every M. These are arithmetic accounting
values, not hardware wall-clock measurements.

The factor-size-16 intervention therefore creates a genuine geometry tradeoff:
the state-reranking fraction becomes substantially smaller, but capability does
not reach the registered floor at the tested lower populations.

## Exhaustive-reference limitation

The exhaustive comparator reproduces the prior degradation:

- M=128: 0.386
- M=256: 0.166
- M=512: 0.053

At M=512, only 53 of 1,000 pooled evaluation tasks are exhaustive successes.
Retention values above 1.0 are therefore possible at the seed/cell level when
selective success exceeds this small denominator, and they should be treated as
reference-relative diagnostics rather than absolute capability measurements.

This limitation does not change the registered verdict: the arms already fail
the capability floor at lower M values. It does, however, constrain interpretation
of the high-M retention ratios and motivates the separately preregistered
exhaustive-reference calibration study (PR #46 / Issue #45).

## Pre-run prediction

The prediction recorded before dispatch was that retention at M=128 and M=256
would depend primarily on B/F rather than F, with all F=16 beams remaining below
0.90 at both levels.

The artifact falsifies the universal form of that prediction only if an F=16 arm
reaches >=0.90 at both M=128 and M=256. No arm does so: beam 5 reaches 0.9157 at
M=256 but only 0.7565 at M=128; beams 4 and 6 are below 0.90 at both levels.
Thus the preregistered B/F-only prediction is not falsified by these data.

## Scientific interpretation

The measured result is negative for the specific registered objective:
factor-size-16 three-factor geometry reduces computation substantially, but the
same geometry does not recover >=0.90 retention at all registered M values.

The result narrows the mechanism question. Factor size changes the compute
constant strongly, but lower-population capability remains admission-limited.
Increasing beam from 4 to 6 improves proposal coverage at M=128, but it still
does not reach the registered floor; at M=256, the beam relationship is not
monotonic because beam 5 exceeds the floor while beam 6 falls below it.

This non-monotonicity means retention cannot be reduced to a simple monotone
function of B/F on this workload. The B/F hypothesis remains a useful geometric
descriptor, not an established sufficient statistic for capability.

## Non-claims

This measurement does not establish:

- sublinear or constant-state asymptotic reranking;
- semantic retrieval guarantees;
- universal optimal beam or factorization depth;
- hardware speedup;
- general C5 support on origin/master;
- the validity of the successor architecture;
- PLM validity.

C5 remains UNTESTED on origin/master.

## Next experiments

1. Preserve this measurement unchanged as the negative factor-size-16 result.
2. Use PR #46 to determine whether exhaustive-reference degradation is a teacher
   distribution/calibration problem.
3. Use PR #48 to test factorization depth at fixed B/F=0.625 and determine whether
   the geometric reduction survives deeper factorization without unacceptable
   capability loss.
4. Treat routing combinatorics separately from state-reranking cost; a depth
   increase can reduce scored-state fraction while increasing product-cell
   enumeration exponentially in d.
