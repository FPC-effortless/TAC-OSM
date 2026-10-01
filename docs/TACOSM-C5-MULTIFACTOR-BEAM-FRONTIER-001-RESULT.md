# TACOSM-C5-MULTIFACTOR-BEAM-FRONTIER-001 RESULT

Status: MEASURED — VALID.

## Provenance

- GitHub Actions run: 36802320379
- measurement head: 1bc8a27b66bea8740b7db6e2ca7925a4816c8e54
- full artifact: TACOSM-C5-MULTIFACTOR-BEAM-FRONTIER-001-full
- artifact id: 11136178923
- artifact digest: sha256:6c5c3eba75383175497545b717854f8c3bd85d9d23b4f7f370aea8792a93842
- protocol: H=256, M={128,256,512}, K=32, factor_count=3, factor_size=8, beams={4,5,6}, seeds=10-19, 100 evaluation steps per seed/M/arm
- smoke and full measurements: passed
- state-distinct executor and exact pooled count conservation: passed

## Primary pooled result

| M | beam | exhaustive target recall | selective target recall | capability retention | proposal retention | states scored/M | mean states scored | mean query MACs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 0.386 | 0.329 | 0.8523 | 0.776 | 0.1396 | 17.872 | 573.95 |
| 128 | 5 | 0.386 | 0.360 | 0.9326 | 0.897 | 0.2506 | 32.076 | 801.22 |
| 128 | 6 | 0.386 | 0.372 | 0.9637 | 0.951 | 0.4195 | 53.694 | 1147.10 |
| 256 | 4 | 0.166 | 0.145 | 0.8735 | 0.749 | 0.1378 | 35.283 | 852.53 |
| 256 | 5 | 0.166 | 0.154 | 0.9277 | 0.872 | 0.2506 | 64.149 | 1314.38 |
| 256 | 6 | 0.166 | 0.161 | 0.9699 | 0.942 | 0.4209 | 107.762 | 2012.19 |
| 512 | 4 | 0.053 | 0.055 | 1.0000* | 0.755 | 0.1370 | 70.146 | 1410.34 |
| 512 | 5 | 0.053 | 0.051 | 0.9623 | 0.886 | 0.2514 | 128.713 | 2347.41 |
| 512 | 6 | 0.053 | 0.051 | 0.9623 | 0.946 | 0.4204 | 215.261 | 3732.18 |

*The raw pooled ratio is 55/53 = 1.0377; the runner records the retention as 1.0 for eligibility purposes. The small denominator makes this M=512 cell statistically fragile.

## Capability-floor decision

The pre-registered capability floor was 0.90 at every M.

- beam 4 is **not eligible**: retention is below 0.90 at M=128 and M=256.
- beam 5 is **eligible**: minimum retention = 0.9277; maximum states scored/M = 0.2514.
- beam 6 is **eligible**: minimum retention = 0.9623; maximum states scored/M = 0.4209.

Neither capability-eligible arm remains below the approximately 0.15 state-reranking fraction of the validated two-factor factor16/beam6 reference.

The lowest-compute arm, beam 4, remains near 13.7–14.0% of M but fails the 0.90 floor at the two lower populations. Raising the beam recovers capability only by approximately doubling or tripling the reranking fraction.

## Interpretation

The intervention therefore exposes a coupled capability/computation frontier rather than a resolution of the frontier.

For three-factor factor-size-8 addressing:

- beam 4 is the sparse regime, but its proposal boundary loses too much target coverage;
- beam 5 restores the registered capability level, but state reranking rises to about 25% of M;
- beam 6 improves coverage further while rising to about 42% of M and paying a large routing-combination cost.

This is negative evidence for the specific hypothesis that widening the beam on the factor-size-8 three-factor construction can recover the missing capability without returning toward exhaustive computation.

It does **not** refute product-key addressing generally or the broader C5 claim. It localizes the current three-factor geometry: the missing capability cannot be recovered by beam expansion alone at the desired compute fraction.

## Arithmetic

Mean query arithmetic rises monotonically with beam:

- beam 4: about 574, 853, and 1410 MACs at M=128, 256, and 512;
- beam 5: about 801, 1314, and 2347 MACs;
- beam 6: about 1147, 2012, and 3732 MACs.

Pair-generation work is exactly 64, 125, and 216 product cells for beams 4, 5, and 6 respectively.

These are arithmetic estimates, not hardware wall-clock measurements.

## Reference-capability limitation

The exhaustive reference itself degrades strongly with M on this workload:

- 0.386 at M=128;
- 0.166 at M=256;
- 0.053 at M=512.

Therefore capability retention is preservation relative to an increasingly weak reference, and the M=512 ratios are based on only 53 exhaustive target successes pooled over the ten seeds.

This does not invalidate the preregistered result, but it limits what can be inferred from the high-M retention ratios. Future C5 measurements should separately stabilize or characterize the exhaustive-reference capability before using retention as the sole capability endpoint for large-population claims.

## Scientific status

Supported on this workload:

- three-factor factor-size-8 addressing has a measurable beam-dependent capability/computation frontier;
- beam expansion increases proposal retention and target recall;
- the beam-4 regime can remain near the prior sparse fraction, but does not meet the 0.90 capability floor at all tested M;
- beams 5 and 6 meet the 0.90 floor but no longer preserve the prior approximately 15% computation fraction.

Not established:

- a sub-15% configuration satisfying the 0.90 floor;
- sublinear asymptotic scaling in M;
- universal semantic retrieval;
- hardware wall-clock speedup;
- the full end-to-end C5 theorem.

## Next research direction

The next intervention should change the **addressing geometry**, not simply widen the beam again.

The most direct registered test is three-factor factor-size-16 addressing with beams 4, 5, and 6, using the same H, M, K, seeds, teacher, training-only codebook fitting, state-distinct executor, and exact pooled aggregation.

That experiment can distinguish two possibilities:

1. finer cells improve coverage efficiency enough to cross the 0.90 floor below the prior ~15% reranking fraction; or
2. finer cells remain recall-limited even as the beam increases, localizing the bottleneck to factorized representation/admission rather than beam width.

