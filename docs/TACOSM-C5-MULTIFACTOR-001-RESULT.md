# TACOSM-C5-MULTIFACTOR-001 RESULT

Status: MEASURED.

## Provenance

- GitHub Actions run: 36795458948
- final measurement commit: 06eaf82654dd1940f07eb793d31818281a423f55
- full artifact id: 11133268415
- artifact digest: sha256:c232bba833c4e99175b4c7a2cfc4c30366e3d5ab5446e843b1d06929c306b8ce
- protocol: H=256, M={128,256,512}, K=32, seeds 10-19, 100 evaluation steps per seed/M/arm
- repository preconditions and final smoke: passed

## Artifact integrity

An earlier successful full run (36794683147) used the generic EndToEndExecutor and is invalid for downstream capability claims. It is superseded. This result uses the corrected self-contained StateDistinctEndToEndExecutor and passed the executor-distinctness smoke audit.

## Exact pooled retrieval result

The capability ratios below are reconstructed from exact pooled counts across the ten seed/M cells; they are not averages of per-seed ratios.

| M | Arm | Exhaustive target successes | Selective target successes | Capability retention | Proposal retention | Mean states scored | States scored/M | Mean MACs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | two_factor_16_beam6 | 386 | 336 | 0.870466 | 0.821 | 19.765 | 0.154414 | 732.24 |
| 128 | three_factor_8_beam3 | 386 | 297 | 0.769430 | 0.566 | 8.341 | 0.065164 | 421.456 |
| 128 | three_factor_16_beam3 | 386 | 185 | 0.479275 | 0.315 | 2.126 | 0.016609 | 450.016 |
| 256 | two_factor_16_beam6 | 166 | 152 | 0.915663 | 0.832 | 39.091 | 0.152699 | 1041.456 |
| 256 | three_factor_8_beam3 | 166 | 135 | 0.813253 | 0.559 | 16.872 | 0.065906 | 557.952 |
| 256 | three_factor_16_beam3 | 166 | 126 | 0.759036 | 0.297 | 4.061 | 0.015863 | 480.976 |
| 512 | two_factor_16_beam6 | 53 | 43 | 0.811321 | 0.828 | 79.742 | 0.155746 | 1691.872 |
| 512 | three_factor_8_beam3 | 53 | 52 | 0.981132 | 0.574 | 33.076 | 0.064602 | 817.216 |
| 512 | three_factor_16_beam3 | 53 | 40 | 0.754717 | 0.311 | 8.041 | 0.015705 | 544.656 |

## Primary finding

Three-factor addressing materially reduces candidate-state computation.
At M=128/256/512, three_factor_8_beam3 scores 6.52%/6.59%/6.46% of the state population, versus 15.44%/15.27%/15.57% for two_factor_16_beam6.
three_factor_16_beam3 reduces this further to about 1.66%/1.59%/1.57%, but with much larger retrieval loss.

## Capability tradeoff

three_factor_8_beam3 retains 0.7694, 0.8133, and 0.9811 of exhaustive target recall at M=128,256,512. Relative to the two-factor reference, this trades about 10 percentage points of retention at M=128 and M=256 for roughly 58% lower reranking fraction.
three_factor_16_beam3 is substantially sparser but retains only 0.4793, 0.7590, and 0.7547 at the three M levels.
The M=512 ratios are statistically fragile because the exhaustive reference has only 53 pooled successful target events.

## Arithmetic

At M=512 the mean query cost is 1691.872 MACs for the two-factor reference, 817.216 for three_factor_8_beam3, and 544.656 for three_factor_16_beam3.
These are registered arithmetic estimates, not hardware wall-clock measurements.

## Scientific status

Supported on this registered synthetic workload:
- factorization depth is a real control on product-cell occupancy;
- a 3-factor product-key can reduce state reranking by more than 2x relative to the validated 2-factor frontier;
- the 3-factor configurations expose a stronger sparsity/capability frontier.

Not established:
- a 3-factor configuration that preserves the full two-factor capability level;
- an acceptable universal capability floor;
- absolute O(1) candidate computation;
- asymptotic sublinear scaling;
- semantic/general memory retrieval;
- hardware speedup.

## Next experiment

The useful 3-factor operating point is factor_size=8, beam=3. Before adding another factorization dimension, sweep beam={4,5,6} at factor_count=3 and factor_size=8 using the same ten seeds and M levels. The purpose is to determine whether the lost capability can be recovered while remaining materially below the ~15% candidate fraction of the two-factor frontier.
