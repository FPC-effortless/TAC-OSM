# ME043 / ME043B: complete compute-adaptive architecture results

Original authentic-data training; historical development only. No newly sealed test, merge or promotion.

A: 96 complete downstream fits. B: 64 separately registered follow-up fits after A failure. Eight original train-only AR parents are reused by B, not refitted.

A compresses context and trains both exits densely. B retains the full original contextual predictor and trains native memory/router on every eighth update. Both perform real hard conditional native dispatch at inference. These are new trained architectures, not equivalent execution changes.

## ME043

### etth1: oil temperature standardized historical DEV MSE

| Complete model | 250 updates | 5 optimizer CPU seconds | Native fraction, updates / CPU |
|---|---:|---:|---:|
| adaptive_001 | 0.146524 | 0.150863 | 0.000 / 0.000 |
| adaptive_005 | 0.145858 | 0.149310 | 0.000 / 0.005 |
| adaptive_free | 0.146518 | 0.150842 | 0.303 / 0.423 |
| context_cross | 0.145864 | 0.155283 | n/a |
| gru_cross | 0.146836 | 0.154989 | n/a |
| innovation_tac | 0.147081 | 0.148738 | n/a |

Locked per-candidate gates:

| Model | >=1% accuracy gain | Horizon safety | Equal-CPU accuracy | All gates |
|---|---|---|---|---|
| adaptive_001 | FAIL | PASS | FAIL | FAIL |
| adaptive_005 | FAIL | PASS | FAIL | FAIL |
| adaptive_free | FAIL | PASS | FAIL | FAIL |

### tetouan: three-zone power standardized historical DEV MSE

| Complete model | 250 updates | 5 optimizer CPU seconds | Native fraction, updates / CPU |
|---|---:|---:|---:|
| adaptive_001 | 0.125667 | 0.118144 | 0.000 / 0.002 |
| adaptive_005 | 0.128267 | 0.118339 | 0.000 / 0.000 |
| adaptive_free | 0.128226 | 0.114563 | 0.615 / 0.693 |
| context_cross | 0.124132 | 0.103114 | n/a |
| gru_cross | 0.123862 | 0.108097 | n/a |
| innovation_tac | 0.118766 | 0.120957 | n/a |

Locked per-candidate gates:

| Model | >=1% accuracy gain | Horizon safety | Equal-CPU accuracy | All gates |
|---|---|---|---|---|
| adaptive_001 | FAIL | FAIL | FAIL | FAIL |
| adaptive_005 | FAIL | FAIL | FAIL | FAIL |
| adaptive_free | FAIL | FAIL | FAIL | FAIL |

Replay: **96 original model checkpoints, 21,639,168 forecast values bit exact**, including authentic truth and routing-array verification. Independent saved-array float64 metric recomputation: 96 models, maximum float64 reduction discrepancy 8.570921750106208e-14 (checked against an operation-count-derived float64 bound). No new fits or test rows in either audit.

## ME043B

### etth1: oil temperature standardized historical DEV MSE

| Complete model | 250 updates | 5 optimizer CPU seconds | Native fraction, updates / CPU |
|---|---:|---:|---:|
| scheduled_001 | 0.145632 | 0.153605 | 0.000 / 0.002 |
| scheduled_free | 0.146402 | 0.152425 | 0.587 / 0.246 |
| wide_context | 0.146531 | 0.155231 | n/a |
| wide_native | 0.146965 | 0.149771 | n/a |

Locked per-candidate gates:

| Model | >=1% accuracy gain | Horizon safety | Equal-CPU accuracy | All gates |
|---|---|---|---|---|
| scheduled_001 | FAIL | PASS | FAIL | FAIL |
| scheduled_free | FAIL | PASS | FAIL | FAIL |

### tetouan: three-zone power standardized historical DEV MSE

| Complete model | 250 updates | 5 optimizer CPU seconds | Native fraction, updates / CPU |
|---|---:|---:|---:|
| scheduled_001 | 0.129006 | 0.106698 | 0.000 / 0.000 |
| scheduled_free | 0.130350 | 0.104769 | 0.402 / 0.474 |
| wide_context | 0.126932 | 0.105386 | n/a |
| wide_native | 0.117621 | 0.120993 | n/a |

Locked per-candidate gates:

| Model | >=1% accuracy gain | Horizon safety | Equal-CPU accuracy | All gates |
|---|---|---|---|---|
| scheduled_001 | FAIL | FAIL | FAIL | FAIL |
| scheduled_free | FAIL | FAIL | PASS | FAIL |

Replay: **64 original model checkpoints, 14,426,112 forecast values bit exact**, including authentic truth and routing-array verification. Independent saved-array float64 metric recomputation: 64 models, maximum float64 reduction discrepancy 8.748557434046234e-14 (checked against an operation-count-derived float64 bound). No new fits or test rows in either audit.

## Measured batch1 complete forecast CPU latency

Each datum includes96-observation normalization, full-history AR, context encoder, router/dispatch and native/readout work where actually selected. Five warmups then5x20calls, single Torch CPU thread. Table geometric-means per-checkpoint medians across both horizons/seeds. Not microsecond streaming, training speed or persistent state reuse.

| Study / domain / lane | Model | Full forecast CPU ms |
|---|---|---:|
| A / tetouan / exposure | adaptive_001 | 0.233 |
| A / tetouan / exposure | adaptive_005 | 0.233 |
| A / tetouan / exposure | adaptive_free | 0.813 |
| A / tetouan / exposure | context_cross | 0.215 |
| A / tetouan / exposure | gru_cross | 0.483 |
| A / tetouan / exposure | innovation_tac | 1.954 |
| A / tetouan / cpu | adaptive_001 | 0.236 |
| A / tetouan / cpu | adaptive_005 | 0.231 |
| A / tetouan / cpu | adaptive_free | 0.845 |
| A / tetouan / cpu | context_cross | 0.218 |
| A / tetouan / cpu | gru_cross | 0.486 |
| A / tetouan / cpu | innovation_tac | 1.966 |
| A / etth1 / exposure | adaptive_001 | 0.236 |
| A / etth1 / exposure | adaptive_005 | 0.239 |
| A / etth1 / exposure | adaptive_free | 0.353 |
| A / etth1 / exposure | context_cross | 0.219 |
| A / etth1 / exposure | gru_cross | 0.474 |
| A / etth1 / exposure | innovation_tac | 2.031 |
| A / etth1 / cpu | adaptive_001 | 0.237 |
| A / etth1 / cpu | adaptive_005 | 0.239 |
| A / etth1 / cpu | adaptive_free | 0.768 |
| A / etth1 / cpu | context_cross | 0.221 |
| A / etth1 / cpu | gru_cross | 0.474 |
| A / etth1 / cpu | innovation_tac | 1.975 |
| B / tetouan / exposure | scheduled_001 | 0.235 |
| B / tetouan / exposure | scheduled_free | 0.544 |
| B / tetouan / exposure | wide_context | 0.211 |
| B / tetouan / exposure | wide_native | 1.910 |
| B / tetouan / cpu | scheduled_001 | 0.233 |
| B / tetouan / cpu | scheduled_free | 1.263 |
| B / tetouan / cpu | wide_context | 0.211 |
| B / tetouan / cpu | wide_native | 1.921 |
| B / etth1 / exposure | scheduled_001 | 0.236 |
| B / etth1 / exposure | scheduled_free | 0.818 |
| B / etth1 / exposure | wide_context | 0.217 |
| B / etth1 / exposure | wide_native | 1.956 |
| B / etth1 / cpu | scheduled_001 | 0.236 |
| B / etth1 / cpu | scheduled_free | 0.798 |
| B / etth1 / cpu | wide_context | 0.219 |
| B / etth1 / cpu | wide_native | 1.924 |

The registered secondary speed/accuracy checks are retained in [secondary-latency-gates.json](secondary-latency-gates.json). Faster cheap-only routes are not counted as accuracy wins.


## Comparison contract and remaining limits

- All active parameter counts within each registered arm matrix differ by <1%; counts and every original model are retained. B has more capacity than A, so B-versus-A improvements are unmatched engineering observations.
- Exposure lanes match250 batch4 observed training windows and source-index stream per seed. Actual CPU differs. CPU lanes match5 optimizer-CPU-second budgets with final-step overshoot; actual examples differ. B additionally varies native expert-update exposure by design. Full three-way parity FAIL: **NO ARCHITECTURAL SUPERIORITY CLAIM**.
- All evaluated endpoints fixed before execution; three A and two B cost candidates are registered. B is explicitly post-A development, with repeated historical-DEV access, not an independent confirmation. No validation-selected final-test opening.
- Per-fit process CPU and process high-water RSS are recorded. Per-fit elapsed wall time was omitted by the original runner despite the A protocol; it is not fabricated. High-water RSS is not isolated model activation memory. No FLOP/energy parity established.
- A route collapsing to cheap-only is preserved. Forecast gains from the complete system cannot be attributed to native memory without later mechanism studies. Routing probability is not a physical safety/decision policy.
- The first independent metric audit demanded bitwise float64 reduction equality and failed; those logs remain. Original strided forecast arrays and reloaded contiguous arrays can reduce in different orders. The corrected metric check uses a float64 operation-count gamma bound, derived from epsilon and target count rather than the observed discrepancy. Original float32 forecasts and router arrays still replay bit exactly; no forecast tolerance changed.
- Old ME035B strict ridge-equivalence tolerance failure remains unchanged. Current local exact checkpoint replay does not guarantee identical predictions on every CPU/library combination; separate CI replay reports its actual result.
- Default runtime and earlier model choices are unchanged. No action benefit, counterfactual qualification, adaptive System2, arbitrary-lifetime streaming or commercial readiness is established.
