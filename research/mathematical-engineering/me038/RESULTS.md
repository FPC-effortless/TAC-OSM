# ME038 — a new native state-transition variant

Native fast/slow dynamics now receive associative recall and perform prediction-error matrix writes. Historical cross-channel context is shared equally across neural arms. This is a new implemented TAC variant; associative/delta-rule primitives are established, so no first-in-literature novelty claim is made.

## Exposure-matched complete models

Two seeds, 500 joint updates after identical validation-selected 1,000-update AR parents. Five legacy sources are exposed development evidence. Tetouan is a separately reserved temporal test within one city, not zero-shot or multi-city generalization. Lower MSE is better.

| Model | Legacy mean MSE | Tetouan all 8 channels | Tetouan 3 power channels |
|---|---:|---:|---:|
| Innovation TAC | 0.340467 | 0.285395 | 0.153310 |
| Matched context | 0.343203 | 0.290836 | 0.156326 |
| Matched GRU | 0.341094 | 0.288834 | 0.156362 |
| AR adaptation | 0.348027 | 0.312920 | 0.202714 |

Development gain vs AR: 2.17%; vs context: 0.80%. Registered development target **FAIL**.

Prospective power gain vs AR: 24.37%; vs context: 1.93%. Registered primary power target **PASS**. H720 power regression vs AR: -12.85% (positive worse).

ME037 historical native macro MSE was 0.339944; new candidate relative change: 0.15% higher MSE. This historical comparison is unmatched for parameters, state capacity, cross-channel information paths and actual CPU; it is not architecture superiority.

## Separate training-CPU-budget lane

Five process-CPU-second optimization deadline, finish the crossing update, at most 10,000 updates. Sample exposure therefore differs. Validation/serialization and worker startup are measured outside this training deadline and included in reported component cost sums. Shared data loading and final trace/report writing are not exhaustively metered; these are not complete end-to-end totals. This is not equal end-to-end compute/data parity.

| Model | Tetouan power MSE (H96/720) | Mean updates, all compute cells | Mean training CPU seconds |
|---|---:|---:|---:|
| Innovation TAC | 0.151090 | 425.2 | 5.0054 |
| Matched context | 0.146461 | 2456.2 | 5.0015 |
| Matched GRU | 0.151773 | 1251.2 | 5.0015 |
| AR adaptation | 0.192950 | 5933.0 | 5.0004 |

Compute-lane power gain vs AR: 21.69%; vs context: -3.16%. Separate power target **FAIL**.

## All dataset/horizon cells

| Lane | Dataset | H | Native innovation | Context | GRU | AR |
|---|---|---:|---:|---:|---:|---:|
| exposure | ETTh1 | 96 | 0.392196 | 0.390423 | 0.386762 | 0.390489 |
| exposure | ETTh1 | 192 | 0.433585 | 0.436541 | 0.440263 | 0.440134 |
| exposure | ETTh1 | 336 | 0.471255 | 0.476757 | 0.472518 | 0.481811 |
| exposure | ETTh1 | 720 | 0.491594 | 0.485392 | 0.475108 | 0.479281 |
| exposure | ETTh2 | 96 | 0.282591 | 0.288896 | 0.289543 | 0.290959 |
| exposure | ETTh2 | 192 | 0.372238 | 0.378231 | 0.376884 | 0.377822 |
| exposure | ETTh2 | 336 | 0.417912 | 0.425255 | 0.422453 | 0.419856 |
| exposure | ETTh2 | 720 | 0.419188 | 0.418755 | 0.419218 | 0.418742 |
| exposure | ETTm1 | 96 | 0.345338 | 0.360014 | 0.344236 | 0.358350 |
| exposure | ETTm1 | 192 | 0.386524 | 0.389314 | 0.389285 | 0.396062 |
| exposure | ETTm1 | 336 | 0.419459 | 0.425561 | 0.422105 | 0.428305 |
| exposure | ETTm1 | 720 | 0.478974 | 0.474902 | 0.476842 | 0.490262 |
| exposure | weather | 96 | 0.188987 | 0.192606 | 0.191201 | 0.200401 |
| exposure | weather | 192 | 0.229820 | 0.234316 | 0.229919 | 0.244486 |
| exposure | weather | 336 | 0.283932 | 0.284822 | 0.284941 | 0.297051 |
| exposure | weather | 720 | 0.355688 | 0.356227 | 0.356314 | 0.366053 |
| exposure | electricity | 96 | 0.190431 | 0.193352 | 0.192518 | 0.202691 |
| exposure | electricity | 192 | 0.193606 | 0.194819 | 0.194661 | 0.202531 |
| exposure | electricity | 336 | 0.207858 | 0.208725 | 0.208635 | 0.216825 |
| exposure | electricity | 720 | 0.248157 | 0.249145 | 0.248471 | 0.258421 |
| exposure | tetouan | 96 | 0.221594 | 0.227100 | 0.223555 | 0.256867 |
| exposure | tetouan | 192 | 0.262068 | 0.270297 | 0.268794 | 0.294929 |
| exposure | tetouan | 336 | 0.310779 | 0.313680 | 0.313941 | 0.331824 |
| exposure | tetouan | 720 | 0.347141 | 0.352268 | 0.349044 | 0.368060 |
| compute | ETTh1 | 96 | 0.387719 | 0.396753 | 0.389929 | 0.387861 |
| compute | ETTh1 | 720 | 0.502478 | 0.490900 | 0.485405 | 0.478141 |
| compute | weather | 96 | 0.193074 | 0.178425 | 0.178320 | 0.199014 |
| compute | weather | 720 | 0.360453 | 0.351069 | 0.354184 | 0.365637 |
| compute | tetouan | 96 | 0.228566 | 0.209964 | 0.218996 | 0.251744 |
| compute | tetouan | 720 | 0.347190 | 0.335980 | 0.338714 | 0.365442 |

## Power-only outcomes

| Lane | H | Native | Context | GRU | AR |
|---|---:|---:|---:|---:|---:|
| exposure | 96 | 0.122714 | 0.125400 | 0.123322 | 0.196219 |
| exposure | 192 | 0.140310 | 0.147797 | 0.150425 | 0.205144 |
| exposure | 336 | 0.172135 | 0.170417 | 0.171079 | 0.205157 |
| exposure | 720 | 0.178082 | 0.181692 | 0.180620 | 0.204336 |
| compute | 96 | 0.128150 | 0.122676 | 0.132964 | 0.186627 |
| compute | 720 | 0.174029 | 0.170246 | 0.170583 | 0.199272 |

## Power MAE and channel trade-offs

The following zone MAEs are rescaled using each train-only channel std. Units remain the original publisher numeric units, which UCI does not specify; no Wh/kW conversion or measured savings is inferred.

| Lane | H | Model | Power standardized MAE | Zone1 MAE | Zone2 MAE | Zone3 MAE |
|---|---:|---|---:|---:|---:|---:|
| exposure | 96 | Innovation TAC | 0.241726 | 1526.108 | 1609.799 | 1274.611 |
| exposure | 96 | Matched context | 0.249754 | 1626.680 | 1615.971 | 1332.836 |
| exposure | 96 | Matched GRU | 0.245695 | 1578.015 | 1608.236 | 1307.336 |
| exposure | 96 | AR adaptation | 0.305405 | 2304.886 | 2062.524 | 1238.195 |
| exposure | 192 | Innovation TAC | 0.258570 | 1619.478 | 1759.076 | 1327.661 |
| exposure | 192 | Matched context | 0.262572 | 1642.572 | 1790.849 | 1344.163 |
| exposure | 192 | Matched GRU | 0.266904 | 1623.442 | 1807.853 | 1423.541 |
| exposure | 192 | AR adaptation | 0.306592 | 2279.407 | 2088.956 | 1250.185 |
| exposure | 336 | Innovation TAC | 0.290836 | 1837.061 | 1906.454 | 1571.543 |
| exposure | 336 | Matched context | 0.284349 | 1853.819 | 1903.281 | 1434.867 |
| exposure | 336 | Matched GRU | 0.284211 | 1820.919 | 1919.883 | 1440.314 |
| exposure | 336 | AR adaptation | 0.309039 | 2241.687 | 2139.960 | 1266.187 |
| exposure | 720 | Innovation TAC | 0.294949 | 1879.598 | 2042.517 | 1439.847 |
| exposure | 720 | Matched context | 0.299105 | 1945.826 | 2053.634 | 1447.261 |
| exposure | 720 | Matched GRU | 0.300880 | 1945.990 | 2052.436 | 1483.061 |
| exposure | 720 | AR adaptation | 0.312292 | 2218.110 | 2197.671 | 1276.659 |
| compute | 96 | Innovation TAC | 0.246229 | 1574.365 | 1643.947 | 1275.388 |
| compute | 96 | Matched context | 0.245449 | 1476.999 | 1595.723 | 1408.520 |
| compute | 96 | Matched GRU | 0.253700 | 1551.238 | 1687.199 | 1385.717 |
| compute | 96 | AR adaptation | 0.295252 | 2219.312 | 1994.715 | 1204.033 |
| compute | 720 | Innovation TAC | 0.295582 | 1916.317 | 2012.815 | 1457.291 |
| compute | 720 | Matched context | 0.297501 | 1750.532 | 1983.694 | 1679.360 |
| compute | 720 | Matched GRU | 0.293607 | 1779.335 | 2010.700 | 1543.733 |
| compute | 720 | AR adaptation | 0.308102 | 2181.900 | 2169.874 | 1263.129 |

## Evidence boundaries

All attempted fits, checkpoint selection, data hashes, parameter/exposure/cost contracts and per-channel scores are retained. No fit replacement or post-test modification. Parameter gaps are under 1% among neural arms; native recurrent state is 128 float32 values/channel, twice ME037. AR and persistence are unmatched practical references. Resource/selection costs and supported operator FLOP subtotals are in resources.json and comparison-manifest.json; unsupported work means no full FLOP/energy parity.

Neither lane passes all architecture-parity gates: exposure lane spends different CPU; CPU lane has different sample presentations and total end-to-end spend. **No architectural superiority claim.** Two fitted seeds, one new city and previously exposed legacy benchmarks limit generalization. Small differences are descriptive, not statistical superiority results.

Original-source units are preserved, with no invented power units, measured savings, actions or costs. The model forecasts all eight measurements; the primary power metric is the three actual zone channels. No future weather, calendar variables, synthetic fit evidence, reopened household/KuaiRec data, action/controller/VoI capability, commercial qualification or constant-time streaming claim. Full 96-step replay remains necessary. Prior Weather anomalies and ME035B precision failure remain intact.

Replay: 1,008 parent/adaptation endpoints, 37,310,976 saved validation values bit exact; 8,448,000 initial AR values exact. These arrays are not the complete final test matrix.

Independent final-metric reduction audit: 52 Tetouan models/references, 1,479,493,632 test forecast values recomputed; PASS. Uses the preregistered float64 accumulation bound; does not imply bit-exact storage/replay of every test array or a legacy full-matrix rerun.
