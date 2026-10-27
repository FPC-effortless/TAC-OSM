# LATFUSE019 — exact original real-source 64h NLL attribution: rare physical RAIN dominates tails; calibration still fails broadly

**2026-10-27. No original model optimizer, fitted calibration or source-tuned likelihood. Original LATFUSE018 4 trained 99,390-param model checkpoints SHA/optimizer/source replay preserved unchanged; genuine observed UCI501 original physical source and existing exposed 117 May–June2014 developmental station-hour cases only.** [Original real-source frozen model CI #40468698544 PASS](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40468698544). Complete every seed/channel/station observed residual and top10 original real outlier records in \`RESULTS.json\`, prospective study protocol \`REGISTERED_UNCERTAINTY_AUDIT.json\`. 2015 previously examined and protected2016–2017 source numerics unopened.

## Real physically measured source level

Each original pretrained source checkpoint queried on same original exact May–June 117 station-hour source episode IDs (SHA-checked original PLAN.json), 24 prior+64 observed hours, source-known future calendar6 predicting real all13 standardized source measurements at +6 recorded hours. True observation labels consulted only AFTER prediction. All **1,494 actually observed channel × site × timestamp original sensor values** were retained: no outlier-filtered refit or replacement of original masks/normalizer.

| Frozen real-source trained original model | Original *full* true-source Gaussian NLL ↓ | Diagnostic mean NLL ignoring the *worst 1%* (INVALID as primary metric) | Top10 positive NLL fraction | Actual nominal90% coverage |
|---|---:|---:|---:|---:|
| Associative seed18018 | 21.075 | 1.124 | 92.34% | 78.38% |
| Fastslow seed18018 | 48.358 | 1.018 | 96.68% | 81.53% |
| Associative seed18019 | 36.081 | 1.297 | 94.39% | 73.49% |
| Fastslow seed18019 | 6.165 | 1.155 | 77.31% | 79.92% |

The extreme original NLL is strongly **concentrated in the largest 10 recorded channel-source/time contributions**, not uniformly huge everywhere. These 10 observations account for 77–97% of *positive component NLL* (not fraction of mathematically summed total because some Gaussian NLL contributions can be negative). Dropping them is FOR A DIAGNOSTIC ONLY. The full original untrimmed NLL and poor coverage remain genuine failures and unchanged. NLL tail limitation cannot be removed by reporting a truncated figure as production calibration.

## Repeated physical rainfall source at one actual timestamp

Across all 4 original trained checkpoint forecasts, highest contributor: **RAIN channel of Gucheng station, real observed May31 2014 22:00 original UTC datetime**, followed by RAIN channel at Changping same recorded hour. Source is original authenticated UCI501 station zip. Severe original learned Gaussian scale underestimate for rare precipitation spikes:

| Frozen real source model | Gucheng RAIN predicted standardized sd | Gucheng RAIN true residual ÷ sd | Its single source Gaussian NLL |
|---|---:|---:|---:|
| Assoc18018 | 0.05887 | 120.72 σ | 7,284.3 |
| Fastslow18018 | 0.03401 | 207.56 σ | 21,538.3 |
| Assoc18019 | 0.04454 | 158.75 σ | 12,598.5 |
| Fastslow18019 | 0.12275 | 57.30 σ | 1,640.3 |

The independent Changping RAIN observation at same timestamp has residual/sd103.2,151.2,134.8,44.5 across the four models. Other large original real-site rainfall residuals preserved in per-checkpoint top10 JSON. These are real observed environmental tails; missingness and source data correctness was checked by input source authentic SHA, but we have NOT independently reconciled with a meteorological record/source engineering artifact. No artificial labels, no protected rows.

**Interpretation:** original learned Gaussian predictive rainfall distribution lacks appropriate zero-versus-positive storm-event/tail modeling; a narrow σ is disastrous for a genuinely positive spike. A plausible *later separately controlled* candidate is a physically coherent two-part, zero-inflated precipitation law (probability of no rain + positive rain conditional distribution such as Gamma/lognormal) with real source mask treatment and proper log scoring, explicitly fixed parameters/live compute budgets. This is a CONJECTURE until actual prospective source training/evaluation and calibration. Do not retrofit mixture on previously exposed May2014 event then claim independent generalization.

## Other channel failure, independent of rainfall tails

Even when rainfall is largely responsible for huge NLL, coverage outside rain remains poor. Source-channel nominal90% coverage:
- Original assoc seed18018 **SO2 59.1%** on n110 original real targets and original **WSPM 65.0%** n117; rainfall's own nominal coverage 88.9%.
- Original assoc seed18019 **SO2 29.1%** n110, WSPM 83.8%, rainfall 88.0%.
- Original fastslow seed18018 **WSPM 65.8%**, rainfall81.2%.
- Original fastslow seed18019 **SO2 46.4%** n110, rainfall89.7%.
This disproves a simplistic conclusion that rainfall-tail remediation alone will qualify the whole original forecast distribution: pollutant and wind channel confidence also needs real-data nonstationarity/source-specific calibration. All13 channel and12 source station actual denominator counts,NLL,MAE,scale/coverage statistics preserved in RESULTS.json.

## Proper separation of future research

1. **Physical uncertainty distribution track:** independently preregister source-real zero-positive rain law vs same-param/compute Gaussian baseline, source/station masks, positive event prevalence, observed source calendar, proper log score/NLL, source-tail calibration/PIT and CRPS/quantile scoring where appropriate. Draw independent real future source with unexposed storms; don't use May31 source event for hyperparameter tuning and subsequent qualification.
2. **Physical continuous uncertainty track:** independent source-specific time-split calibration and proper multichannel coverage for SO2, wind and source heterogeneous pollutant tails; either a coherent distributional model or conformal procedures with correct exchangeability limits, no posthoc cross-split leaks. Must preserve honest aggregate NLL including every outlier.
3. **Core content-memory track remains separate:** LATFUSE018 real source H64 associative historical PM2.5 trained content shows source seed variability and failure of incremental useful memory over whole-read-zero/past source prior in one seed. Do NOT blend calibration distribution fixes with memory read/write supervision into a single uncontrolled arm or call calibrated uncertainty a memory achievement.
4. Strict original research constraints: original target 95k–105k parameters, each architecture within1% live counts and complete training-math compute, source-true independently authorized train/dev/test temporal splits, actual action/outcome streams for commercial control claims, original 6,800-byte recurrence and complete train optimizer/RNG/source/checkpoint SHA independent replay. Lean finite-precision/AD/probability-law formal proof still OPEN.
5. **Do not train any new model on already exposed May–June2014 source to claim qualified scientific generalization.** All frozen original model weights and original numeric protected source unchanged, CI source-replay complete.

**Decision:** real H64 extreme NLL has a localized and physically plausible rainfall-tail mechanism, while broad multichannel undercoverage remains. This is a *failed calibration gate*, not evidence of robust future physical intelligence or autonomous decision-making.
