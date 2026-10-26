# ME109 — parent-preserving residual native TAC: completed real-data study

Full registered development gate **FAIL**; complete seed passes **0/5**. All originals preserved; no promotion, merge, gate relaxation or extra fits.

Five104663-parameter jointly trained models,4096updates each, native recurrence and six families unchanged. Initial parental forecast/state preservation is verified, but is not a post-training improvement guarantee.

## Original evidence and scope

UCI501 Beijing Multi-Site, CC-BY-4.0, original archive SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.2013–2014FIT only; exposed2015REPORT,143natural-gap and4288general queries,6h-ahead,256h history. Protected2016–2017 numerical rows remain closed. Primary gap queries require current PM2.5age8–255h and an available future target; this is conditional offline forecasting, not all-outage safety. Seeds share the same labels and do not multiply independent observations.

Source452a1a977275823d21b3f04265a26ff836518ce6; frozen prefit6cbcd6c3. All5 original model/optimizer/RNG/plan/receipts were Git-frozen65e2f81791f0af048cc70a720374d8a7b89e0b45 before REPORT. All10original fitted FP32/64 graph/state/45parameter/input-gradient and read-only optimizer-proposal audits passed and were committed c8aed5ec before REPORT. No audit optimizer step.

Pre-training setup failures are preserved in PREFIT_STATUS.json. After fitting, an initial checkpoint staging attempt was refused by sparse-checkout rules; the fitted auditor then refused unfrozen files. The sparse scope was expanded, and identical original bytes were committed before any REPORT scoring; no fit was retried or changed.

## Five-seed measurements

| Metric | ME103 parents | ME108 prior candidate | ME109 | Change vs ME103 |
| --- | ---: | ---: | ---: | ---: |
| Gap PM2.5 MAE ↓ | 37.524168 | 41.132003 | 44.437162 | +18.4228% |
| General PM2.5 MAE ↓ | 33.403120 | 31.572055 | 32.014751 | -4.1564% |
| Gap six-channel marginal NLL ↓ | 5.151004 | 5.138193 | 5.127208 | -0.4620% |
| General six-channel marginal NLL ↓ | 4.944393 | 4.889856 | 4.891830 | -1.0631% |
| Gap PM90 interval score ↓ | 379.623199 | 241.566815 | 222.612326 | -41.3597% |
| Gap PM90 mean width ↓ | 369.909332 | 229.904205 | 197.248291 | -46.6766% |
| Gap PM90 coverage | 0.956643 | 0.966434 | 0.937063 | -2.0468% |
| General PM90 coverage | 0.898647 | 0.886707 | 0.891931 | -0.7474% |

Coverage rows above are fractions; point/width/interval score use original concentration units. Historical references are unmatched: ME10398495parameters, ME108102713, ME109104663 and different additional fits/seeds/plans. These are developmental system results, not attribution or architectural superiority.

| Seed | Gap PM MAE | Max allowed | Gap PM90 coverage | Full gate |
| --- | ---: | ---: | ---: | --- |
| 109001 | 42.419655 | 36.353154 | 95.1049% | FAIL |
| 109002 | 44.094875 | 38.074168 | 93.7063% | FAIL |
| 109003 | 46.436878 | 35.684479 | 91.6084% | FAIL |
| 109004 | 44.726963 | 41.476278 | 94.4056% | FAIL |
| 109005 | 44.507439 | 39.849239 | 93.7063% | FAIL |

Gate pass counts: `{"coverage": 5, "daily_point": 5, "gap_nll": 5, "gap_point": 0, "interval_score": 5}`. All5 must pass all original gates. No averaging into an individual pass.

## All-channel calibration, not only the primary gate

| Pollutant | General 90% coverage | Gap 90% coverage |
| --- | ---: | ---: |
| PM2.5 | 89.1931% | 93.7063% |
| PM10 | 91.6266% | 85.8741% |
| SO2 | 93.0778% | 82.6950% |
| NO2 | 91.5307% | 75.3623% |
| CO | 89.9087% | 81.8571% |
| O3 | 88.5206% | 90.0000% |

Only gap PM2.5 has a registered coverage acceptance gate. The complete table must still be assessed before claiming useful multi-channel uncertainty. Neither a PM-only pass nor a positive scale floor proves all-channel calibration.

## Update and verification boundaries

Actual updates20480; block checks266240; maximum computed applied point dot0.0; projected blocks94810; actual parameter-rounding rejected blocks24443. New-update CPU seconds1396.580641, excluding historical parent training and preprocessing; profiler FLOPs are partial and exclude optimizer/unsupported operators.

The guard protects the current-minibatch six-channel average, not each held-out channel/missingness regime or finite-step population loss. Seven limited new Lean identities/role/bound/count contracts and independent executable equations pass. Full neural/AD/IEEE/censored-law source refinement, useful associative retrieval, transfer, intervention intelligence, control and product safety remain OPEN.

Independent clean restoration and final replay evidence are recorded separately in the original bundle when completed. No further ME109 fits or changed thresholds are permitted.
