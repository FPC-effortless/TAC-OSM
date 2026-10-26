# ME110 — point forecasting gains, complete qualification failed

Five original104663parameter models completed4096updates each. Native ME109 observation-to-output graph, fixed primary152FP32scalar state, and all original parent tensors were retained. Coordinate-typed learning and source-diverse sampling were tested as one integrated candidate; no component-attribution or matched-architecture superiority claim.

## Original exposed-2015 development results

Authentic UCI501 six-hour forecasts:143natural-gap and4288general queries. Previously exposed cohorts, potentially overlapping origins, same cases across seeds; not independent/site qualification. Protected2016–2017numeric rows unopened.

| Metric | ME103 parents | ME109 historical candidate | ME110 |
|---|---:|---:|---:|
| Gap PM2.5 MAE ↓ |37.524168|44.437162|34.815926|
| General PM2.5 MAE ↓ |33.403120|32.014751|30.972970|
| Gap six-channel raw-density NLL ↓ |5.151004|5.127208|5.075791|
| General six-channel raw-density NLL ↓ |4.944393|4.891830|4.865349|
| Gap PM90 interval score ↓ |379.623199|222.612326|401.850403|
| Gap PM90 interval width ↓ |369.909332|197.248291|396.649622|
| Gap PM90 coverage |95.664336%|93.706294%|96.643357%|

ME110mean point MAE improves7.2173%gap and7.2752%general versus its own historical parents. All five seeds improve both point scores versus their mapped parent. New4096updates, changed objectives/sampling and seed initialization mean this is system development, not equal-resource architectural attribution.

Practical references on these same frozen queries: gapLOCF56.076923/ridge43.915070; generalLOCF34.515625/ridge32.767415. All five ME110point scores beat both practical references. They are not parameter-/training-compute-matched neural competitors.

## Unchanged registered gates

**Complete gate FAIL0/5. No promotion or additional ME110fits.**

| Gate | Passing seeds |
|---|---:|
|coverage|4/5|
|gap_point|4/5|
|daily_point|5/5|
|gap_nll|5/5|
|interval_score|0/5|

| Seed | Gap PMMAE | General PMMAE | Gap PM90 coverage | Gap interval score |
|---|---:|---:|---:|---:|
|110001|31.662191|30.825500|95.804196%|399.699860|
|110002|35.890079|31.136215|96.503497%|417.777069|
|110003|35.709831|30.562202|98.601399%|348.924988|
|110004|36.726639|31.809498|95.104895%|443.006439|
|110005|34.090889|30.531433|97.202797%|399.843658|

Seed110003gapMAE35.709831 exceeds its unchanged35.684479limit; coverage98.601399%exceeds98%maximum. Neither is rounded into a pass. All five fail the interval-score improvement requirement. Mean coverage alone cannot pass a per-seed gate.

Gap coverage by channel PM2.5/PM10/SO2/NO2/CO/O3:96.6434/88.6713/83.6879/75.0725/82.1429/90.8571%. General:88.5261/91.5519/92.3767/91.1404/89.0335/86.9663%. PM-only thresholds do not qualify all-channel uncertainty.

## Was training insufficient?

Fixed internal FIT monitor:64gap+256general cases, excluded from new target draws but ancestor/stats-exposed; histories overlap. These are monitoring curves, not independent generalization evidence. No checkpoint selection or early stopping.

| New updates | Gap PMMAE ↓ | General PMMAE ↓ | Gap PM interval score ↓ |
|---:|---:|---:|---:|
|0|28.717729|35.599698|970.421619|
|1024|24.888126|30.866501|964.130640|
|2048|24.825922|30.659743|1002.976160|
|4096|24.145534|30.585923|1041.280994|

At4096, gap point error is15.9212%lower and general14.0838%lower versus initialization (5/5seeds each). From2048to4096, gap improves2.7406%, general0.2408% (4/5seeds each). Gap interval score worsens3.8191%in the latter half. Additional updates still help gap point accuracy modestly, while ordinary point learning is nearly flat and gap uncertainty gets worse. More training alone is not supported as a remedy for complete qualification. Longer-run convergence remains unmeasured.

## Mathematical engineering and actual learning checks

100253point/state coordinates receive only availability-balanced point gradients;4410uncertainty coordinates receive only likelihood/interval/reconstruction gradients. Actual cross-role forbidden coordinates are checked every batch. Independent role clipping prevents auxiliary magnitude from controlling point clipping. It is a custom routed update field, not the gradient of one scalar total objective.

266240actual13-block conditional point certificates pass;0fallbacks, maximum computed upper dot0. These are local first-order conditional IEEE-envelope checks, not finite-loss/generalization guarantees. Uncertainty changes leave source medians unchanged at fixed point coordinates, but point learning still changes uncertainty features/scales. This one-way coupling and the semantic distinction between current-state belief and future correction remain architectural issues; no causal explanation is inferred from these results alone.

PNDS PR#35 owns source-pinned executable Python and limited Lean binary-mask/conditional-congruence contracts. CI40374187754passed3source parity tests, Lean build and axiom audit. Full neural/AD/IEEE refinement and calibration proofs remainOPEN. All17legacy mappings remain intact; six active families unchanged.

## Exposure, originals and replay

New actual updates20480; examples327680; each model65536draws/7163648history exposures. Sampler2natural-gap+14general,57344distinct general update origins/model,307eligible natural-gap update origins after64monitor exclusions. Ancestor training is additional; no claim of fully matched historical pipeline compute.

New fit-pipeline CPU1306.635196s; monitor CPU9.926047s. Partial profiled training FLOPs19793348616960; excludes optimizer/unsupported operations, so not complete FLOP certification or CPU-parity claim.

Original source/registration/preflight freeze449a3677. All fifteen trained stage checkpoints, optimizer/RNG, five initialization monitor views and receipts frozen in6389521before REPORT; original monitor+fitted-source audit1a078eb before REPORT. Original scores/arrays/postfit/summary frozen c8e25cc. No retraining, extra fit or threshold change.

Original FIT monitor replay:115200saved quantile scalars across20views; ten fitted FP32/FP64source-reference forward/backward checks pass. Full local prefit suite1171testsPASS. Original score, summary and independent detached restoration checks are recorded separately when completed. Scoped anti-cheat checks PASS, full integrity INCONCLUSIVE. No autonomous/control/customer or useful-associative-memory claim.

Operational incidents are retained in EXECUTION_STATUS.json. They include the pre-training workspace reset, sparse-checkout omissions, Lean setup/download failures, corrected PNDS ancestry, and an explicitly post-start supplemental status receipt. Mandatory training source/registration and five actual preflight checks were frozen before updates; the supplemental receipt is not retrospectively claimed as preflight.

## Next substantive capability direction

Retain the improved native point predictor. Next architecture work should distinguish current-state belief from future residual prediction and engineer conditional uncertainty scale/tails without letting uncertainty objectives reshape point coordinates. Preserve source-aware uncertainty safeguards and test proper interval scores/calibration together. The present experiment does not prove which shared feature, floor or distribution mechanism caused the sharpness failure. Do not spend another ME110budget merely repeating updates.
