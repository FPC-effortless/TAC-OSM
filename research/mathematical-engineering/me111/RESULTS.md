# ME111 — conditional belief learning: real sharpness gains, gap calibration failure

Five original104663parameter native models completed1536additional updates each. All initial parent weights retained fromME110. New source replaces the unsupported pooled-difference width floor and admits genuinely hidden-current mean supervision on696existing coordinates. Same native fast/slow/8x8associative scan and152FP32primary recurrent payload. No Transformer/SSM conversion, extra model family, autonomous policy or deployment.

## Original exposed2015 development results

Authentic UCI501;143natural-gap and4288general queries, six pollutants,6hour target,256hour history. Same query population across all five seeds; cohorts can overlap and are not5times as many independent outcomes. Neither cohort is fresh. Original2016–2017numerical rows remain closed.

| Metric | ME110 parent mean | ME111 mean |
|---|---:|---:|
| GapPM2.5MAE, original units↓ |34.815926|34.762480|
| GeneralPM2.5MAE↓ |30.972970|30.637195|
| Gap six-channel raw-density NLL↓ |5.075791|5.000335|
| General six-channel raw-density NLL↓ |4.865349|4.862126|
| GapPM2.5proper90intervalscore↓ |401.850403|196.633594|
| GapPM2.5mean90width↓ |396.649622|115.168202|
| GapPM2.5coverage |96.643357%|86.433566%|
| GapNO2coverage |75.072464%|71.159420%|

Interval score improves51.0680%and width decreases70.9648%. All five interval-score gates now pass. Mean ordinaryPMerror decreases1.0841%; gapPMerror is essentially unchanged (0.1535%lower mean, only1/5pairedseeds improves). These are system-development comparisons with additional/different training, not compute/data-matched architecture superiority. Narrower intervals do not establish calibrated risk or sensor-gap safety.

| Seed | GapPMMAE | GeneralPMMAE | GapPMcoverage | GapPMintervalscore | Complete legacy / joint gate |
|---|---:|---:|---:|---:|---|
|111001|31.832123|30.406466|88.111888%|206.427628|FAIL / FAIL|
|111002|36.003151|31.194040|86.713287%|189.141663|FAIL / FAIL|
|111003|35.773457|30.411442|86.713287%|217.082870|FAIL / FAIL|
|111004|36.739006|30.974340|86.013986%|165.939255|FAIL / FAIL|
|111005|33.464664|30.199686|84.615385%|204.576553|FAIL / FAIL|

Legacy gate passcounts: coverage0/5, gappoint4/5, generalpoint5/5, gapNLL5/5, intervalscore5/5. Seed111003still misses its unchanged historical gap-point limit. All-channel nonregression point checks5/5; parent interval improvement5/5; all-channel coverage0/5. **Full legacy gateFAIL0/5; new joint gateFAIL0/5.** No rounding, waiver, selection, extra ME111updates or promotion.

Gap channel coverage in orderPM2.5/PM10/SO2/NO2/CO/O3:
86.4336/84.4755/80.2837/71.1594/80.5714/85.1429%.
General:89.0112/91.8086/92.7086/91.6859/89.7777/86.8305%.
Conditional missing-regime reliability remains the limiting capability.

## Persistent current-state correction

A separately labeled **post-hoc descriptive** FIT check compared the original ME111initialization and final model on identical320FITorigins/masking, without fitting or changing any gate. Only genuinely recorded values deliberately hidden from the model supply targets:94/96/100/95/86/103labels by channel. These are overlapping channel measurements, not574independent cases. Ancestors/statistics sawFIT; no independent qualification.

Mean six-channel current-state reconstruction MAE in **standardized log input units**:
0.525913→0.507204 (3.5573%lower). Channel means improve across all six; current GaussianNLL1.019193→0.989639. SeeCURRENT_STATE_DIAGNOSTIC.json for every seed/channel. This demonstrates developmental state-estimation improvement of the integrated trained candidate, not causal attribution to the R route alone, physical-unit accuracy, or natural-gap truth recovery. Natural missing readings have no invented reconstruction labels.

## Learning curves and training-time interpretation

Fixed320originalFITmonitor, no checkpoint selection:

| Updates | GapPMMAE | GeneralPMMAE | GapPMcoverage | GapPMintervalscore |
|---:|---:|---:|---:|---:|
|0|24.145534|30.585923|96.2500%|185.215634|
|512|23.960646|30.679370|96.8750%|163.063898|
|1024|23.852308|30.542674|96.5625%|165.941971|
|1536|23.519529|30.644000|96.5625%|163.153262|

Stage0already uses the new scale law; initial width contraction is architectural, not learned. Subsequent training improves monitor gap point/interval scores, while general point scores are essentially flat. FITgapcoverage96.56%versus REPORT86.43%shows a population/generalization gap, not proof that more epochs alone will repair calibration. Masks, age/site regimes, scarcity of natural outages, feature restrictions and tail specification remain hypotheses for a new prospective study. No further ME111fits.

## Mathematical and numerical verification

Source and all5live preflight/plan/tool receipts frozen0e712a798ce943b9f25c306bec1250e0b1e29031; reciprocal source snapshot6e01290ddf3f10d298c4bcf463a219c9728c0565. Initial source graph versus independent serial reference:10FP32/FP64checks across5models, all45parameter and3inputgradient interfaces. All5initial medians/native tensor states bitwise preserved; initial side scales deliberately changed. Scoped anti-cheat/Observatory probes PASS; complete integrity INCONCLUSIVE.

[PNDS draftPR37](https://github.com/FPC-effortless/pnds-lean/pull/37), commit72461c77a5f8197af08d0ac58404aefa61c7920c:11Python/source law/mask/routing tests; actual-real plus standalone Lean builds and axiom auditPASS in run40415435779. GenericPNDSCI40415435880PASS. These prove limited scale/order/role statements, not full neural/AD/IEEE correspondence, physical identifiability or statistical coverage. Initial real-role compilation failure (missing noncomputable annotation) remains recorded and was repaired before training.

Passing gate receipt/artifact contract frozen35c3c6ebbe20de7cb196619aa56287c7ec3fb832before any update. All5fits,15trained checkpoints, original optimizer/RNG/history/monitor receipts frozen**26ccea9fe5cb73ebd940e1bc07d277e017ec17db before REPORT**. Current local original-stage restoration reproduced115200monitor quantiles across20views and10final-model FP32/64forward/backward checks. Independent clean restoration and remote final-replay status are recorded separately inRESTORATION_RECEIPT.json; do not infer them from these local checks.

Full1181local prefit testsPASS;69targeted testsPASS after metadata corrections.625.473809seconds new fit-pipelineCPU (excludes monitors/checkpointIO/setup),10.104208monitorCPU;7680updates/122880exampledraws; original ancestor training additional.99840actual conditional blockcertificates,0fallbacks,maxcomputedupperdot0. Unsupported optimizer/operation FLOPs remain excluded from partial profiler estimates. No speed/compute-superiority claim.

## Research decision

Retain the positive-scale law and explicit reconstruction supervision as meaningful integrated development, with their original learned checkpoints. Do not promote ME111 or claim sensor-gap reliability. The next research target is transferable conditional tail calibration across authentic outage regimes while preserving point/state gains; another unvalidated deterministic width floor is not a solution. Independent real-data qualification, useful associative retrieval, action outcomes, vision/mobile and company utility remain separate open capabilities. Master plan P0state correction/uncertainty and issues#441/#437 guide this work; no issue is closed by this result.

Final restoration receipt: detached823a60bclean,398790REPORT+115200FITmonitor forecastscalars restored, alloriginalgates/summary and10fittedsource FP32/FP64gradientchecksPASS. GitHubfresh originalfinalreplay40418385349and finalsinglemodelparityPASS. All77SHA256payloads verified; full1181postfitlocaltestsPASS. Current reconstruction diagnostic is posthoc and not a new qualification gate. Masterplan/ideasregistry completed snapshot8f377522663bfd6a25b15bc9218814ce973267e6readbackequal.
