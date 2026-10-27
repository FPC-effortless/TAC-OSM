# MODEL01-GRU-103K-CONTROL-001 — original real forecasting control results

**Date:** October 27, 2026 • Temporal lane • **status: development near-tie, not architecture superiority**. Data: authentic original UCI Appliances Energy Prediction publisher ZIP SHA256 `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a` under original source attribution and license. Exact 1,968 already-exposed 60-minute-ahead DEV forecasts per original seed from one household. Original final 20% protected rows have NOT been opened.

## Actual original matched-scale models

- Source neural TAC: **103,252 learned parameters**, original fast32/slow32/8×8 gated-delta associative state, three Git-owned and no-refit-replayed original source weights from draft #502.
- Standard GRU reference: **102,883 learned parameters**, one layer hidden145, 256-wide nonlinear readout and persistence residual. Parameter difference **0.3574%**; both within mandatory 95k–105k. All actual parameters counted once, no dummy weights.
- Same exact UCI source ZIP, chronological FIT/CAL/DEV split, FIT-only normalization, 48 source-history steps with the same appliance Wh/clock features, 6-step (60min) target, source windows and CAL/DEV metrics; seeds **1701, 1702, 1703**; *exact same original FIT indices for every minibatch* verified against native checkpoint sampler hashes, 160 updates × batch64 and same AdamW/Huber/clip/hyperparameters. Neither used protected future source.
- **Critical limitation:** FLOPs per update, state mechanism and pre-experiment model design search are NOT equalized. Original TAC work examined this DEV repeatedly prior to the GRU study. This is **parameter/data/minibatch/update controlled but NOT strict compute/search fair architecture comparison**. CPU timing parity itself is not required.

## Original authentic same-case exposed DEV metrics (not TEST)

| Fixed original seed | TAC MAE Wh ↓ | GRU MAE Wh ↓ | TAC RMSE Wh ↓ | GRU RMSE Wh ↓ | TAC 90% interval score ↓ | GRU 90% interval score ↓ |
|---|---:|---:|---:|---:|---:|---:|
|1701|35.4242|36.4532|86.6638|86.3777|447.2049|442.7216|
|1702|39.1408|38.0633|86.1045|85.9743|440.3443|439.0320|
|1703|39.2296|38.4873|83.6013|84.5435|422.9428|431.0793|
|**Three-seed mean**|37.9315|**37.6680**|**85.4565**|85.6318|**436.8307**|437.6109|

Mean GRU MAE 0.6948% lower than native TAC, while native TAC interval score 0.1783% lower than GRU. GRU wins two seeds on MAE and native wins one. Both are materially better on exposed DEV point MAE than original small-source classical ridge (44.7923 Wh) but compared models and training budgets for ridge differ, so do not interpret as a matched architecture superiority claim. Original mean empirical 90%-nominal interval coverage: TAC **90.72%**, GRU **90.46%**; actual `available_at` data and independent forward coverage unknown.

**No architecture winner was demonstrated.** GRU and TAC perform similarly on short-history one-household 60-minute forecasting; this source cannot demonstrate that TAC associative memory provides more useful predictive value or long-delay recall than a conventional GRU. Do not select the best checkpoint or construct confidence intervals treating repeated overlapping source predictions as IID.

## Original checkpoint and source proof

- **Original real FIT and same-run replay**: [Actions #40499963562](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40499963562), 10 software tests and all three 160-update authentic source GRUs trained, with complete original `torch.save` learned weights/optimizer/torch and NumPy sampler/Python RNG states and FIT indices hashes.
- Original source GRU Git bundle `research/replay-bundles/MODEL01-GRU-103K-CONTROL-001/ORIGINAL_FIT.json`, frozen TAC bundle `research/replay-bundles/MODEL01-TAC-REAL-FIT-001/ORIGINAL_FIT.json`; model SHA256s and matching source sample hash verified per seed.
- Entire old/new checkpoint-derived source result committed at `research/source-evidence/MODEL01-TAC-vs-103K-GRU-ORIGINAL-DEV.json`; [source scorecard CI #40500738597](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40500738597) passed 8 original integrity tests. Scores derived solely from original records and source-checksum pinned, no refit.
- **Independent fresh-checkout replay** [#40501645798](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40501645798) PASS on the final documented-tolerance source. **Compulsory research parity** [#40501645804](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40501645804) PASS.

### Replay numerical honesty

Several earlier clean GitHub runners failed original 1e-6 relative interval-width equality while all original model bytes and sample digests matched. [Source CPU numeric probe #40501308177](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40501308177) localized this to original seed1703 CAL q90 halfwidth difference of **0.000152587890625 Wh**; corresponding full-width relative change **1.713×10⁻⁶**, interval score relative change **2.124×10⁻⁷**, with source DEV MAE/RMSE and empirical coverage unchanged. Disabling oneDNN did not remove tiny numeric difference.

The replay contract now **does not claim bitwise-identical floating-point outputs**: original compressed Torch checkpoint SHA256 and exact original minibatch digest are bitwise checked, and original CAL q is checked within **0.0002 Wh absolute**, full interval width within **0.0004 Wh absolute**, and all other source DEV scores within **1e-6 relative**, without changing weights or re-fitting. Failed prior checks remain visible and are not marked passing. These tolerances are below source measurement precision but must remain auditable.

## Scientific/release scorecard

- **Capability:** authentic one-household development point forecasting with 48-step history. TAC improves over old ridge but not over matched-scale GRU on mean MAE; model novelty/persistent-memory contribution **unproven**.
- **Evidence tier:** `PARAMETER_SOURCE_UPDATE_MATCHED_EXPOSED_DEV_NOT_FLOP_OR_TUNING_MATCHED`; retrospective DEV repeatedly exposed; original physical source `available_at` unverified; no independent household split.
- **Metrics:** original per-seed MAE, RMSE, interval score, mean coverage/width retained; no cherry-picking. TAC phase-buffer 94% postprocessor in separate #507 is **post-hoc and not applied to GRU**, so comparing that later interval to the GRU original calibration is not a fair uncertainty-head study.
- **Reproducibility:** actual original three GRU and native weights/optimizer/sampler in Git, independent real source replay within documented strict physical-unit tolerances. Scientific `DEV` improvement != promotion. **No user, verified device, actual Woyengi integration, automatic meter control, intervention effect, cloud cost or revenue**.

**Next decision:** seek an entirely new, same-target legally authorized appliance site with genuine available-at timestamps, stable meter identity, observed outcomes and frozen qualification gates; compare trained full models, no-fit persistence and classical historical-feature reference under equal data, param and training-FLOP/search budgets where asserting architecture superiority. Until then, native source forecasting is useful development research, but it is not independently superior to standard GRU.
