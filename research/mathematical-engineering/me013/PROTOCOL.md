# ME-013 — Multi-site real observational predictive-state generalization

**Registered 2026-10-20 BEFORE opening UCI 501 data or running any ME-013 training.** Preserve all ME004–ME012, PNDS Lean, all unified-master-plan hypotheses and failed gates without change. TAC-OSM remains a native predictive-executable world model, **not** an SSM. This is a *narrow functional test* of shared recurrent dynamics and learned observational context routing, NOT an attempt to recreate a new architecture.

## Grounded source and leakage contract
Primary **UCI Beijing Multi-Site Air Quality Data**, DOI https://doi.org/10.24432/C5RK5G; published hourly observed values from **12 real stations** March 2013–Feb 2017 with 6 air pollutants, meteorological variables, and missing values. Official primary archive URL: `https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip`. Exclude all unofficial short example/test CSVs. Record original archive SHA-256/byte length and exact canonical CSV member list in successful original run; subsequently freeze the observed digest, do not silently allow archive change.

Fixed **10 training stations**: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Tiantan, Wanliu.
Fixed **two never-opened held-out stations** for later zero-shot test: Shunyi, Wanshouxigong.

For ten training stations, original training script parses and normalizes:
- FIT anchors/target up to **2016-01-01 exclusive**, from first source 2013-03-01.
- Historical development anchors/targets **2016-01-01 inclusive to 2016-07-01 exclusive**. Use only for engineering diagnostics, not score-driven model/hyperparameter selection.
- Registered **unopened TIME test** 2016-07-01 through 2017-02-28 inclusive on ten known stations.
- Registered **unopened SITE test** same 2016-07-01 to 2017-02-28 on the two entirely withheld stations.
- Training script MUST NOT parse any rows on held-out stations or source rows at/after 2016-07-01. A downloaded publisher ZIP may contain all member bytes, but test labels remain unopened/unparsed. Later scoring requires hash-verified frozen checkpoints FIRST; no future labels enter recurrent inputs; test source cutoff is inclusive to 2017-03-01.

## Mathematical design and genuine hypothesis
Observed vector `x_t` consists of 11 measured physical numeric features (PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,WSPM), their *observed* Boolean masks, and 4 cyclic **known present/past** hour/day-of-year features (26 scalars); wind direction is omitted identically for every arm. If input observation is missing, value is normalized to zero **only behind a separate missingness flag**, not interpolated from future. Train-only global means/std derived from ten known stations and dates before 2016-01-01; never fit site-specific normalizers on heldout sites.

Prediction target: **24-hour future PM2.5 physical concentration** expressed as a signed change from most recent valid PM2.5, normalized using TRAIN-only PM2.5 standard deviation. Valid anchor and valid future target required; sample by fixed 12-hour stride; entire 48-observation history must remain chronological hourly. Output returned as signed delta, evaluated physical MAE µg/m³. Exact 3-arm comparison:
1. **native_plain:** proven native bounded fast/slow recurrence (32 dims each) with an ordinary zero-initialized learned output head.
2. **native_routed:** EXACT same native fast/slow recurrence, **plus 4 anonymous learned computational experts** with differentiable softmax routing from the learned world state, no hardcoded expert roles, entity IDs or station labels at inference. Outputs combine with base readout. Routing and expert specialization **not presumed**; measure per-station route utilization/entropy only as diagnostics.
3. **gru:** ordinary GRU with the *same physical/mask/time inputs*, width 44 to approximate the native-routed parameter count. Report exact counts and wall-time; optimizer-update matching does NOT imply exact compute matching.

All three arms: 3 fixed seeds 13001/13002/13003; 500 fixed AdamW optimizer updates each, batch 64, lr .001, weight decay .0001, smooth-L1 beta .2, global clip1. No model choice/refit from development. All nine checkpoints and source audit SHA256 archived. Unit/synthetic random inputs may test *software contracts only*, never demonstrate scientific predictive success.

## Honest comparators and outcome gates
Genuine later frozen tests (strictly separate PR/workflow, one first-exposure):
- Last observed PM2.5 (persistence), **same hour previous day** (seasonal naive) if observed, plus a true historical 24h trailing mean when all needed observations valid.
- Native TAC plain, native TAC routed, GRU, three seeds, same target/missing rules.
- Macro physical MAE and per station/group; quiet/large-change errors with **TRAIN-derived** threshold, negative/positive outcome changes, station-level count and route utilization, CPU fit/inference time.
- A routing benefit only if routed native beats plain native AND capacity-appropriate GRU on both known-station future and never-seen-site future in all 3 paired seeds, and beats at least one sensible physical naive baseline. Otherwise preserve failure, keep cheaper plain model; do not invent labels for latent experts.
- No learned action-cost/verification/general causal ability is qualified here: station data contain observations and true later pollution measurements, not independently verified repair interventions or costs. PNDS proof is about typed admission, not model prediction accuracy.

**No automatic merge, no publication as state of the art, no deletion of failed historical experiments.**
