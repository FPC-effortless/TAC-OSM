# LATFUSE013 — original-source historical PM2.5 content decodability

**2026-10-27. Original UCI501 real physically measured source, original trained TAC checkpoint SHA VERIFIED, source-only external ridge diagnostic SUCCESS; the prospectively registered H64 scientific gate FAILS. ZERO original TAC model training.**

[Real-source GitHub Actions #40460640807 — SUCCESS](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40460640807). Precommitted REGISTERED_REAL_CONTENT_PROBE.json before true source measurement evaluation. Complete 2 original SHA-pinned source-trained LATFUSE008 assoc query checkpoint models 80701/80702, original same full 99390-parameter 6800-byte recurrence, no changes to source writer, content encoding, training data/split/checkpoints, no protected numerical source read. Sidecar ridge only, source-provenance current original FIT numerical measurements.

## Real physical source and split

Original licensed UCI501 Beijing 12 sensor stations, archive SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. Original historical anchor target is **measured raw PM2.5 µg/m³**, model training normalizer from original 2013Mar–2014Feb before target source records. Original 96 matched high>=150 versus ordinary<=35 authentic physical source station/hour-of-day label pairs sampled chronologically from March1–March30 2014, and the EXACT originally preregistered 24 May–June2014 real matched pairs to evaluate sidecar. No new true source model optimizer, source input clocks/physical owner/missingness honored. Model recurrence consumes full prior24+H real sequential hourly values until read time; **target PM2.5 magnitude consulted only after frozen memory reads** for external offline ridge labels. Entire earlier May–June source was originally inside FIT and previously inspected; this is **NOT independent pretrained model holdout** and cannot support claims of new research capability. No original 2015 exposed DEV/2016–2017 protected numerical records parsed.

## Controls and proper accounting

At H16/64/128/512, for each frozen source-trained seed, fit a separate **external fixed-lambda=1 dual ridge** of actual recorded original past PM2.5 log1p magnitude on March source observations, with train-only feature/target standardization, fixed feature clipping/dimensionality scaling, no May hyperparameter fit. Baselines and reads are all in same original matrix value16 coordinate space where applicable:

- **source_only**: legally observed present source measurements/masks/last/age plus source ID and known requested historical clock;
- **nonassoc**: source_only plus original fast80/slow80/source owner entity80 without matrix;
- **native_read**: source_only plus original learned historic clock/belief-conditioned matrix read16; no old-event oracle key;
- **oracle_read**: source_only plus source-event original WRITE-key query into current trained associative matrix, privileged diagnostic not deployed;
- **wrong_key_read**: source_only plus original OTHER member of high-vs-ordinary SAME station/clock pair WRITE key read16, privileged diagnostic; **this is LABEL-CORRELATED, NOT a neutral negative** because pair construction requires opposite extremes;
- **random_key_read**: source_only plus deterministic source-id/time-seeded normalized two-head random8D query read16, no target class fed into sampler;
- **oracle_only/wrong_only/random_only**: no nuisance features, privileged external decode capacity checks;
- **current last-observed PM2.5 persistence** and frozen original train-mean target prior.

External ridge has additional fitted diagnostic parameters (source-only dim? exact recorded in JSON) and is NOT a trained native TAC model or proof of constant-memory inference. Each original model 99390 parameters and nominal6800B state unchanged. The original source high-vs-low pairs let even label-correlated wrong-key queries indirectly predict target class; avoid treating wrong-key superiority as clean retrieval-specific evidence.

## Authentic evaluation — original log(1+PM2.5) physical reconstruction MAE (lower better)

| H64 source read/control | Original seed80701 | Original seed80702 |
|---|---:|---:|
| Privileged write-key oracle+source | 0.9586 | 1.0392 |
| Opposite-label physically matched wrong-key+source | **0.7861** | **0.8470** |
| Deterministic isotropic random-key+source | 1.3189 | 1.2911 |
| Current+fast/slow/entity, NO associative memory | 1.0393 | 0.9659 |
| Native 16D trained historic read+source | 1.4302 | 1.3760 |
| Current source nuisance alone | 1.3424 | 1.3424 |
| Current source legal last-observed PM2.5 | 1.1844 | 1.1844 |

**Primary predeclared pass criterion:** oracle+source H64 must improve original log PM2.5 MAE by >=5% compared with EACH of wrong-key+source, random-key+source and nonassoc+source in BOTH original fitted seeds, each contrast with station-block 4,000-draw 95% bootstrap lower bound>0. **FAIL BOTH SEEDS.** Against label-correlated wrong-key baseline oracle is worse by22.0% and22.7%. Against nonassoc seed80701 improves7.8% but station-block CI crosses0, seed80702 worse by7.6%. Against random key it improves27.3% (80701) and19.5% (80702) but latter station-bootstrap CI includes0. Frozen complete results in RESULTS.json.

Original privileged oracle+source high-event versus ordinary paired RANK is95.8% in both seeds at H64, illustrating how paired ranking can look strong even with inaccurate numerical values or label-correlated source pair setup. At H16 the source nonassoc beats oracle substantially (MAE .693/.600 vs oracle .974/.871). At H128/512 privileged oracle may beat nonassoc in some seeds but wrong-key numeric source read often beats oracle; all nonprimary H metrics descriptive, no new model selected.

### Critical interpretation: outcome-informed mismatched-key confounding

**The wrong key is computed from the other anchor of the intentionally high-PM2.5 vs ordinary pair.** It is not an unbiased negative: the pairing selects opposing physical magnitudes and dates, making wrong-key features correlated with the TRUE anchor event's label. An external decoder trained on extreme pair labels can invert this class relation and score numerically better without faithful temporal retrieval. We must not infer actual writer failure, inverse address quality or an ability to retrieve the correct event from this control. Even the privileged true key is a label-dependent oracle (computed from the ground-truth past sensor input) and is unavailable to the current legal historic-time-only readout. The proper comparison for next mechanism audit requires **same-site intervening observed write keys selected without conditioning on high/ordinary class**, plus past-target-shuffled genuine event clock negatives and a centered within-source hard-negative identity retrieval test.

Linear ridge probing is a **lower bound on stored decodable content**; a negative score is not a proof that information is absent (nonlinear retrieval could exist), and a positive score alone doesn't prove source factual recall from a legal timestamp-only query. The previous original pretrained models saw the internal FIT source including the May evaluation anchors, and sample count24 physical pairs / only2 original seeds and weather-correlated stations are small.

## B source-content reconstruction decision thresholds and current status

**Do not greenlight a query-only B neural training run based solely on uncentered oracle cosine or H64 probe.** Prospective gate FAIL. A new independent and source-unbiased **write-key specificity test** is needed before declaring value writer failure or recommending a writer-relevance intervention:
1. use physically observed same-source-window intervening keys, sampled by *timestamp only* and with no binary event-class or target magnitude selection; record source site/date/missingness;
2. center value16/read vectors using **TRAIN source only** then classify/continuously reconstruct genuine past PM2.5 from oracle vs no-label-conditioned wrong/random/fastslow, train/test strictly chronology, source station bootstrap;
3. predeclare **real-source within-window top1 event/content retrieval** among 8 authentic past timestamps, with exactly seven source-date-independent negatives of same source window and 12.5% random rank chance, no target content fed to native query;
4. train a matched legal native B0 historical-content reconstruction head and nonassoc query baseline ONLY after independently measured full forward/AD/optimizer FLOPs <=1% and active/deployed model parameters <=1%, identical original genuine source sample/time/optim/budgets/2+ seeds, exact checkpoint optimizer RNG/data hashes and independent clean replay. Original 6800B recurrence and 99k live parameters must remain, protected data unopened;
5. do not combine B0 supervised real source content reconstruction, B1 within-window contrastive alignment and later paired time-addressed A as one unisolated arm. Report source observed multi-channel +6 forecast/persistence and channelwise calibration separately.

**Status**: source true continuous historical-value external probe finished, registered H64 decision gate FALSE and recorded without reruns/tuning. No original model re-fit, actual query fix, causal physical control or qualified real source generalization claimed.
