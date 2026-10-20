# ME021 — Original first-user5 results: small real interaction, powerful simple baselines

**Original success:** [Actions run 39455453892](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39455453892); [artifact 12280597844](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39455453892/artifacts/12280597844), artifact archive SHA256 \`e3efbb63e3e053203f544f6909c23212111b5643d1ca44bcd97f65e42285beac\`. Original inner results \`results/me021/frozen_new_user5_interaction.json\` SHA256 **\`efd858457c257d01802a3d2b24084e22416f35d8fdfb813717ab3c13b33cd63c\`**. Original archive also preserves true event user IDs/labels and all 18 original and calibrated predictions plus 17 simple-baseline predictions in \`frozen_future5_event_scores.npz\`. An independent calculation from the archived event-level arrays reproduced the primary point estimate EXACTLY and checked individual baseline Brier scores to floating precision.

## Fixed real-world source and integrity

Official publisher KuaiRec \`big_matrix.csv\`, source ZIP SHA256 \`c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c\`. All original 18 exact 135050-parameter frozen trained checkpoints from ME019 successful original run 39443325909, original train manifest SHA256 \`2a10ae3205e866ed792f5ad145e3c3c775e95c39de675f421184de44e56455b6\`, individually SHA-verified BEFORE source read. Fit-only original users ID%16=0 before Aug01 2020, historical calibration DEV Aug01–Aug14, and once-first future user ID%16=**5**, target Aug14–Sep6 2020. The code fixed all baseline fitting and 18 calibrators before opening user5 outcome source. No original recurrent weights were updated; previously opened user4 scores not reopened or retuned.

Actual completely new cohort: **42,319 factual logged video impressions/outcomes from 436 real unseen users**, event full-watch prevalence 0.325646. All models scored exactly the same unpadded 128-eligible target actions and current response labels. The most-recent 32 features/priors are identical for T32 and T128. Authenticated causal features only. Real future performance evidence, *not* causal intervention efficacy.

## SINGLE PREREGISTERED PRIMARY: across-seed paired Brier interaction

For each real target i and frozen seed s:

    d_{i,s} = (e_TAC128 - e_TAC32) - (e_originalGRU128 - e_originalGRU32)
    e_model = (forecast_full_watch_probability - observed_full_watch)^2

Mean all actual events after averaging the same three trained seeds:

**+0.0015700983 Brier** — positive means longer history hurts TAC's error trajectory relative to original GRU.

A single correctly paired **user-ID clustered bootstrap** (2,500 bootstrap user samples, RNG 21021, same user resample shared across *all* seed models and both history lengths) yields **95% CI [+0.001303445, +0.001844673]**, conditional on the original three fixed trained seeds and this study cohort. 436 independent observed-user clusters are used; the seeds are NOT independent study subjects. Individual seed interaction points: +0.0011640, +0.0017000, +0.0018463. The interval excludes zero for this specific real future cohort and frozen checkpoints, but the effect is small in absolute Brier units (around 0.78% of the ~0.20 underlying Brier). It does NOT prove a recurrent Jacobian mechanism or equal-compute TAC/GRU architecture superiority; original train compute/FLOPs across arms were not perfectly matched.

Secondary (NOT new primary) paired historical-DEV-calibrated interaction: **+0.0015365402**, user clustered 95% CI **[+0.001258973, +0.001828371]**. **Simple historical Platt recalibration does NOT remove the contrast** on this first new-user cohort. These fixed calibrators were fit only on earlier genuine already-inspected development responses before opening user5; no calibrator choice or response training from user5.

## All original frozen architecture means on new future user5 (three fixed seeds)

| Original architecture | Context T | Raw Brier ↓ | Pre-DEV-calibrated Brier ↓ | Raw ROC AUC ↑ |
| --- | ---: | ---: | ---: | ---: |
| Native TAC chrono+prior | 32 | 0.20182602 | 0.20141578 | 0.671729 |
| Native TAC chrono+prior | 128 | 0.20246671 | 0.20190588 | 0.669175 |
| Original unanchored GRU | 32 | 0.20178385 | 0.20151371 | 0.673604 |
| Original unanchored GRU | 128 | **0.20085444** | **0.20046726** | **0.678141** |
| Anchored GRU | 32 | 0.20290291 | 0.20136655 | 0.671658 |
| Anchored GRU | 128 | 0.20284354 | 0.20119401 | 0.672492 |

Mean TAC within-architecture extension degrades raw Brier +0.00064069. Original GRU extension improves −0.00092940. The resulting interaction is +0.00157010. The AUCs also show worse native discrimination with longer history, and the original GRU-128 the best of these six trained recurrent arms. Historical-DEV Platt recalibration does not reverse relative interaction. The original complete JSON contains per-seed calibrated reliability slope/intercept and descriptive test-population slope/intercept (descriptive only; NEVER applied in predictions).

## Strong *simple* real baselines: MUCH bigger limitation than memory effect

All simple-learned arms fit on only the original genuine pre-Aug01 FIT target observations, no current or future response in causal features. Same 42,319 authentic first-future user5 target rows. Real causal predictors: prior32, prior128, seven fixed exponentially-weighted means of previously observed user outcomes, pre-Aug01-only target video popularity, current video duration and historic timing gaps. New logistic and small 16-unit MLP trained exactly 300 AdamW steps, 3 seeds, no user5 model updates. These have only **13** and **225** active parameters, respectively, and are **diagnostic baselines NOT equal-parameter neural architecture arms**.

| Actual real baseline | Brier ↓ | ROC AUC ↑ |
| --- | ---: | ---: |
| **16-hidden-unit MLP on causal EWMA+metadata features, 3-seed mean** | **0.16975203** | **0.798526** |
| **Logistic regression on identical causal features, 3-seed mean** | **0.17005695** | **0.799535** |
| Fixed EWMA, half-life 16 events | 0.19851793 | 0.687834 |
| Fixed EWMA, half-life 32 events | 0.19891383 | 0.68381 |
| Fixed user128 Laplace mean | 0.20147326 | 0.67189 |
| Fixed user32 Laplace mean | 0.20177369 | 0.67586 |
| FIT-only target-item popularity | 0.22074914 | 0.50803 |
| FIT-only global event rate | 0.22106833 | 0.50000 |

The best recurrent model (mean original GRU-128 Brier 0.200854) is **~0.0311 absolute Brier worse** than the simple small MLP (0.169752), a MUCH larger difference than the TAC-vs-GRU history-length interaction (+0.001570). Likewise native TAC32 mean 0.201826 trails both simple learned baselines. This is a strong warning that the existing neural model's training representation/optimization is inadequate for the current factual logged video-response task, **not evidence that all high-dimensional recurrence is inherently useless**. The simple baselines have extra explicit engineered rate features and entirely different trainable capacities, so do not claim an equal-capacity or FLOP-matched scientific architecture win. It is a highly relevant **commercially and scientifically necessary sanity check**.

Original original_128_windows() constructs past history via records[i-128:i] and the current label from records[i], excludes current response from the rate features. Target video popularity is fit only to original FIT reactions; current video duration is a pre-response known covariate. No label leakage was identified in the implemented feature construction. Need future dataset/source replication before claiming generality.

## Frozen random-state probe undercuts strong memory-learning inference

The exact same original ME020 fixed genuine FIT 1,200 history windows, 975 train-probe examples and 225 probe-test examples across **68 distinct probe-test users**, ridge λ=1, real oldest8 mean watch outcome target, final time-128 states:

| Source of final state | Three-seed mean true user-disjoint ridge probe MSE ↓ |
| --- | ---: |
| Trained native TAC-128 | 0.03458351 |
| **RANDOM INITIALIZED, UNTRAINED** native TAC-128 | **0.03656167** |
| Trained original GRU-128 | 0.04142207 |

Random recurrent TAC state also preserves much linearly decodable information. Trained TAC beats its own random-init state only modestly (~0.00198 MSE), and random untrained TAC already beats trained GRU on this probe. **Do not cite earlier ME020 'TAC stores distant information better than GRU' as evidence of *learned* memory without this control**. Probe evaluation is within original FIT and is not independent user5 prediction.

## Chrono, learning curves, and zero-residual math

Verified actual source uses fixed gate biases: fast initial update probability 1/8, slow 1/24, unchanged at T=32 vs T=128. This is a fixed timescale prior, not T128-range Tallec–Ollivier chrono initialization. It is now reasonable to test explicit horizon-aware timescales but no correction has been experimentally validated.

Original ME019 learning traces record sparse batches only at optimizer steps **1, 150, 300**, not a continuous training curve; no conclusion about optimizer convergence can be drawn from them. Original trained recurrent arms used exactly 135050 trainables and 300 matched data-index draws per seed, but FLOPs/time were not fully equal. The original fixed analytic prior32 already had a zero-initialized neural output head (same predicted baseline at step zero), and nonetheless post-training outcomes can get worse. A zero-initialized learnable delta over an EWMA prior does **NOT** guarantee never-underperform after fitting.

## Binding next research decision

**DO NOT insert an extra memory component, arbitrary attention, MTSK, Titans/MIRAS or force Jacobian-preserving gates based on ME020 gradients.** First rule out training/feature inadequacy on this authentic dataset with fixed-rate feature access for trained neural controls, proper learning curves, controlled compute budgets, and exact same known-at-decision-time covariates as the simple logistic/MLP. Preserve learned/native recurrence identity and original 32 checkpoint, strong GRU controls, and the ME018 exact-math kernel. Establish baseline-competitive predictive value before any claims of commercially meaningful learned world-state reasoning. Frozen future5 cannot be recycled as a new unexposed test after inspecting this result.

This is an immutable, original first-exposure and random-control documentation report. Draft PR only, no merges or production promotion.
