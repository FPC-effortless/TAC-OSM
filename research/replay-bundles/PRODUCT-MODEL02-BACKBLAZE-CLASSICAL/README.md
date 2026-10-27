# Model 02 — Actual Backblaze exploratory classical risk reference (not TAC-OSM)

**Scientific status: original real publisher FIT and exposed DEV, but NOT independently qualified.** No source-authorized novel-TAC neural fitting, unseen population validation, hardware/cloud deployment, autonomous drive replacement policy or verified commercial product.

Original source [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) allows derivative work with attribution but not resale of the original dataset. Source-locked 2013 archive ZIP SHA256: 7f5a53e79b16e695b4b034955806bb3bb194534b169f6eca460dfd3dc48096fe.

## What the genuine fitted model actually is

- A **nine-cell smoothed binary lookup**, using only the original **same-day** SMART 5 raw and SMART 197 raw values (positive/zero/missing categories). No entity IDs, future failure flags, hidden labels, or future telemetry are input to the risk function.
- A drive-day is eligible after seven *previous/current*, observed non-failure days. A next-seven-day outcome is positive only if Backblaze flags a recorded failure after the prediction day. Unknown/missing follow-up and phase-end incomplete horizons are **right-censored**, never negative.
- FIT exposure: October 1–31, 2013. Exposed DEV: November 1–15, 2013. ZIP members dated November 16 onward are not opened/parsed for scoring. Full compressed ZIP bytes are acquired to find earlier original CSV members.
- The rolling admission and features were selected after initial 46-day case counts had already been inspected. One drive can create multiple correlated, positive seven-day windows. Published daily snapshots are not independently authenticated as live prediction-time available in 2013, and source-level failure flags are not independent adjudicated physical outcomes.

## Actual result on exposed original DEV

| Metric | Original classical risk lookup | FIT-only constant probability baseline |
| --- | ---: | ---: |
| Brier (lower better) | 0.0018760174 | 0.0019389544 |
| Log loss (lower better) | 0.0110565994 | 0.0141308740 |

The DEV matured sample comprises **204,373 complete labeled seven-day windows** but **65 distinct drives with positive windows**, and 180,532 source-eligible windows were right-censored. These scores are descriptive for the source's noncensored exposed windows and may be seriously affected by censoring and repeated examples; they do not prove useful prospective or memory-dependent prediction. No false-alert, precision-recall, time-to-failure, cross-manufacturer, or independent source transfer qualification is established.

## Original evidence and reproduction

- [First authentic source FIT and exposed DEV run #40476674797](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40476674797): nine constructed fixture tests passed, publisher original ZIP verified, source-only classical lookup fitted and scored. Earlier [fixture failure #40476555096](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40476555096) was caused by a constructed drive resurrected after its failure date; test corrected without weakening original source guards.
- Git-owned `ORIGINAL_CHECKPOINT.json` contains original nine category likelihoods, FIT prevalence and counts; `ORIGINAL_EXPOSED_DEV.json` stores original complete scores. Neither includes original source serial numbers, private CSV or permission claims.
- No-fit replay is performed by `python specialist_models/model02_replay_original.py`, which fetches the same original publisher ZIP, verifies source SHA, loads the original checkpoint and computes FIT/DEV outcomes and full original score receipt **without fitting or changing weights**. Consult the corresponding clean Git checkout workflow status; a submitted workflow alone is not verification.

## The training/qualification distinction

`specialist_models/model02_source_readiness.py` continues to report **BLOCKED_FOR_QUALIFIED_TRAINING** on this original post-exposure source. This does not contradict an explicitly unqualified historical classical *development* fit. The retained rules ban claims of confirmed new real-world skill, TAC superiority, deployability and protected-source evaluation until actual legal rights, source-time availability, independent enrollment and censor-aware evaluation are established.

Next full research step: independently authorize/freeze a newly sourced longitudinal cohort, include manufacturer/model/drive age and left truncation checks, use a proper censored survival estimator and strong tree/history-feature baselines. Only then examine matched native TAC neural memory with equal source history, identity and train/data/parameter compute.


## Actual exposed source drive-weighted sensitivity (October 27, 2026)

Previous 204,373 labeled DEV windows overlap within physical drives and can create misleading effective sample-size and confidence impressions. A **no-fit, publisher original ZIP SHA-pinned, source-DEV-only** diagnostic in `specialist_models/model02_drive_cluster_sensitivity.py` groups each original physical (serial, model) entity in-memory and reports only aggregate counts and scores. It does not print or commit drive IDs.

[Actual original-source per-drive workflow #40478889145](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40478889145) **PASS** with 8 constructed numerator/case/group tests, original 25,649 physical drives with at least one matured DEV window, only 65 of which have positive windows. Comparison includes the immutable original FIT-only constant-probability reference.

| Exposed DEV scoring definition | Original nine-cell model Brier | FIT prevalence reference Brier | Mean paired Brier difference (model minus reference) |
| --- | ---: | ---: | ---: |
| All overlapping windows (prior reported micro average) | 0.001876 | 0.001939 | -0.000063 |
| Each physical drive equally weighted (macro average) | 0.001945 | 0.002009 | -0.000063 |
| Only the earliest matured window per physical drive | 0.001190 | 0.001207 | **-0.000017** |

On the last diagnostic, the mean drive-level Brier advantage of 0.000017 has descriptive standard error roughly **0.000015** from a naive independent-drive approximation; shared manufacturer/site correlations, left-truncation, informative right-censoring and post-exposure selection are uncorrected. The smallest-effect diagnostic therefore cannot justify qualified risk reduction. Even equal-weight macro scoring does not correct informative censoring.

Full original published receipt retained at `research/source-evidence/MODEL02-BACKBLAZE-2013-EXPOSED-DRIVE-EQUAL-DEV-SENSITIVITY.json` including per-drive paired macro Brier/log-loss means, earliest-matured paired diagnostic, original denominators and clear NOT_QUALIFIED metadata. The original risk-model checkpoint and exposed DEV scores remain unchanged. Next experiment needs independently justified sampling and censor-aware outcome estimation, not another repetition of the same exposed cases.