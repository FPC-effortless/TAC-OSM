# SYS1-SYS2-REAL-002: real KuaiRand source audit and observed-reward baselines

**Date:** 2026-10-19. **Outcome:** Real-source workflow PASSED. TAC-OSM pretrained checkpoints were NOT evaluated. No multi-step planning or causal advantage claimed.

## Unchanged original evidence

- [Successful original GitHub Actions run 39318654169](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39318654169), artifact 12218133513, sanitized source-audit.json.
- Dataset: official publisher KuaiRand-Pure archive at https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz.
- Original publisher MD5 verified **0820331067a3784d9691136f772b35a7**. Entire original archive SHA-256: **c814bf6f3624c0cfae83c57de3df26b2ed206e5c57bab4c4dcbfabbabe20cbf0**.
- Real training records from prior standard exposure 2022-04-09..21: **1,141,112**, 26,210 users, 7,538 videos.
- Real random-exposure records total: **1,186,059**, 27,285 users, 7,583 video IDs. Only early 2022-04-22..27 rows scored: **247,590**. Later rows excluded from *fitting and prediction scoring*: **938,469**.

## Observed-click prediction results: 247,590 randomized exposures

Every baseline was fitted only on prior standard-exposure statistics. User and video rates use fixed 20-pseudo-observation shrinkage, with a half/half ensemble.

| Baseline | Brier (lower better) | Logloss (lower better) |
|---|---:|---:|
| Global click constant | 0.227004694 | 0.647076763 |
| User-smoothed rate | 0.231117516 | 0.653061095 |
| Video-smoothed rate | 0.214501907 | 0.618998129 |
| User/video ensemble | **0.213433341** | **0.617506797** |

Training click rate on standard recommendations was 0.4634470587. The out-of-policy shift to randomized video exposure is material; even the superior predictive scores do not identify policy reward improvement. The user-only model is worse than the global constant, retained as a negative control.

## Mandatory correction: future-period exposure in original audit

The first source-audit run computed aggregate click rates across the later random and later standard CSV rows while parsing, even though the 938,469 later random rows were never used for model fitting or validation scores. **Therefore the later date range is NOT an untouched sealed TEST.** The original artifact remains unchanged. The subsequent source-audit correction explicitly records the original future aggregate exposure and stops aggregating the later outcomes into published click rates. Use a *new*, actually unexamined full-timeline cohort for any future confirmatory qualification. Do not rewrite the original artifact or falsely claim total test blindness.

## What this test does and does not establish

- Genuine publisher real-world logs and logged action/outcome provenance: **yes**.
- Observed single-step click-probability baseline: **yes**.
- Native DEV067 learned policy applied to this recommendation task: **NO**.
- SEQ008 frozen checkpoint evaluated on this corpus: **NO**.
- Full sequential user histories in KuaiRand-Pure: **NO**.
- Valid multi-step off-policy policy improvement or deliberative System 2 gain: **NO**.
- Commercial performance, online deployment or TAC superiority: **NO**.

## Next scientific gate

Acquire the publisher's original KuaiRand-1K full interaction-log archive (publisher MD5 **6b0b9c8222d67fcd4c676218edca3f1f**) for contiguous user action timelines; establish action propensity/coverage and unobserved-exposure limitations. Register genuinely fresh user/time holdouts before training. Then compare history-matched global/routed GRU, reactive ranking, explicit planners, TAC trained from scratch, pretrained DEV067 with a clearly trained adapter, and no-CASM using equal action information. Only report multi-step counterfactual policy values when propensity and effective-sample-size assumptions are met; otherwise evaluate factual reward/dynamics prediction, not planning success.

Reference protocol: research/system1-system2-real-002/PROTOCOL.md. Keep draft/unmerged, preserve DEV067 and SEQ008 checkpoints.
