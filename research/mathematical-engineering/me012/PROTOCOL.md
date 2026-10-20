# ME-012 — Frozen post-hoc quantile calibration on authentic ETT continuation

**Registered 2026-10-20 BEFORE parsing final unseen ETTh1/ETTh2 periods.**
Prior native TAC and GRU quantile models are frozen in ME011 original run 39388044790. ME011F discovered severe miscalibration on previously unopened rows [11640,14520): native TAC 80%-nominal central interval covered only 18.9% on ETTh1 and 40.1% cross-site ETTh2; GRU 34.3% and 36.6%. Both models lost to simple persistence in forecast point MAE. **No retraining/tuning/retrospective selection on opened ME011F period.** Use a fixed, established post-hoc calibration algorithm.

## Mathematical method — horizon-specific split conformalized quantile regression
For each original model checkpoint and horizon, calculate calibration scores using **only historically opened ETTh1 development/calibration row period [8760,11640)**:

`s_i(h)=max(q_{.1}(history_i)-y_i, y_i-q_{.9}(history_i))`.

For central nominal `1-alpha=0.8`, with `n` historical scores sorted ascending, select order statistic `k=ceil((n+1)*0.8)` (clamped at n; no train/test hyperparameter tuning). Correction `c_h=s_{(k)}(h)`, applied `[q_.1-c_h, q_.9+c_h]`. If c is negative it contracts overly broad intervals; do not clip without preregistration. Preserve the original median unchanged (calibration cannot improve point MAE). Report coverage, interval width and pinball before/after, but note extended adjusted extremes need not remain empirical 10/90 conditional quantiles. These are *calibrated interval endpoints* not fresh learned quantiles.

**Coverage theorem's finite-sample guarantee requires score exchangeability**, and adjacent overlapping time-series horizons or different-site readings violate or plausibly violate that assumption. Consequently **no formal distribution-free coverage guarantee is asserted for ETT temporal or cross-site observations**. This is an empirical test of transfer.

## Honest competing baseline
Persistence predicts zero OT delta. Calibrate its symmetric zero-centered interval using the **same old historical ETTh1 dev period** scores `|delta_true|`, and the identical finite-sample order statistic. No training on any later labels. Report zero-delta median MAE and central 80%-nominal interval coverage/width as the fallback reference. This controls whether a learned quantile head contributes any useful predictive interval efficiency over a simple sensor baseline.

## Frozen prospective TEST boundaries
- **ETTh1 further chronological source continuation [14520,17420)**: NEVER parsed by any ME011/ME011F scripts; two preregistered forecast horizons 24 and 168 hours; honest chronological t anchors.
- **ETTh2 different physical transformer station [14520,17420)**: SAME untouched continuation and ETTh1 train-only normalization and ETTh1 calibration corrections, with NO site-specific fitting.
- Parse no future source row >=17420; full publisher CSV checksum required, validate hourly timestamp continuity, exact column schema and no missing channels. Authenticate ETTh1 original SHA256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` and ETTh2 existing publisher MD5 `51a229a3fc13579dd939364fefe9c7ab` plus prior ME011F observed SHA256 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`. Verify six original trained checkpoint SHA256 hashes BEFORE reading any test label.

## Original evaluation gates
Report per-site:
(1) point median MAE °C (must not be cosmetically modified),
(2) raw/calibrated central interval coverage vs nominal .80,
(3) raw/calibrated interval width °C,
(4) comparison to conformal persistence baseline coverage/width;
(5) empirical pinball remains a diagnostic, not a proper score for intervals with adjusted ends.
Check structural noncrossing of calibrated endpoints around median if adjustment could contract beyond median; if violated, report that failure and do not silently clamp. Count observations and source integrity. Fixed 3 seeds per native TAC and GRU, with exact original models and no refits.

**Qualifying progress** = coverage improvement on both sites *with widths comparable to or tighter than persistence calibrated intervals*, without depending on selected favorable seed; if intervals only become extremely broad or fail cross-site transfer, report failure and preserve original raw trained model. No claims about learned entity identities, CASM action effectiveness, commercial risk, or world-model generality. No auto-merge.
