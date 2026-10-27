# GP-UNC-005 — real original trained ME111 neural-to-GP memory bridge

**Research date:** 2026-10-27. **Evidence:** authentic UCI501 2013–2014 original FIT source observations, original trained native TAC neural weights, READ-ONLY forward operator, no target scoring, no adaptation and no new fit. **Not** a qualified uncertainty improvement, independent TEST score, neural GP training, latent prediction superiority or deployed latency result.

## Source identities and replay

- Original model source/forensic wrapper pinned to [TAC commit `77c753a69f10fa13cd7fdf849edb12bc53f03e6d`](https://github.com/FPC-effortless/TAC-OSM/commit/77c753a69f10fa13cd7fdf849edb12bc53f03e6d). The actual frozen training source is the inherited ME111 graph; no model update occurs in this study.
- Original publisher: UCI501 Beijing Multi-Site Air-Quality Data (CC BY 4.0). Original archive ZIP SHA256: `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
- Five original ME111 seeds `111001,111002,111003,111004,111005`. Original trained parameter count per checkpoint **104,663 FP32**, original 13 source sums +13 masses +62 evidence sums +8×8 memory (152 persisted FP32 values per station, aside from counters/metadata).
- Checkpoint hashes individually checked against original `research/replay-bundles/ME111/FIT_INDEX.json`, actual bytes checked before and after inference. The pre-existing original ME111 2015 exposed REPORT interval-coverage qualification **FAILED**; this work does not change that finding.
- Source: twelve original genuine station IDs. Two FIT-origin source hours `12000,12064`, each with the original 64-hour input-only contextual window. Forecast horizon belongs to native ME111 original graph (six source hours ahead), not a newly trained output. Six forecast channels per station.
- Raw FIT loader **pre-parses original FIT data**, including future FIT observations and performs FIT-wide normalization. The bridge does **not** extract those future targets for scoring or pass them as predictor inputs. This historical-source negative-control audit does not establish online data availability, an independent future cohort or future-free pretrained parameter estimation.

## Exact results and replay receipt

Revised implementation [source commit `8d32b7b1232a77c11dc6d6323e05b6aed166a27e`](https://github.com/FPC-effortless/TAC-OSM/commit/8d32b7b1232a77c11dc6d6323e05b6aed166a27e).

[GitHub Actions native neural source bridge #40500563279](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40500563279) **SUCCESS**, with derived archived JSON evidence artifact `gp-unc-005-me111-original-fit-neural-no-fit` (artifact ID `12732945826`).

| Source-backed software invariant | Actual result |
|---|---|
| Original trained checkpoint count | 5/5 |
| Per-checkpoint useful trained neural parameters | 104,663 |
| Real-FIT native neural forecast issue events: 5 seeds ×12 stations ×6 channels ×2 origins | **720** |
| Native original normalized log point predictions unchanged by GP sidecar | 720/720 |
| Native original left/right uncertainty scales finite, positive and unchanged | PASS for all forecast events |
| Source future-only numerical poison at original hour t+6, applied to original `raw` + standardized `values` | 720/720 per-forecast/channel equivalent checks |
| Bit-identical original source→native inference input frames after future-only poisoning | PASS |
| Bit-identical original native point, scale and source/evidence/associative carried state after future-only poisoning | PASS |
| Original trained associative memory `8×8` used as actual **64D GP feature vector** | 720/720 |
| Nonzero native memory coordinates across 24 station-source frames per seed | 1,536/1,536 for each seed |
| Model parameter-version mutations, optimizer updates, GP residual assimilation | **ZERO** |
| Exposed 2015 REPORT target scoring and protected 2016–2017 numerical data access | **ZERO** |
| New GP likelihood, observed coverage improvement or independent evaluation | **NOT EXECUTED** |

Original five model checkpoint SHA256 identifiers verified in workflow artifact:

| Seed | Original checkpoint SHA256 |
|---|---|
| 111001 | `337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449` |
| 111002 | `445558b5aae9e67b8c3227a935517a3c9bfe484916f4eae20619465e46975f40` |
| 111003 | `063ffca5e19e96f7b4169f46dea64067e8cd59f23b3bc23aee3db229afc35927` |
| 111004 | `fffbb0ea89286d01fcb73395a6926aa81a413112ba5084e01406b802d519df26` |
| 111005 | `ddb4b3dce96ef75d5b644d9eabc2c30d0108fa736f0e8a35e1154a0c91882b1c` |

## Unhidden resource cost

The prototype uses one independent scalar Bayesian GP filter per station×channel, **72 filters per model**, each with two *fixed nontrained* 64D inducing coordinates and a small posterior covariance/mean. According to the reference `retained_numeric_scalars` rule, each GP filter retains 137 floating-point scalars. The full sidecar therefore retains **72×137 = 9,864 floating scalars ≈39,456 bytes in FP32** if implemented in contiguous FP32 form, beyond original neural 12×152×4 = 7,296 station-state bytes. Thus this particular per-channel prototype requires **~5.41× the native model's 12-station persistent float state in extra GP state**. Actual Python implementation stores double-precision objects and has further overhead; it has not been profiled for all 72 filters' deployed latency or memory. The numbers are theoretical float-count illustrations, NOT measured GPU/RAM footprints or model parameter additions. This must be optimized and budgeted before Edge-Lite claims.

## Interpretation, open qualifications and experiments

This shows that one frozen Gaussian posterior sidecar can receive genuine original trained TAC memory vectors while preserving point and original split-scale outputs under an explicit *zero learned correction*. The original ME111 conditional left/right scales are asymmetric: blindly adding symmetric GP variance to both does **not** preserve the original trained predictive distribution or prove calibrated quantiles. A principled distribution-combination law and proper scoring (NLL/CRPS/coverage) remain OPEN.

To claim predictive improvement, run an explicitly registered, source-qualified, signoff-eligible study on genuine **separate** chronological source cohorts; freeze base checkpoints, original issue/availability clock and independent future outcomes, conformal CAL only, GP inducing/training hyperparameters, matching trained model and baseline exposure/FLOPs, and scoped uncertainty/decision utility metrics. Reuse prior public FIT for no-fit diagnostics only; 2015 exposed REPORT is not an independent heldout; protected 2016–17 data remains sealed. No false progress from synthetic constructed numerical diagnostics or original FIT replay.

**Research state:** `NO_FIT_SOURCE_NATIVE_BRIDGE_PASS`; `REAL_NEURAL_GP_IMPROVEMENT_NOT_TESTED`. Related [TAC GP draft PR #491](https://github.com/FPC-effortless/TAC-OSM/pull/491), [PNDS partial mathematical draft #45](https://github.com/FPC-effortless/pnds-lean/pull/45), [GP study qualification tracker #492](https://github.com/FPC-effortless/TAC-OSM/issues/492).
