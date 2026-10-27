# GP-UNC-006 — genuine original FIT residual assimilation: NEGATIVE result

**Date:** 2026-10-27. **Status:** historical original-FIT in-sample neural-plus-GP diagnostic completed, **architecture promotion FAILS**. Original five trained ME111 checkpoint weights preserved; no new neural training, protected TEST, 2015 REPORT scoring, GPU or extra learned neural parameters. The bounded Bayesian residual GP **does assimilate 5,370 real FIT residual events** (a statistical adaptation)—do not conflate this with the prior GP-UNC-001–005 no-fit tests.

## Strict source/experiment boundary

- Original real original UCI501 Beijing Multi-Site Air Quality ZIP SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`. Historical original source/weights pinned to [TAC source commit `77c753a69f10fa13cd7fdf849edb12bc53f03e6d`](https://github.com/FPC-effortless/TAC-OSM/commit/77c753a69f10fa13cd7fdf849edb12bc53f03e6d), original checkpoint SHA per seed checked against FIT_INDEX, before/after read-only neural inference.
- Five original ME111 native trained networks, **104,663 learned FP32 parameters each**, already trained using the **same 2013–2014 FIT-period records scored here**. Five seeds are not five independent new real-world populations, and the original FIT loader had already parsed all FIT including later source observations.
- Previously fixed origins: 16 source hours `12000,12008,...,12120`; source-specific 6-hour future target, 64-hour native context, twelve station IDs, six forecast channels. Two fixed inducing memory addresses from earlier FIT hour inputs `11968,11984` and no label-based selection. Fixed before metrics: `noise_variance=0.5`, 2 inducing points × 64D each, per station×channel. No sweep, no retransformation to raw physical units.
- Three exactly scored original outcomes per row: frozen original ME111, constant-input blinded projected GP, actual trained original 64D native-associative-memory projected GP. Both GP arms share same two-point x64 storage/observed residual events, frozen hyperparameters, label timing and original checkpoint. GP updates only after original index-clock `t+6`, strictly before next later source-issued prediction. Historical index does not certify operational label availability.
- Original mean and corrected point error measured in **standardized log pollutant units**. No NLL/interval/coverage statistics produced, so **no probabilistic uncertainty calibration claim**.

## Exact source-run result

[Source-pinned GitHub Actions run `40510749996`](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40510749996): SUCCESS. Raw JSON artifact `gp-unc-006-original-2014-fit-in-sample` (ID `12736573872`). The workflow's science guard rejects independent/qualification claims and the original learned parameter counts remain unchanged.

- Original issued neural forecasts: **5,760** (5 seeds ×16 hour issues ×12 stations ×6 channels).
- Original observed target events scored: **5,730**; genuinely missing observations **30** excluded identically for all arms.
- Actual FIT residual-assimilation events applied to *both* bounded-GP arms: **5,370**. Remaining trailing observed results are source-time-ineligible to update earlier predictions.
- All five trained original checkpoints retained **104,663 learned parameters**, zero optimizer steps, and no protected source rows evaluated.
- Original publisher archive was pre-parsed in FIT loader and original ME111 checkpoint had trained on the scored FIT labels: this is **in-sample**, *not generalization or independent evaluation*.

| Arm | Original-scored cases | Mean absolute error | RMSE |
|---|---:|---:|---:|
| Frozen native ME111 | 5,730 | **0.406139** | **0.537303** |
| Native 64D-memory GP | 5,730 | 0.433357 | 0.574997 |
| Constant 64D-blinded GP | 5,730 | 0.443675 | 0.587922 |

Native-memory GP's descriptive in-FIT MAE is **6.70% worse than the original frozen neural model**. Blinded GP is even worse; memory GP is **2.33% better than blind GP** but **still loses to the neural model**. This differential does not prove associative-memory causal importance or independent useful retrieval.

**Each original seed independently fails the original neural baseline point-error comparison** (seeds share the same original FIT sources):

| Original seed | Original native MAE | Memory GP MAE | Blind GP MAE | Memory GP vs native |
|---|---:|---:|---:|---:|
| 111001 | 0.407382 | 0.430678 | 0.446795 | +5.72% worse |
| 111002 | 0.397538 | 0.423595 | 0.434128 | +6.55% worse |
| 111003 | 0.402133 | 0.434827 | 0.443436 | +8.13% worse |
| 111004 | 0.407814 | 0.437272 | 0.444312 | +7.22% worse |
| 111005 | 0.415829 | 0.440416 | 0.449704 | +5.91% worse |

## Scientific decision — STOP and preserve negative result

**Do NOT promote this GP sidecar for Temporal, Enterprise, Physical, System 1, Edge-Lite or Counterfactual variants.** No newly trained equal-budget architecture comparison, real original heldout source validation, calibrated likelihood improvement, decision utility or robust model generalization has been demonstrated.

Do not tune inducing points/kernel length/noise/feature projection repeatedly on exposed ME111 FIT, previously exposed 2015 REPORT or LATFUSE019 and then describe a subsequent improvement on those same observations as new independent success. Preserve the receipt and source/score recipe exactly as the negative diagnostic. A new frozen, independently source-qualified longitudinal cohort and pre-registered scorer are required to evaluate any NEW candidate. Historical original native ME111 2015 undercoverage FAIL remains unchanged. Properly combine asymmetric native uncertainty and additional GP residual variance only after defining and verifying a coherent probability law; until then GP latent variance is NOT a calibrated confidence measure.

Original code [TAC PR #491](https://github.com/FPC-effortless/TAC-OSM/pull/491); mathematical partial [PNDS PR #45](https://github.com/FPC-effortless/pnds-lean/pull/45); [research Issue #492](https://github.com/FPC-effortless/TAC-OSM/issues/492). Both PRs remain draft and unmerged.

**Explicit proof/training parity boundaries:** original neural checkpoints saw all historical FIT target events during their own fitting; sidecar posterior adaptation updates from these *already-seen* FIT labels. GP memory states are not the same thing as learned neural parameters and do increase deployed memory and online arithmetic. The constant-GP control matches sidecar dimensions/inducing count, not all training FLOPs/optimization of a newly trained GRU or Gaussian-head comparator. No general architecture-superiority or calibration claims are permitted.
