# S1MOTION003 — first trained real ALOHA tape System1 results (2026-10-26)

**ORIGINAL PREREGISTERED PREDICTIVE GATE: FAIL 0/3 original seeds.** Study is authentic measured robot joint trajectories, not synthetic evaluations. These are same-publisher episode DEV observations, not independent robot/site safety qualification. Unmerged draft PR #418; parent untrained source candidate PR #416.

## Preregistered source and immutable original artifacts

- Source-only prefit eligible workflow [#40277021599](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40277021599) **PASS**, before any fit. Original 50 episodes × 700 real recorded 14D ALOHA joint states/recorded actions at 50 Hz, Apache-2.0; LeRobot snapshot `92955878beb0da15eee5d4d869841577f87e903e`; data/chunk-000/file-000.parquet SHA256 `ff4e1640f3ab7586863332f333dc9341e3cf86b919664c6be90110d37e0ad6d4`.
- Original complete real fit + first-scored workflow [#40277450986](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40277450986), **CI SUCCESS**. Original source/code ref commit `2ebd67eced868a5d9f134cc6b08bcfb05ddc2582`. All six models frozen with source checkpoint SHA256, entire optimizer/RNG/epoch schedule **before DEV access**. Original evidence artifact [12630707934](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40277450986/artifacts/12630707934) including raw original score JSON and model/optimizer states (time-limited retention); source-controlled [S1MOTION003-ORIGINAL-LEDGER.json](S1MOTION003-ORIGINAL-LEDGER.json) permanently stores exact recorded primary scores and six model SHA hashes. Never replace these original observations with later re-tuned estimates.
- FIT original episodes 0–24; DEV 25–34; reserved TEST episodes 35–49 **not used** in this experiment's normalizers, training or scoring (whole publisher Parquet file downloaded, which contains their bytes). At each position input only actual state x[t], prior logged action u[t-1], and genuine previous state x[t-1]; output next observed x[t+1]. Source action is logged evidence, NOT established physical `do(action)`. Secondary measured label: actual next-motion above FIT-only p90 — **HIGH-MOTION PROXY, NOT injury/accident/adverse event**.
- Both new models were initialized from scratch and trained end-to-end: three seeds (27001,27002,27003), 192 optimizer updates × batch8 × 32 frames per arm/seed, identical sampled FIT cases, task/loss/optimizer; TAC **95,907** vs GRU **95,869** trained/deployed parameters (0.040% difference). Both use identical source-conditional constant-velocity anchor and `0.1 × bounded learned innovation`. Original S1MOTION001 candy FAIL preserved, no old checkpoints touched. **Complete training FLOP parity is not proven**; source reports a partial PyTorch operator subset only, so architecture comparisons are exploratory rather than fully compute-matched scientific wins.

## Actual first DEV scoring: 14-joint next-frame physical MSE (lower is better)

**Exactly 2,550 authentic next-frame outcomes from 10 episodes per model/seed**, not 2,550 independent experimental clusters.

| Original frozen seed | Constant velocity | Velocity-anchored TAC System1 | Same-velocity GRU | Persistence |
|:--|--:|--:|--:|--:|
| 27001 | **2.9410397e-6** | 3.0490341e-6 | 2.9257204e-6 | 4.1481335e-5 |
| 27002 | **2.9410397e-6** | 3.1469311e-6 | 2.9220689e-6 | 4.1481335e-5 |
| 27003 | **2.9410397e-6** | 3.0189166e-6 | 2.9188605e-6 | 4.1481335e-5 |
| 3-seed mean | **2.9410397e-6** | 3.0716273e-6 | 2.9222166e-6 | 4.1481335e-5 |

TAC model **4.44% worse than velocity**, and **5.11% worse than the parameter-matched, velocity-anchored GRU**, using the original three seed means. GRU beats velocity by only **0.64%** — below the original required ≥1% improvement even if the method had been TAC. TAC high-motion-subset MSE vs velocity worsened by **+6.40%, +10.86%, +1.94%** across seeds respectively, exceeding the 5% high-motion harm gate in the first two seeds. **Scientific gate FAIL all three, not close to success.**

## Measured high-motion-proxy event ranking

True measured high-motion event prevalence on DEV: **10.196%**. Scores from the *secondary classifier* are never called actual risk calibration.

| Seed | TAC AUPRC | GRU AUPRC | TAC AUROC | GRU AUROC |
|:--|--:|--:|--:|--:|
| 27001 | 0.31370 | **0.62186** | 0.84393 | **0.90631** |
| 27002 | 0.29656 | **0.62928** | 0.80735 | **0.90611** |
| 27003 | 0.30393 | **0.60541** | 0.86931 | **0.90162** |
| Mean | 0.30473 | **0.61885** | 0.84020 | **0.90468** |

The model learned some event-ranking signal, but the GRU is substantially better on the proxy, with similar parameter counts and training exposures. Report Brier, per-episode and full precision in original artifacts.

## Diagnosis and NEXT closed decisions
- **What worked:** Observed-velocity prior massively outperforms the old persistence-based source signal and is causally available; the parallel/stream core engineering tests pass. That is a genuine architectural engineering improvement in baseline formulation, NOT differentiated learned TAC skill.
- **What failed:** TAC's learned correction degrades strong velocity reference, is weaker than GRU neural correction, and breaks original high-motion robustness; its risk head also loses relative to GRU. 0/3 gate binding despite both beating persistence.
- The new bounded `lambda=.1` correction and joint point/BCE/uncertainty objectives may compete for scarce native feature capacity, but these are **hypotheses**, NOT measured proven causes. No retrospective tuning to the opened DEV or protected original TEST. Check previously frozen models' FIT-only gradient objectives and causal correction alignment in a read-only follow-up and register those results separately.
- Pending: independent robot/task generalization, actual adverse/hazard labels and decision utility, source-level verified real action/timing, equal *full* training compute, onboard CPU/device latency, and complete PNDS exact/FP/AD neural graph proof. No production safety claim or seventh model.

**Original negative research preserved**; future successor requires a **fresh preregistered real training/eval split and fully matched or transparently unequal compute**. No re-fitting the original on its scored DEV.
