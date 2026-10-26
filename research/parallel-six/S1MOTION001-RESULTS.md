# S1MOTION001 — ORIGINAL authentic robot System1 capability results

**2026-10-26 · completed original FIT / first DEV scores / independent SHA-pinned cold replay · SCIENTIFIC GATE FAIL 0/3.**
Source PR: #415 (draft/unmerged). Exact preregistration: [S1MOTION001-PROTOCOL.md](S1MOTION001-PROTOCOL.md). Original machine-readable frozen checkpoint and source hashes: [S1MOTION001-ORIGINAL-LOG-LEDGER.json](S1MOTION001-ORIGINAL-LOG-LEDGER.json).

## Immutable experiment and audit receipts
- Original training/checkpoint+DEV GitHub Actions [run #40274339695](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40274339695) at commit `005695ffe71b49d3727b17cc1ae0086782522ba7`. Engineering tests passed. All six 95,907/95,869-parameter models (three seeds each arm) received **192 equal optimizer steps, 8 source windows/step, 32 real frames/window**. All six original model+optimizer+RNG records frozen before first DEV scores; original artifact [12629442926](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40274339695/artifacts/12629442926) uploaded (90-day GitHub retention). Original source pinned to real MIT-licensed ALOHA static candy snapshot `6728831ef9674b74f3ce20bf517890e9ee20b0d9`; complete publisher parquet source SHA recorded in original before-score JSON.
- **Original run CI concluded failure AFTER all six models fitted and scored** due to final JSON assembly using duplicate `fit_records` argument. Do not present this first CI as success or erase failure. The complete six original score records and checkpoint SHA were printed in the original logs before the failure; per-seed evidence also registered in the immutable-priority log ledger. Later packaging source corrected without retraining.
- Recovery **read-only** [run #40274837234](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40274837234) PASSED. It downloaded *the original* six protected checkpoint payloads, verified each SHA and retained optimizer/RNG/source manifest, reloaded the same real publisher snapshot, recomputed FIT-only scaling and the entire original DEV, checked all original MSE/AUPRC/AUROC/Brier results against the original logged values, and archived complete per-episode scores and single-thread CPU stream timings in [replay artifact 12629224120](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40274837234/artifacts/12629224120) (90-day retention). **NO NEW OPTIMIZER STEP**; original training and failure remain visible. This is not a new independent source study.

## Raw original DEV results: 14D next joint-state MSE (lower is better)

Same **2,550 authentic future frames from ten DEV episodes** for every arm and seed. Train: episodes0–24; DEV25–34; episodes35–49 excluded from training and this scoring. Training/DEV are separate episodes of the same publisher robot, NOT independent actors/sites. Historical EWA002 already scored a subset of this source, so do not call it unseen transfer.

| Original seed | TAC System1 95,907 params | GRU 95,869 params | Persistence | Constant velocity |
|:---|---:|---:|---:|---:|
| 26001 | 0.00002400438518 | 0.00002064640703 | 0.00003507450318 | **0.00000305908035** |
| 26002 | 0.00002999342057 | 0.00001959884245 | 0.00003507450318 | **0.00000305908035** |
| 26003 | 0.00002089951320 | 0.00001748869413 | 0.00003507450318 | **0.00000305908035** |
| Mean across seeds | **0.00002496577298** | **0.00001924464787** | 0.00003507450318 | **0.00000305908035** |

TAC's 3-seed mean MSE is **8.16× constant velocity** and **29.73% worse than the GRU mean**, despite 0.040% parameter-count parity. Both learned models beat persistence but failed against the physically informed velocity reference. It is NOT legitimate to claim System1 world-state forecasting superiority from persistence gains.

### Measured high-motion-proxy classification (NOT hazard/failure identification)
High joint motion: actual mean squared next motion exceeding the 90th-percentile threshold fit **only** on train source; DEV prevalence **8.67%**. The inherited `adverse_event_logit` is used here exclusively as an observational high-motion-proxy score.

| Seed | System1 AUPRC | GRU AUPRC | System1 AUROC | GRU AUROC |
|:---|---:|---:|---:|---:|
| 26001 | 0.25510 | **0.46767** | 0.78428 | **0.86765** |
| 26002 | 0.20807 | **0.46471** | 0.74126 | **0.86332** |
| 26003 | 0.21992 | **0.34353** | 0.73926 | **0.85811** |
| Mean | 0.22770 | **0.42530** | 0.75493 | **0.86303** |

The model detected some informative signal above event prevalence, but GRU had better ranking across all three original seeds. System1 high-motion-regime MSE also missed the velocity robustness threshold by a wide margin. Thus **0/3 preregistered predictive gates passed**, not a near-pass.

## Scientific limitations that remain binding
1. **Original CI failure is documented; cold replay success does not rewrite its original status.** Recovery was independent of optimizer training, but it is the same source, trained checkpoints and originally exposed DEV observations.
2. Train exposure and parameters were closely matched, but the Torch matmul profiler is an **incomplete forward/backward FLOP subset**. Equal full training compute was not proven. Matched-param, matched-batch/steps comparison does not qualify as fully compute-matched head-to-head architecture superiority. CPU fit and replay latency are diagnostic, not device SLA.
3. S1MOTION001 trained a normalized state-change corrective decoder centered at persistence. It does **not** expose an explicit velocity extrapolation prior to its neural point head, even though constant velocity is a strong legitimate causal predictor based on earlier states. This is the next architectural bottleneck to falsify, not a justification for changing frozen original scores.
4. The label is observed high movement, not verified bad outcome, task success, accident risk, policy value or causal `do(action)` intervention. There are no genuine safety labels, autonomous controller, verified commands, hardware latency/safety certificate, complete source-level AD/IEEE/Lean proofs or independent external gains.
5. Exact floating-state live memory `88+64=152` FP32 scalars (608 B) **excludes** weights, time/identity/timestamps, normalization storage and runtime. Historical negative native/Physical/Enterprise variant claims, and PNDS 6-family registry, remain unchanged.

## Forward architecture direction (not a tested result)
**S1MOTION002 hypothesis:** an observed-history velocity prior plus a bounded, learned, causally retrieved innovation may be superior to an unstructured persistence-centered prediction head; preserve typed source/clock authority and read-before-write scan, count extra last-observation memory and external weights, and compare against the *same velocity base* for all neural controls. No new tuning or SCORE of episodes35–49 is authorized by this result. Preregister new FIT/DEV data and compute parity before prospective model promotion.
