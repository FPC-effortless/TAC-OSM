# LATFUSE004 — original real-data 2×2 independent head-width / routing study

**October 27, 2026.** Implemented, source-preflighted, all EIGHT original real-data models genuinely trained and original checkpoints independently clean-replayed. Architectural scientific qualification and full equal-training-compute parity are NOT established. No original LATFUSE002/003 weights changed or drafts merged.

## Source and audited runs

- Original licensed UCI501 Beijing Multi-Site Air Quality 12 physically recorded stations, ZIP SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.
- [Nofit PyTorch CI #40439588574](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40439588574) PASS: 40 focused tests, all39 trained-model gradient tensors per arm finite on authentic original FIT observations, bitwise no-source head isolation and exact original LATFUSE003 four-head hard forward/read parity. An earlier nofit failed on a stale live test count, corrected only the current-count reference (original frozen327 untouched).
- [Eight original trained real-source checkpoint and clean checkout replay CI #40439922115](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40439922115) PASS. Seeds40401 and40402, four variants each ×128 AdamW optimizer steps, batch8 ×24 genuinely recorded prior observations per case, six-hour original measured future target read ONLY after model forecast. EXACT original station/hour training sample ID order for all four arms within each seed, identical loss/optimizer schedule and hyperparameters.
- FIT source 2013-Mar through2014-Aug, 2014-Sep–Dec chronological validation, previously exposed 2015 DEV numerical targets parsed only AFTER all original eight checkpoint SHA bytes and FIT receipts persisted, original 2016–2017 protected numerical source NEVER parsed.
- All8 FP32 checkpoint weights, AdamW optimizer/RNG, full loss history, training-order/source/code SHA256, FIT normalizer, validation scores, 2015 exposed DEV prediction arrays and per-channel calibration in research/replay-bundles/LATFUSE004, restored and bitwise replayed in separate clean Git checkout without refitting. No extra seed selection or hidden fit.
- [Read-only source-authentic original-trained head audit CI #40440201998](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40440201998) PASS on real June2014 FIT 12 stations×256 observed hourly inputs. No optimizer, target labels, exposed2015 numerical scores or protected rows used. Results in TRAINED_HEAD_DIAGNOSTIC.json.
- [PNDS mathematical operator CI #40440079419](https://github.com/FPC-effortless/pnds-lean/actions/runs/40440079419) PASS; source-scoped local exact-real theorems only. Full PyTorch source/AD/IEEE/probability refinement remains OPEN.

## Factorial intervention and invariant state

| Arm | Gated delta independent memory partition | Source write routing | Actual learned params | Carried FP32 bytes |
|---|---|---|---:|---:|
| 4h-hard | 4×(16value×4key) | hard top2 | 99,747 | 6,800 |
| 4h-soft | 4×(16value×4key) | softmax all4 | 99,747 | 6,800 |
| 2h-hard | 2×(16value×8key) | hard top1 | 99,261 | 6,800 |
| 2h-soft | 2×(16value×8key) | softmax both | 99,261 | 6,800 |

All four retain 256 FP32 associative matrix cells, fast80, slow80 and 16 original source slots x80 = 1696 FP32 recurrent scalars plus16 bool flags =6800 nominal bytes per independent stream. No extra recurrent history/KV cache. All within mandatory 95k–105k live parameter band, max between-arm parameter gap0.4872% (PASS <=1%).

Common FIT loss across ALL arms: 0.8 source-masked normalized Gaussian NLL+0.2 normalized absolute error +0.25 observed source-sensor-distinct within-head key cosine hinge +0.03 soft route-usage balance; only future-known calendar conditions the forecast and is NOT an intervention/action. No retrospective 2015 hyperparameter tuning.

**Compulsory resource integrity:** exact observed-input PyTorch CPU supported-op forward+backward prefit profile:
4h-hard 62,072,899 FLOPs; 4h-soft 62,072,323;
2h-hard 61,376,965; 2h-soft 61,376,965.
Maximum gap **1.12116%**, above the compulsory <=1% requirement. Optimizer and some operators not profiled, full actual FLOPs equality UNVERIFIED. All outcomes descriptive only; NOT fair architecture-comparison success. Do not pad useless FLOPs, silently change optimizer/sample counts or claim superiority.

**Subfactor caveat:** Hard/soft changes head selection and memory retention eligibility together. Hard keeps unselected heads bit-exactly (alpha=1,beta=0); soft all observed heads receive their retention and share differentiable weighted writes. This experiment separately varies routing style and key width within a 2×2 design, but DOES NOT fully isolate router gradient flow from different forgetting dynamics; a matched-decay-law diagnostic is the next falsifier. Two-head hard top1 and four-head hard top2 keep half the heads active per observation.

## Real training results — exposed 2015 development, 312 station-hour original forecasts and 4,024 measured eligible target channels

| Arm, TWO seed mean | PM2.5 MAE ↓ | All-channel normalized MAE ↓ | Gaussian NLL ↓ | Pooled nominal90 coverage |
|---|---:|---:|---:|---:|
| 4h-hard | 26.5194 | 0.585473 | 1.9810 | 89.89% |
| 4h-soft | 27.4138 | 0.584407 | 2.2711 | 89.75% |
| 2h-hard | 27.3913 | 0.588443 | 1.9524 | 89.82% |
| 2h-soft | 27.7035 | 0.590251 | 2.0521 | 89.43% |
| practical LAST-OBSERVATION persistence | 33.3725 | **0.559802** | n/a | n/a |

All four improve original PM2.5 physical MAE against an UNMATCHED PRACTICAL reference on exposed DEV, but **all four LOSE to persistence on average overall normalized MAE**. Between-seed variability matters: four-head soft PM2.5 MAE23.80 in seed40401 but31.03 in seed40402, while four-head hard24.15 and28.89. No consistent soft or larger-head improvement.

Chronological 2014 validation 108 authentic station-hour origins/1392 measured target channels, TWO-seed all-channel normalized MAE: 4h-hard0.581735, 4h-soft0.585762, 2h-hard0.581540, 2h-soft0.583796, versus practical persistence **0.475008**. ALL four lose to this practical reference on validation too. Original fit only128 updates per seed; undertraining is possible, but extra steps/new fits are NOT permitted under exhausted original registration.

## Real-source routing and within-head geometry — same checkpoint, zero new training

Original trained June2014 FIT input history of256 hourly steps across12 distinct original monitoring sources. Normalized write entropy: 1 means evenly spread probabilities within sample, 0 means deterministic one-head write. Hard two-head is by construction top1 and has normalized write entropy0; entropy alone is not a skill test.

| Original seed | Arm | Mean normalized write entropy | Mean observed WRITE head mass |
|---|---|---:|---|
| 40401 | 4h-hard | 0.439 | [0.649,0.117,0.175,0.059] |
| 40401 | 4h-soft | **0.846** | [0.499,0.167,0.206,0.128] |
| 40401 | 2h-hard | 0.000 | [0.673,0.327] |
| 40401 | 2h-soft | **0.969** | [0.552,0.448] |
| 40402 | 4h-hard | 0.479 | [0.232,0.008,0.567,0.192] |
| 40402 | 4h-soft | **0.907** | [0.225,0.100,0.382,0.293] |
| 40402 | 2h-hard | 0.000 | [0.764,0.236] |
| 40402 | 2h-soft | **0.980** | [0.524,0.476] |

**Finding A:** Soft writes broadly spread updates and eliminate the HARD zero-write gradient eligibility for original observed heads, greatly increasing entropy. Yet the decoder's learned **read** weights remain seed-sensitive/concentrated; e.g. 4h-soft reader mean weights seed40401 [0.756,0.077,0.091,0.075], seed40402 [0.116,0.067,0.293,0.524]. Broad writing alone does not establish multiple useful perspectives.

Genuine historical SAME source sensor-distinct pair means >=3 jointly observed original normalized physical variables with RMS source difference>=1, head key absolute cosine >0.9 considered potential collision. Original route-weighted per-head fractions from frozen original FIT outputs:
- Seed40401 4h-hard [13.7%,9.0%,14.9%,14.1%], 4h-soft [11.4%,13.8%,6.5%,5.4%], 2h-hard [3.1%,6.6%], 2h-soft [2.5%,3.9%].
- Seed40402 4h-hard [8.4%,unmeasurable (head nearly unused),6.0%,7.5%], 4h-soft [9.0%,8.1%,8.6%,8.6%], 2h-hard [0.5%,3.5%], 2h-soft [1.0%,1.0%].

**Finding B:** 8D-head key spaces show fewer near-parallel source-different vectors than 4D spaces. But normalized 8D random vectors would ALSO be less likely to exhibit very high cosine; this geometric proxy does NOT establish more accurate retrieval of actual old events or better long-horizon forecasting.

## Calibration and uncertainty: separate failure

Pooled90 coverage near90 is NOT per-channel proper calibration. Original exposed 2015 per-channel arrays and metrics are preserved per trained arm/seed. Rainfall has high/unstable Gaussian NLL (for seed40401, RAIN NLL3.76 in 4h-hard and4.81 in 4h-soft); source may be zero-inflated and poorly represented by a single normal law. Seed40401 2h-soft actual 90% channel coverage: TEMP79.17%, PM1082.20%, PM2.583.66%. Important individual features remain undercovered, despite pooled coverage near target. Model is unqualified for risk-sensitive control.

## Research conclusion and next falsifiers

**Both user-selected changes have been implemented AND genuinely trained separately and together** while preserving the original fixed-size recurrent state. Soft routing improved write allocation balance (not readout specialization or forecasting). Wider 8D keys lowered within-head angular collisions (not proven factual retrieval or forecasting). Their combination did not improve aggregate 2014/2015 forecast errors. Neither concept is a qualified architectural solution at the current budget or evidence tier.

1. Run genuine source-authored delayed-event retrieval, source-time/missingness relevance, H16/H64/H128/H512, with original authorizations and independent source-truth labels. Do not use synthetic-only recall to claim real abilities or tune on exposed 2015.
2. Isolate fixed-erase-law routing effects versus gradient availability and how read weights respond; preserve original checkpoint bytes, zero fits for diagnosis.
3. Separate per-channel/zero-inflated rainfall and tail calibration, source-era heldout proper scoring. Do not assume memory fixes repair uncertainty.
4. If justified, register matched real source two-seed factorial with <=1% actual full optimizer/training FLOPs and <=1% active params, plus original source/split/sample/optimizer equality and rigorous postfit replay. Do not use these unequal-FLOP outcomes to select a winner.
5. Full end-to-end Lean/Python source, automatic differentiation, FP32 and probability-law refinement remains OPEN. PNDS local exact-real head-count/soft-blend math CI does NOT prove predictive intelligence.

All original LATFUSE002/003 and new LATFUSE004 trained checkpoint families preserved on separate unmerged branches. No protected 2016–2017 original numerical source read or evaluated.
