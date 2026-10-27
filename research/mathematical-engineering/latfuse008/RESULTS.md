# LATFUSE008 original real-source historical query experiment

**October 27, 2026 — genuine-source exploratory training COMPLETE and independently replayed. NO qualified historical recall or architecture-superiority claim.**

## Original provenance and scientific gates

[Registered nofit CPU source, gradient, exact state and matched query control CI #40446863349](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40446863349): PASS. Candidate and nonassociative control both carry exactly 1,696 FP32 values + 16 occupancy flags (6,800 bytes), and exactly 99,390 actively trained/deployed parameters. 43 parameter tensors each received finite source gradients. Noassoc historical query logits are unchanged when old associative values are altered; LATFUSE007 historical query responds to those values. Shared source writer and original six-hour forecasting path are unchanged at matched initial weights. Actual CPU profiled supported forward/backward FLOPs 209,663,861 (associative) vs 209,666,037 (fast/slow control), difference **0.00104%**. Unsupported operators and optimizer work are not fully accounted. This is NOT yet the compulsory verified equal-*complete-training-FLOPs* scientific comparison.

[Two seeds × two models original real-source training, model/optimizer/RNG checkpoint preservation and clean-checkout replay #40447445348](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40447445348): **PASS**. Exactly 128 total original optimizer steps, 32 per seed/model. Two seeds 80701 and 80702; identical station-hour sample order, observed true physical event target/forecast target, optimizer policy, 88 actual prior observed hourly updates and 100k-parameter budget. Original physical UCI501 monitored Beijing source ZIP SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. March–April2014 source train period, May–June2014 internal evaluation. True HIGH event original PM2.5 >=150 µg/m³; ordinary <=35 µg/m³, matched original source station/hour-of-day, at least 24h separated. 96 genuine source high/ordinary training pairs and 24 observed evaluation pairs. Source FIT normalizer trained only on earlier historical 2013March–2014March raw observations (not any later source anchors). Event class and real+6h outcomes are consumed for supervision only AFTER model prediction; no event label passed as known historical clock query. No original 2015 DEV or 2016–17 numeric protected data accessed.

All four trained model full FP32 weights, AdamW optimizer/RNG, complete 32-step losses, exact original source/case-order/code hashes, source normalizer and genuine-evaluation arrays saved in research/replay-bundles/LATFUSE008. Independent clean Git checkout restored original checkpoints and **bitwise exact** prediction arrays and metrics without optimizer. No retraining/restarter selection after seeing scores. The May–June data are internal FIT-era source and previously encountered generally in research, not an independent untouched test.

## Actual genuine May–June2014 binary historical event paired ranking (TWO seed mean; higher better, neutral=50%)

| Delay since event | Original LATFUSE007 associative historic query | Same-params current fast/slow-only historic query |
|---|---:|---:|
| 16h | 22.92% | 41.67% |
| 64h (trained delay) | 27.08% | **56.25%** |
| 128h | 27.08% | 31.25% |
| 512h | 22.92% | 45.83% |

**Historic true-event retrieval FAILS.** The native associative read ranks genuinely high physical PM2.5 events worse than same-source/hour ordinary observations, worse than the nonassoc query in both seeds' average across all horizons. Tiny 24-pair evaluation means no population-level causal/generalization claim.

## Binary probability Brier (lower better; 50% fixed-prior=0.25)

| Delay | Associative historical query | Fast/slow historical control |
|---|---:|---:|
| 16h | 0.31738 | 0.33567 |
| 64h | 0.32213 | 0.33388 |
| 128h | 0.34185 | 0.41672 |
| 512h | 0.32279 | 0.33772 |

Associative query modestly lowers mean Brier versus noassoc on this small internal sample, but both are **worse than a constant 0.25 neutral probability** at EVERY length; probability calibration and source-factual prediction are not successful. Pair ranking and Brier measure different things and must be reported together.

## Original source +6h all-channel normalized MAE (lower better)

| Delay | Associative historic query model | Matched noassoc historic query control | True last-observation persistence |
|---:|---:|---:|---:|
| 16h | 0.71075 | 0.73295 | **0.55966** |
| 64h | 0.67350 | 0.69646 | **0.53877** |
| 128h | 0.67221 | 0.67805 | **0.57044** |
| 512h | 0.78382 | 0.80491 | **0.67644** |

Associative model modestly improves descriptive all-channel point forecast over the alternative history query condition, but **both are worse than practical persistence across all horizons**. Both original forecast heads still access the same associative memory, so these are not pure associative-memory forecasting ablations.

## Diagnosis and next falsifier

Previous LATFUSE006 external full-matrix sidecar at64h ranked 73.7% on a DIFFERENT real-source normalized event task, with external fitted coefficients and pretrained parent FIT overlap; it is not a directly comparable native retrieval success. The native LATFUSE007 additional 16D read head and cyclic past clock6 representation alone did not yield useful event-specific addressing. Trained only32 updates, thus undertraining is plausible. LATFUSE005 established key-directed overwrite substantially attenuates original prefix memory after long genuine source delays. It is not scientifically valid to immediately blame either insufficient readout or overwriting without a new read-only gradient/write-use attribution.

The next source-safe zero-update diagnostic should measure source/event-anchored derivative of original historic-event prediction loss with respect to associative memory at event time and query time, and the fast/slow pathway, at H16/64/128/512. This would separate a *missing task-gradient signal at old writes* from *current readout unable to use existing state*; the source labels must remain outcome-only and model checkpoints unchanged. Any further neural model changes (explicit source lag input, supervised write targets, selective overwrite) require independent new original-data preregistration and complete <=1% full-training-FLOPs/parameter budget before any qualified architecture comparison. No protected source opened, drafts unmerged, no outcome-based resampling/tuning.

**Decision:** original two-model query-only hypothesis is **not supported**; preserve the negative result and prioritize actual source-anchored gradient/retrieval-path diagnosis before another fitted variant.
