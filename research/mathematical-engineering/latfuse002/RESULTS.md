# LATFUSE002 — original real-data training results (2026-10-26)

**Original real FIT training COMPLETE; Git-owned original checkpoints + independent clean-checkout score replay PASS. Full scientific qualification NOT ESTABLISHED.**

Study: LATFUSE002-REAL-PREDICTION-SINGLE, original UCI501 Beijing 12 original monitoring stations (CC-BY-4.0). Original source SHA256: d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.

Actual training/replay CI: https://github.com/FPC-effortless/TAC-OSM/actions/runs/40417659771 (SUCCESS).
Standalone source-only preflight: https://github.com/FPC-effortless/TAC-OSM/actions/runs/40417376452 (SUCCESS).
Companion PNDS exact-real local operator proof CI: https://github.com/FPC-effortless/pnds-lean/actions/runs/40417081201 (SUCCESS, full-model proof OPEN).

## User-requested state invariant and repairs

LATFUSE002 changes learnable initial matrix retention to sigmoid(5.2933)=0.995; initial write sigmoid(-2.1972)=0.1; uses multiple fixed slow retention rates spanning0.9..0.99999 and belief-conditioned keys for content retrieval. This keeps **fast80 + slow80 + 16x16 matrix + 16x80 entities = 1,696 FP32 recurrent scalars** and16 Boolean flags = **6,800 nominal bytes per stream**, *exactly the widened LATFUSE001 H80 state size*. Auxiliary fixed time-rate buffer is model-level static, not per-stream recurrent memory. Live model-size = **98,775 FP32 trained/deployed weights**, both actual seeds, compulsory 95k–105k PASS.

No additional recurrent-state tensor; fixed size is NOT evidence of improved retrieval, mobile readiness or lower operation count. The larger contextual-query projection is trained but is a hypothesis, not a guaranteed repair of key collision.

## Immutable original real-data fit and evaluation

Exactly **2 seeds 202601/202602 x128 original AdamW optimizer steps x8 actual real station-hour cases each**, history24, six-hour source-observed forecast. Source FIT normalization restricted to original station observations prior2014-09-01; chronological validation 2014-09 through 2014-12; numerical 2015 DEV first read after BOTH originals weights frozen; protected 2016–17 numeric rows NEVER parsed. Calendar features at target hour are deterministic future-known exogenous conditioning, not physical actions or do-operator counterfactuals.

Every original trained model, optimizer/RNG, original normalization and sample-order hash, exact SHA checkpoint, score and prediction arrays are preserved under research/replay-bundles/LATFUSE002. All originals replay bit-identically from clean Git checkout without fitting; prefit/actual postfit parity validator PASS. Original SHA:
- seed202601 final.pt 7b05fbd3226167b4f31b81cf3d235fd0dbbc2a33637957df6da3f872f70d215d
- seed202602 final.pt 5fd3e162653ccdc23ca9bb343e5ab4262c149953e00b548e5e80e0eedd30f0a0

## Genuine original 2015 exposed-DEV results — 312 station-hour queries, 4,024 valid observed target channels

| Metric | Seed202601 | Seed202602 | Persistence |
|---|---:|---:|---:|
| Six-hour raw PM2.5 MAE | **29.5365** | **26.1773** | 33.3725 |
| All-channel normalized MAE | 0.58743 | 0.54480 | 0.55980 |
| Normalized Gaussian NLL | **5.82957** | **2.90826** | unavailable |
| Nominal 90% all-channel coverage | 87.13% | 90.63% | unavailable |
| First/last sampled FIT minibatch loss | 1.4577 / **11.9571** | 1.5510 / 0.7094 | unavailable |

PM2.5 MAE decreased11.49% and21.56% versus the *unmatched practical last-observation* reference on already-exposed 2015 DEV. The model was NOT compared under equal parameter/data/compute conditions against trained native, GRU, SSM or attention controls. **Do not claim architectural superiority.** Seed202601 worsened aggregate normalized MAE, exhibited subnominal interval coverage and very high NLL. Seed202602 had modest normalized MAE gain but unproved calibration. Sparse development data and correlated station-time queries are not independent qualification.

Internal2014 heldout-from-this-fit validation (108 chronological queries, 1,392 valid target channels): PM2.5 MAE 42.7438 / 30.3671 versus persistence33.1038; all-channel normalized MAE0.57275 /0.52069 versus persistence0.47501; pooled coverage88.29%/91.59%. Both seeds lost overall normalized MAE to practical persistence in internal validation. First seed also lost PM2.5. Thus results **MIXED_TRADEOFF / FAIL scientific promotion**, even though actual trained checkpoints and Python engineering gates replay.

Seed202601 final source training minibatch loss spiked to11.957 from first1.458, but one sampled minibatch alone cannot distinguish outliers, genuine source shift, optimization instability or bad gradients. No rescue training run or hidden seed selection is authorized by the original registration; no protected numeric TEST access occurred.

## Source and anti-cheat honesty

The previous initial no-fit source preflights rejected an outdated test-count provenance and a missing valid SSE original wind category, respectively; both failures were preserved, corrected and revalidated before ANY optimizer update. Current project test suite and numerical source boundary were PASS on scoped original input tests. End-to-end independently instrumented leakage control, reward/outcome verification, long-history associative retention correctness, per-regime calibration, probability-law/IEEE/whole-model formal proof and meaningful causal intervention decisions remain OPEN/INCONCLUSIVE, not automatically PASS.

## Next falsifiers, no further fit under this registration

Measure trained checkpoint alpha/beta gates, key effective rank, real FIT time-lag memory probes, state-lesion output sensitivity, original specific per-channel DEV NLL/coverage/outlier errors, and readout dependence on contextual query versus recorded future-known calendar, using READ-ONLY original weights. Do not explain the seed202601 NLL spike without evidence. Any new corrective model must have a new source-pinned preregistered study and genuine comparison budgets before any fit.

**Return to original native TAC latent-space diagnosis** in https://github.com/FPC-effortless/TAC-OSM/issues/444 when this exploratory candidate has been honestly assessed. LATFUSE002 is not native TAC, not a qualified controller and does not change historical ME110 FAIL0/5 or any checkpoint. Drafts remain unmerged.


## Subsequent source-authentic original-trained memory diagnosis (READ ONLY)

[GitHub Actions #40418497414](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40418497414) **PASS**. Actual SHA-verified original seed202601 and202602 trained checkpoints were restored; genuine, chronological original source 2014-06 256 FIT hourly observations ×12 sites was used. **No optimizer step, no checkpoint mutation, no held-out scores or 2015–17 numeric target use**.

| Frozen FIT-only local property | Seed202601 | Seed202602 |
|---|---:|---:|
| Learned retention gate alpha mean | **0.99768448** | **0.99888873** |
| Learned write gate beta mean | 0.15818049 | 0.28782359 |
| Mean recent 32-key absolute off-diagonal cosine similarity | **0.8275125** | **0.87052739** |
| Approximate associative matrix rank (SVD 0.1% max cutoff), mean | 14.83/16 | 13.75/16 |
| Mean absolute forecast change if associative matrix zeroed | 0.03854 | 0.08538 |
| Corresponding fast-state zero-lesion change | 0.39888 | 0.33428 |
| Corresponding slow-state zero-lesion change | 0.09708 | 0.16631 |
| Corresponding entity-slot zero-lesion change | 0.13082 | 0.21155 |

**What has and has not been fixed:** The initial random gated retention measured ~0.278 for LATFUSE001 was repaired: original LATFUSE002 models now exhibit trained retention near0.998 on *these FIT source inputs*, with no recurrent state increase. The matrix is active and its lesion affects forecasts, but highly aligned/aliased write keys reveal an OPEN addressing/interference problem. SVD rank near16 is not evidence of precise individual-pair recall, and lesion output sensitivity is not a causal world-state attribution or proof of utility. The recent-key similarity is a *local trained checkpoint observed fact*, not a global theorem for all sources or long lags.

**Next work before another fit:** instrument source-time-key labels and genuine distinct historical event retrieval, assess key representation collapse, measure long-delay survival when similar events intervene, and evaluate whether source-authorized key-decorrelation/admission or partitioned addressing is a defensible remedial mechanism. Independently register any changed graph as a NEW variant, same 6,800 FP32 recurrent bytes, genuine source/matched resource preflight and untouched evaluation. This original LATFUSE002 single-model registered training budget is exhausted and must not be silently retried.
