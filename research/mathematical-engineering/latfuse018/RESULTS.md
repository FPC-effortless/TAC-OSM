# LATFUSE018 — actual trained genuine-source content-history B0, matched original 2×2, exploratory results

**2026-10-27 — original real physically observed UCI501 source-trained and independently replayable, not a qualified real-world architecture superiority demonstration.**

[Original complete source-trained + independent clean-checkout GitHub CI #40467840405 PASS](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40467840405). [Source-first, PRE-FIT immutable genuine true station/time sample plan](https://github.com/FPC-effortless/TAC-OSM/blob/research/latfuse018-real-source-content-training-2026-10-27/research/mathematical-engineering/latfuse018/PLAN.json) separately committed and pushed **before optimizer started**. [Original training registration](https://github.com/FPC-effortless/TAC-OSM/blob/research/latfuse018-real-source-content-training-2026-10-27/research/mathematical-engineering/latfuse018/REGISTERED_REAL_FIT.json) committed before workflow, authentic original SHA-pinned sensor archive \`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8\`. Source data train original March–April2014 only, no numerical original exposed2015 or protected2016–17. **Prior May–June2014 source evaluated repeatedly before this experiment**, so all following true-source development results are *exploratory, NOT a virgin model validation*, despite real physical outcomes.

## Fairness/compute and replay

- Four newly trained models: original SourceTimedEventQuery with real associative past-clock read, versus previous SourceTimedNonassociativeQuery with fast/slow-only historical clock read. Two fixed seeds18018 and18019. Both model arms have exactly **99,390 actually trained/deployed parameters**, no dummy weights, and original streaming **1696 FP32 states+16 bool flags=6800 nominal bytes**. Both retain identical original forecast backbone including matrix read.
- Registered identical **64 AdamW optimizer updates** each, batch16, original 24 prior hours+64 observed continuation physical source and +6 true physical forecast, identical source normalized PM2.5 smoothL1 physical historic content + original all-channel masked forecast loss, LR.0005, weight_decay.001, maxgrad1, foreach=False/fused=False.
- Original source pre-fit plan **1,137 truly observed eligible train site/hour cases** (March1–April18, station/12h grid) with exactly SHA-committed RNG sample order per seed. Original May1–Jun10 eligible development **117 physically observed cases** (12 station/96h grid). No future source target value inserted into source past clock or original oracle write-key. True historic PM target has already passed through legitimate physical sensor state at its historical step; supervised historical and future labels read ONLY after neural query/forecast output.
- Matched original CPU FP32 H64 batch16 88-event input full forward+backward+gradient clipping+AdamW mathematical operator parity proof: [LATFUSE017 CI #40466771608 PASS](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40466771608), conservative FLOP upper discrepancy **0.00649%**, identical matmul multiply-add FLOPs, all non-differing transcendental/operator shapes identical. Not hardware/CUDA or other batch/sequence proof; CPU wall-time NOT a fairness condition.
- All trained models saved original full source model state_dict, AdamW state, torch CPU RNG, exact source order/data/program SHA, all 64 training losses, initial/final trained parameter budget and source-time legal evaluation results. Exact trained source predictions H16/64/128/512 replayed once before commit and **again from independent clean checkout** in same CI, SHA strict; original LATFUSE002–017 trained models unchanged.

### Main true-source developmental H64 historical-content result

| Real original physical PM2.5 log1p MAE (lower better) | Seed18018 | Seed18019 |
|---|---:|---:|
| Genuine trained associative past-clock query | **0.42555** | **0.35959** |
| Matched source-trained fastslow-only historic query | 0.44569 | 0.79705 |
| Original source before-March observed log1p train prior (zero standardized) | ~0.394 | ~0.394 |
| Genuine trained assoc model historic-query matrix ZEROED, same state | ~0.385 | ~0.385 |
| Query-time last observed PM2.5 persistence (an imperfect historical comparator) | 1.22440 | 1.22440 |

Genuine physically measured full-source PM2.5 logged reconstruction **not only event class ranking**. Assoc beats trained fastslow control in both original seeds, BUT **assoc is worse than its OWN zero-read query in seed18018** and worse than the source prior (~0.394) there; seed18019 assoc improves 0.3596 versus zeroed0.385, a modest **~6.6% normalized MAE reduction**. Exact normalized historical PM2.5 MAE assoc0.39007/0.32960 versus original zero-read0.35291/0.35285 and train-prior0.36132. **Strict incremental useful persistent memory gate FAILS 1/2 original seeds**. Previously evaluated source does not permit any independent qualification even if both had won.

Training original logged total objective initial→last: assoc seed18018 **1.6073→0.7102**, nonassoc **1.6142→0.8458**; assoc seed18019 **1.9115→0.8933**, nonassoc **1.8988→0.9764**. Last16 average objective separately recorded in original RESULTS.json. Decrease demonstrates optimizer learns on training source, NOT independent useful memory. All four models underwent exact64 steps. The older 32-step undertraining diagnosis is partly addressed but not closed.

### Authentic source historical H16/H128/H512 generalization (descriptive only)

| Real delayed historical PM2.5 log1p MAE | Assoc18018 | Fastslow18018 | Assoc18019 | Fastslow18019 |
|---|---:|---:|---:|---:|
| H16 | 0.37873 | 0.43597 | **0.28219** | 0.71399 |
| H64 train lag | 0.42555 | 0.44569 | **0.35959** | 0.79705 |
| H128 | 0.55312 | 0.59967 | **0.46078** | 0.94394 |
| H512 | 0.52405 | **0.32430** | **0.42939** | 1.13124 |

Original source historical content supervision only at H64, all H16/H128/H512 shown WITHOUT source additional fitting. Strong seed/lag instability persists. At H512 fastslow seed18018 beats assoc, while fastslow seed18019 deteriorates severely. No source long-range retention superiority gate met.

### True physical future+6 outcome forecasting and interval calibration

| H64 physical all-channel source normalized score (lower better) | Assoc18018 | Fastslow18018 | Assoc18019 | Fastslow18019 | Original source last observation |
|---|---:|---:|---:|---:|---:|
| All-channel MAE | .65829 | **.63430** | .69507 | .67423 | .66344 |
| Gaussian NLL | 21.07474 | 48.35831 | 36.08053 | 6.16530 | — |
| Nominal 90% interval coverage | 78.38% | 81.53% | 73.49% | 79.92% | — |

This is a serious failure of distributional reliability. The very high H64 Gaussian NLL suggests the original variance head can heavily penalize rare source channels/residuals; further source/channel-level outlier/decalibration diagnostics are warranted **without adjusting or fitting to this exposed evaluation cohort**. H64 associative future point MAE is only better than persistence in seed18018 (.658 versus .663, slight gain), worse in seed18019 (.695 versus .663). Hence source future forecasting quality is NOT improved consistently by original B0 history supervision and conditional uncertainty is NOT qualified.

All original H16/H128/H512 physical source all-channel futures, every checkpoint, full user source provenance and exact train losses stored in \`research/replay-bundles/LATFUSE018/RESULTS.json\`.

## Qualification and next scientifically valid steps

**Engineering gates passed**: authentic real data, bounded source owner/causality, matched original active params and state, very tight original mathematical training-operator compute parity at registered full graph, exactly same source IDs/train data/order/optimizer steps and 2 seeds, full Git checkpoints+optimizer+RNG and independent clean checkout exact replay. No synthetic ability promoted.

**Scientific gates NOT passed**: robust incremental event-specific source content from matrix over zero-read/prior at every seed; real long-delay content; +6 multichannel future forecast superiority/persistence; nominal90% calibrated source intervals; independent original source holdout. Two seeds and repeat-exposed physical data are insufficient for production deployment.

Next: source-real FROZEN checkpoint-only (i) H64 output calibration tail NLL by individual actual channel/station, scale/censoring, whether source masks and preprocessing cause extreme standardized outcomes; (ii) controlled matrix-only ablation by source station/time and bootstrap episode CI, no weights fit; (iii) verify native legal query's specific source-write residual and time aliasing after training, versus source currently observed prior; (iv) propose separately preregistered B0 data/label/forecast balancing or writer objective adjustments only on NEW independent future source train/dev cases (do not tune on already exposed May–June); (v) separate true long-history semantic top1 among same-source genuine timestamps with causal query and no key oracle. Any new content-training improvements must preserve matched computational graph/resource rule and complete replay.

**Status**: 4 authentic original B0 content models trained, preserved and replayed. Core superiority, factual historical memory and predictive uncertainty/calibration qualification **FAILED**, PR DRAFT/UNMERGED. All original historical checkpoint versions and protected numerical source intact.
