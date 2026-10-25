# EDGEACTION001 — real logged joint-setpoint next-state pilot (2026-10-25)

**Scientific gate: FULL FAIL 0/5 original new seeds.** Genuine source-trained 14D observational target-conditioned forecasting, but the learned candidate loses badly to an extremely inexpensive **constant-velocity** reference. **FIT-only internal development, not independent TEST, causal motor effect, virtual/real robot controller, multi-domain capability or matched learned architecture superiority.** No additional fit is authorized under this registered run.

## Original, frozen real-data experiment

- Source: `lerobot/aloha_mobile_cabinet` revision `7a752b39f7e69de7e38aee485a6bab07528a061a`, source SHA256 `a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a`, 14 reported robotic joint positions and 14 same-frame logged joint-position targets, 50Hz nominal recordings. Actual motor-target execution delay and causal effects were **not** verified; this is a logged conditional observation study. This is not the synthetic SIM001 physics evaluation.
- Five **original 99,760-parameter** active/deployed `ActionQueryEdgeWorld` models, seeds `91001..91005`, each **256 optimizer steps**, batch 8, source-observed history 32 frames, 2,048 sampled genuine FIT origin presentations per model and complete source-gradients checked by CI. Same intrinsic 88D/8×8 affine evidence graph; command probe read-only.
- All processing statistics and servo prior fitted ONLY to source FIT episodes **0..48** (49 real recordings); original new model checkpoints/optimizers/RNG/sample schedules and SHA records frozen in Git **before any score**. Ten source episodes **49..58** were reserved as the one internal FIT subholdout, with **1,840 measured future transitions**, sampled every eighth eligible frame. Ten episodes, **not 1,840 independent natural sites**.
- Historical 13 DEV episodes 59..71 and old last15% TEST >=72 were not scored. All FIT source observations have been accessible in the historical research program; this internal holdout cannot establish untouched independent physical generalization.
- Original failed first attempt [run40188131091](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40188131091) stopped before any fit due to NumPy API misuse; correction did **not** change trial/model/gate. Registered five complete fits [run40188210109](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40188210109) performed 5x256 updates and saved original immutable checkpoint SHA; branch push failed solely because HEAD had advanced. Verified **no retraining** artifact recovery to Git [run40188397507](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40188397507) succeeded, exact five SHA matched. Only afterward, original [scoring run40188456161](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40188456161) **PASS** including original score clean replay. Checkpoints and `FORECASTS.npz` / `RESULTS.json` committed.

## Actual original internal score (raw 14D joint coordinates, 1 recorded-frame ahead)

| Single-step method | MSE lower better | MAE lower better | Movement-rich MSE lower better |
| --- | ---: | ---: | ---: |
| FIT-only velocity extrapolation | **0.000004872** | **0.001211** | **0.000016626** |
| FIT-only per-channel calibrated servo interpolation | 0.000021209 | 0.002376 | 0.000092882 |
| **TAC five-model mean** | **0.000048495** | **~0.003951** | **0.000225685** |
| Last measured state (persistence) | 0.000087896 | 0.004333 | 0.000440614 |
| Published next-position target as forecast | 0.003911929 | 0.027960 | 0.016684 |

Relative to persistence, learned model MSE reduces **44.83%** descriptively, but its MSE is **2.29× FIT-only servo** and **9.95× the strongest velocity** predictor. Moving state MSE is **13.57× velocity**. Velocity is the preregistered strongest fixed comparator for this cohort. A trivial kinematic formula is not an architecture-trained neural competitor; its strong result is the principal scientific challenge. No learned TAC vs GRU/SSM/Transformer parity study occurred, and thus no model-architecture superiority claim.

| Original seed | TAC original MSE | Capability gate |
| --- | ---: | --- |
| 91001 | 0.000047903 | FAIL |
| 91002 | 0.000047841 | FAIL |
| 91003 | 0.000049632 | FAIL |
| 91004 | 0.000048933 | FAIL |
| 91005 | 0.000048167 | FAIL |

**Registered gate:** each seed must have MSE at least 5% below the best FIT-only practical reference and movement-rich MSE no worse. **Every seed FAILS.** The run establishes neither a useful learned command response nor validated action-dependent uncertainty. Its single-step output is a point prediction; no 90% interval coverage is claimed. No selection of the nominally best seed, altered thresholds or further fitting against these scores is allowed.

## Interpretation and next capability target

The model preserves a legal affine parallel scan and read-only alternate-command queries, but that algebra does not imply the model encodes useful instantaneous dynamics or has learned correct logged-setpoint influence. This result is consistent with earlier native-physical failures against strong physically structured baselines. The next hypothesis should *explicitly preserve an independently measured physical reference such as causal velocity or FIT-only servo and train only the corrective residual* **within a newly registered complete model and fair-resource framework**. Real action timing/source may support observational target-conditioned state prediction; it still does NOT support claims about `do(action)`, safety, novel car physics or robot control.

First investigate whether the model's point forecast closely follows the available current-state anchor while ignoring short-term velocity; use descriptive original-case residuals/lesions only if they do not trigger new training against the already exposed holdout. Further actual training requires a different preregistration and must preserve original real source evidence, 95k–105k whole active/deployed parameters, matched original data/compute for neural head-to-head comparisons, strongest physical baselines, untouched external cohort, complete numerical math and original checkpoint replay. CPU wall time never needs to match.

## Immutable code/artifact references

- Registered study: `research/mathematical-engineering/edgeaction001/FAIR_COMPARISON.json`.
- Five original models/optimizer/RNG/seed schedules and live receipts: `research/replay-bundles/EDGEACTION001/seed-*/final.pt` and `live-budget.json`.
- Original model checkpoint index: `research/replay-bundles/EDGEACTION001/FIT_INDEX.json`.
- Original per-sample predictions + score/labels: `research/replay-bundles/EDGEACTION001/FORECASTS.npz` and `RESULTS.json`.
- Actual PyTorch implementation, source audit and frozen operator semantics: `src/tac_osm/variants/action_query_edge_world.py`, `scripts/edge_action_logged_fit.py`, `scripts/edge_action_fit_source_audit.py`.
- Companion mathematical proof boundary: [PNDS draft PR15](https://github.com/FPC-effortless/pnds-lean/pull/15), verified interface-purity and source-clock propositions, **not** whole learned source/IEEE proof.
- Original SIM001 simulation-only negative result: [draft PR399](https://github.com/FPC-effortless/TAC-OSM/pull/399). Its invalid `valid=False` zero-width result remains explicitly withdrawn; corrected 50-frame arm/vehicle-proxy gap coverage 1%/11.4% has **no real-world capability authority**.
