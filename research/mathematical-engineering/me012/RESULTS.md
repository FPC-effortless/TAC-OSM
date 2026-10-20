# ME-012 — Original real ETT split-conformal mathematical calibration results

**Scientific decision: FAIL.** Frozen post-hoc split-conformal correction increases measured interval coverage, but calibrated native TAC and GRU still lose to a conformally calibrated persistence baseline on the original physical transformer and fail to transfer their 80%-nominal intervals to the second transformer.

## Original source and independent execution

- One-time successful first-exposure [GitHub Actions run 39389147643](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39389147643), source SHA `990e140b00eb06bf739402f5ecbcf08fccf82e7b`, original results artifact `12249041358` (`me012-original-ett-conformal-real`).
- Original result `first-frozen-conformal-results.json` SHA256 `ac6c12249ae159fdde754619aa17d0174645bc69afedb6d90737815fd82736d2`.
- Original six learned forecast weights from [run 39388044790](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39388044790), archive `12248760583`, original training commit `957d45e939e66d8f0bbce8d1f7537bed6e2f6d69`. All six exact original checkpoints verified against their SHA256 manifests **before** data exposure.
- Genuine ETTh1 real electricity transformer: source SHA256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`, MD5 `8381763947c85f4be6ac456c508460d6`.
- Genuine second physical transformer ETTh2: source SHA256 `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b`, MD5 `51a229a3fc13579dd939364fefe9c7ab`.
- Frozen original native TAC two-timescale (3 seeds, 7558 params) and GRU (3 seeds, 6966 params) models learned from the ETTh1 TRAIN [0,8760). Calibrate only against previously inspected ETTh1 historical [8760,11640), for `n=452` overlapping real forecast windows. New final real TEST period [14520,17420), **456** windows per site; source observations >=17420 not parsed by evaluator. Training, tuning and model updates in test **zero**.

## Formal algorithm implemented
For each model/horizon, `s_i=max(q_0.1(x_i)-y_i, y_i-q_0.9(x_i))` on genuine earlier ETTh1 calibration data. Compute `c_h` as the `ceil((n+1)*0.80)`th increasing calibration score. Return interval `[q_0.1-c_h, q_0.9+c_h]`, median unchanged. Compare a persistence model centered at the most recent observed oil temperature and calibrated with exactly the same order statistic applied to historical `|true signed delta|`.

**Formal qualification:** The classical marginal-coverage claim assumes exchangeable calibration/test conformity scores. The overlapping time windows, temporal drift and separate-site transfer here do **not** establish exchangeability. Therefore **no mathematical coverage guarantee is claimed**. The implementation verifies the order statistic and that no median is altered, not that the true world follows model assumptions.

## Original authentic final time-test results — means over three saved seeds, nominal target 80%

| Site and method | Raw 80% coverage | After calibrated coverage | After interval width °C | Frozen point MAE °C |
|---|---:|---:|---:|---:|
| ETTh1 final persistence | n/a | **82.895%** | **8.2310** | **2.4068** |
| ETTh1 final native TAC | **52.961%** | 70.175% | 8.8416 | 3.2313 |
| ETTh1 final GRU | 59.613% | 70.651% | 9.3917 | 3.3878 |
| ETTh2 final persistence | n/a | **51.535%** | **8.2310** | **5.2116** |
| ETTh2 final native TAC | 34.101% | 44.079% | 9.3354 | 6.0911 |
| ETTh2 final GRU | 23.830% | 29.423% | 7.1909 | 6.7564 |

- Absolute oil-temperature errors are in degrees Celsius. No recalibration improved the frozen point forecast by construction. TAC wins against GRU in final time-period point error, but **both learned models lose to persistence on both sites**.
- Calibrated ETTh1 TAC still undercovers by ~9.8 percentage points, while calibrated persistence covers nominal target and is narrower. ETTh2 learned coverage is worse; persistence also undercovers because ETTh1 calibration cannot guarantee cross-site exchangeability.
- ETTh1 TAC per-seed adjusted coverage 70.0% / 73.0% / 67.5%; GRU 71.8% / 76.0% / 64.1%.
- ETTh2 TAC per-seed adjusted coverage 37.9% / 45.2% / 49.1%; GRU 30.4% / 28.5% / 29.4%.
- No model had interval/median-order violations after adjustment. Posthoc calibration offsets normalized by ETTh1 TRAIN OT sigma: persistence [0.3624045, 0.5398796] for horizons 24/168; model offsets are recorded per seed in immutable results JSON.
- This study exhausts the originally registered final ETTh1/ETTh2 period through row17420. Do **not** reuse those now-opened data as an untouched test for further learned calibration or architectural search.

## Decision under the unified master research plan
**REJECT** fixed-source CQR as a sufficient reliability solution; retain code/evidence as a diagnostic and mathematically disciplined baseline. The original native TAC recurrence is still an active research implementation, but no ETT checkpoint is authorized for physical intervention or production deployment.

**NEXT:** Stop repeatedly using the same heldout period. Shift to **genuine independent multi-system histories** with predictable environment/context covariates and logged actions/outcomes where available; preregister chronological source splits and strong seasonal/probabilistic baselines. Train predictive distributions with correct conditional losses, explicit time and regime context, and evaluate online delayed-feedback calibration with strictly observable histories. Preserve whole unified plan (structured/learned entity memories, neural CASM, executable VSA operators, CDL-PHR, BDH local plasticity, SSM-grade mathematical constraints, PNDS Lean/Veritas) as separate falsifiable mechanisms, not an all-at-once complexity claim.

TAC-OSM is not converted into an SSM. No state-of-the-art, System-2, safe-control, causal repair, or commercial claims. Draft research branch, **unmerged**.
