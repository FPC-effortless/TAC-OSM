# ME-010 — Original six real-data trained event/readout models

**Status:** original fixed training matrix PASSED; genuine future TEST results recorded separately in stacked ME010F. No model promotion.

- Original failed numerical preflight: GitHub Actions run **39386175014**, a parenthesis typo in the event ECE metric expression, caught before fitting or exposing new April data. Corrected that syntax only; no mathematical architecture/training protocol changes.
- Original successful genuine historical calibration: [run 39386385479](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39386385479), exact source code commit `27d2bb84bf0d3640b290238a908ce4a9cb7a008a`, evidence artifact `12247552687` (`me010-original-event-vs-direct`).
- Source: official UCI MetroPT-3 publisher TRAIN-era authentic sensor histories; publisher ZIP SHA256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`. Original native TAC1000 weights from ME008 frozen and hash-verified, three seeds. Zero native-backbone refits. New head data are historical **days 20–25 previously used as exploratory validation**, now **readout-training/calibration**. It is not held-out evidence.
- Three random seeds 101001/101002/101003 each trained a 291-parameter direct head and equal-sized 291-parameter two-regime event head, fixed 260 AdamW optimizer updates, batch 128. All six exact checkpoint SHA256 digests checked against original results manifest.

## Original historical data results (lower MAE better)

| Model | 101001 | 101002 | 101003 | Mean |
|---|---:|---:|---:|---:|
| Direct correction to frozen TAC | 0.254514 | 0.265096 | 0.248939 | 0.256183 |
| Two-regime event-conditioned dynamics | **0.252045** | **0.260437** | **0.245433** | **0.252638** |

Historical event prevalence is **0.04972952**. Original in-history event-mixture Brier scores 0.04644599/0.04898429/0.04855485; event-probability means 0.11933/0.13158/0.10848. At fixed event-declaration p>=0.5 the future-event recall was 2.31% / 2.82% / 0%. Probability calibration was already questionable despite relative accuracy improvement in these historically exposed records.

**Decision:** Completed executable real mathematical experiment, not a prediction-advantage qualification. This exact model was forwarded to single-exposure unchanged-weight April6–13 test with no after-the-fact model hyperparameter edits. Do not present these results alone as a genuine heldout success.

**Architecture math:** With `E=1{abs(Δ)>TRAIN threshold}`, unconditional expected signed delta is `p*E[Δ|E=1]+(1-p)*E[Δ|E=0]`; the conditional quiet expectation is NOT identically zero. The law of total conditional expectation is an exact population identity, but the resulting conditional mean is not the optimal MAE estimator. The binary event probability receives an authentic Bernoulli log-score during calibration; future targets are never model inference inputs.

All prior TAC/PNDS/DEV/SEQ evidence preserved. No SSM conversion or real action-effect claim.
