# ME-010F — Frozen first real event-conditioned forecast and calibration results

**Scientific decision:** ME010 event-conditioned mixture **FAILS** the preregistered forecast-improvement gate. Original native TAC1000 remains the strongest overall forecaster on this real site/week; the current event head has only a slight Brier advantage over a historical constant event probability, and almost no decision-threshold event recall. No trained checkpoint promoted; no PR merge.

## Evidence provenance and workflow issue
- **First exposure:** GitHub Actions run [39386665512](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39386665512) ran the frozen April6–13 evaluation to the results-artifact stage. A separate push to prevent an unintended rebroadcast of the original training workflow cancelled the first run *after its complete results archive* was preserved. The original artifact is `12247368398`.
- **Duplicate replay:** GitHub Actions [run 39386728877](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39386728877) completed successfully with the frozen checkpoints, artifact `12247189805`. The original complete results JSON SHA-256 is **`03aff3ef9b429ee9f50f98ebb760a9aefcf6868529ca6c3dc82b407307b50eb7`** in BOTH artifacts, verified byte-for-byte. The replay is **not** a second untouched evaluation or a new model selection.
- Prior original training: [run 39386385479](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39386385479), artifact `12247552687`; six original 291-parameter readouts. Old native/GRU/ME009 baselines from original archived runs 39382850579, 39382074165, 39384177806. SHA-256 checks of **every original checkpoint before source opening** enforced by the evaluator.
- Authentic real UCI MetroPT-3 data 2020-04-06 inclusive through 2020-04-13 exclusive, **52,221 real sensor rows**, **4,216 valid forecast windows**, 12 authentic compressor channels, 32 causal historical readouts, 12 and 60 observation-step future horizons, exact original TRAIN-only feature normalization and event thresholds. Test-period optimizer updates **zero** and no source observations at/after April13 read by the evaluator. Official publisher ZIP SHA256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`.

## Original frozen real future score table (MAE lower is better)
| Model | Three-seed mean MAE | Three-seed mean MAE conditional on significant true changes | Quiet false-movement rate |
|---|---:|---:|---:|
| Persistence (untrained) | 0.667162401 | 3.059137777 | 0% |
| GRU41-1000, 3 seeds | 0.556584527 | 1.796205616 | 7.183% |
| **Unchanged native TAC1000, 3 seeds** | **0.521742632** | 1.670526454 | 6.374% |
| Previously rejected ME009 reliability gate, 3 seeds | 0.523263951 | 1.693073409 | 5.808% |
| ME010 direct 291-parameter residual correction, 3 seeds | 0.527470726 | **1.636692680** | 7.940% |
| ME010 event-conditioned two-regime dynamics, 3 seeds | 0.529149459 | 1.665267873 | 7.693% |

All model readout comparisons use matching original native seeds. Mixture fails primary MAE in every seed against unchanged native TAC (native 0.529385/0.529008/0.506835 vs mixture 0.538740/0.534258/0.514451). Event mixture loses to same-parameter direct head by approximately **0.318% relative MAE** on mean. Direct head improves true-change-only magnitude but worsens overall quiet-state forecasting and therefore also fails the primary gate. The previous ME009 reliability gate marginally reduces false predicted movement but has worse overall MAE than unchanged native.

## Proper event probabilities — actual future labels
Observed significant real event prevalence on future period **0.095706828**, versus historical (days20–25) event prevalence ~0.04972952 (a nearly doubled aggregate event rate, but this does **not** by itself prove a causal physical regime switch).

- ME010 mixture event Brier: **0.085965231 / 0.088093147 / 0.086538702**; mean **0.086865693**.
- Constant historical event-rate-only predictor Brier: **0.087791108** on all seeds.
- Mean event probability: **0.125921 / 0.139679 / 0.108948**; three-seed mean **0.124849** (not equal to actual prevalence ~0.0957).
- Fixed 0.5 event probability threshold recall **1.188% / 1.404% / 0%**, mean **0.864%**. Precision **40.07% / 31.70% / 0%** (mean 23.92%). Low recall and overprediction in aggregate reflect poor utility as deployable meaningful-change alert at this preregistered threshold.
- Mean ten-bin ECE **0.035699**, event log-loss **0.317061**. Event scores are observational next-readout event predictions, not independently verified failures/repairs.

## Decision and next mathematical diagnosis
**FAIL** event mixture over both direct equal-capacity correction and unchanged native TAC in MAE. Small Brier gain alone is insufficient; event recall and change/quiet tradeoff remain poor. **Do not tune** thresholds, class weighting or readout losses on this opened April6–13 data.

ME010's decomposition is still mathematically justified for squared-error conditional expectations, but its learned components did not establish better performance. Consider a distinct preregistered experiment on authentic multi-site time series with prior-event rate adapting only to *already observed delayed outcomes*, proper time-based training/calibration/test, calibrated probability scoring, and a strong GRU/persistence reference. Preserve TAC/GRU crossing and sudden persistence wins across prior weeks; do not claim state-of-the-art, deployed safety, world-state reasoning, counterfactual repair or system-two emergence. The PNDS Lean proof remains scoped to admission/convexity, NOT predictive probability calibration.

All previous unified TAC-OSM research ideas and negative evidence retained. TAC-OSM is **not an SSM**; no automatic merge.
