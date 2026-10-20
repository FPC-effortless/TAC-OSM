# ME-009F — Frozen real April reliability-gate evidence and rejection decision

**One-time genuine out-of-time test:** [GitHub Actions run 39384449469](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39384449469), original first-exposure SHA `71ffa66af9dbf42c4138513ffb85088b420f65a2`, original archived scores artifact `12246891774` (`me009f-original-reliability-future`).

**Source:** official UCI MetroPT-3 ZIP SHA256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`; new chronological week **2020-03-30 00:00 inclusive through 2020-04-06 00:00 exclusive**, 47,078 genuine observations; 3,872 valid multi-horizon forecast windows; exact original train-only scale and observation/horizon definitions, 12 physical sensors, missingness masks. No future records at/after April6 opened in this job. Original frozen three native TAC1000, three GRU41-1000, and three tiny gate checkpoint SHA256 digests verified *before* opening the future source. **Zero** test-period training, score calibration or hyperparameter selection.

## Original frozen real next-readout MAE, lower is better

| Model | Seed 240401/909001 | Seed 240402/909002 | Seed 240403/909003 | Mean |
|---|---:|---:|---:|---:|
| Persistence | — | — | — | **0.777239722** |
| GRU41, 1,000 steps | 0.432889889 | 0.433899004 | 0.426735611 | 0.431174835 |
| **Native TAC1000** | **0.414241760** | **0.417625282** | **0.436258023** | **0.422708355** |
| Learned 161-param reliability gate on native TAC | 0.419065504 | 0.424078202 | 0.442695822 | 0.428613176 |
| Fixed half-strength TAC residual | 0.585707414 | 0.586836492 | 0.596589171 | 0.589711026 |

Native TAC1000 wins 2/3 paired individual seeds over GRU41 and by ~1.96% mean MAE across three paired seeds; do not claim statistically robust generalization from three seeds on one installation. Learned gate loses to unchanged native TAC in **all three** frozen seeds by ~1.40% mean relative MAE, despite outperforming persistence and the GRU on mean. A fixed half-strength gate is much worse than all learned predictors.

## Actual quiet-state false-movement and changed-value tradeoff

- Quiet false-movement rate (predicted absolute residual above pre-frozen TRAIN-only change threshold when actual target is below threshold):
  - Original native TAC1000 **3.841%**
  - Learned reliability gate **3.282%**
  - Fixed half-strength **0.918%**
  - GRU41 **3.551%**
  - Persistence **0%**
- Conditional magnitude MAE on observed significant future changes: native TAC **1.72992**, learned gate **1.75175**, GRU41 **1.83956**, fixed half **2.38907**, persistence **3.06784**.
- Mean dynamic gate coefficients for three frozen seeds: **0.9327, 0.9031, 0.8993**. The learned gate generally trusts native forecasts; suppressing some useful changes offsets the gain in reduced quiet false movement.
- Direction accuracy on hindsight-identified real changes cannot be used as evidence of operationally available event detection. Positive/negative source counts remain inside the full archived per-model metrics.

## Qualification decision

**Reject ME-009 reliability gate as a default TAC-OSM forecast component.** It does not improve real MAE over original frozen TAC on the independent registered week; its lower false-movement rate is insufficient to compensate. Do NOT tune it further using the now-opened April first week.

**Retain original native TAC1000** as the current strongest of the measured trained recurrence designs on this week; **do not select architecture globally** because ME008F March23–30 all trained models lost to persistence. Next engineering target is predictive reliability / distribution shift detection with a rigorous independent train/calibration/test setup and corresponding physically meaningful feature decomposition. Needs independent compressor site and real decision-outcome evidence for world model/action utility qualification.

Mathematical containment of forecast interpolation, PNDS Lean verified memory admission, CDL-PHR negative relevance, self-organizing graph and executable operator hypotheses are all preserved. No SSM conversion. No merge/promotion.
