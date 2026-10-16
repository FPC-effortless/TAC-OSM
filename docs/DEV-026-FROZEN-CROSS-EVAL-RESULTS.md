# DEV-026 audited frozen cross-evaluation

**Adaptive diagnostic; no native TAC recovery or confirmatory attribution.** All20 archived diagonals and all40 crossed cells passed their registered numerical, exact new-replay and independent world/cost audits.

## Finding

The observed long-regime failures follow the trained checkpoints across both evaluation grids. DEV-023 models retain >=80% accuracy in every seed on both grids; the same three DEV-024 seeds remain below80% on both grids. Mean grid-switch effects are small compared with observed cohort gaps. This motivates investigating trained-model variation. It does not isolate initialization from training histories or optimization, and none of the registered five-seed contrasts meets the meaningful-separation rule: all exploratory95% intervals include zero. Failure to detect a grid effect is not proof of equivalence.

| Training regime | Frozen cohort | DEV023 grid accuracy | DEV024 grid accuracy | Prediction/swap gates on old / new grid | Selected physical gates on old / new grid |
|---|---|---:|---:|---|---|
| short | DEV023 | 98.54% | 99.38% | True / True | True / True |
| short | DEV024 | 99.17% | 99.17% | True / True | False / True |
| long | DEV023 | 93.75% | 92.29% | True / True | True / True |
| long | DEV024 | 73.54% | 74.58% | False / False | False / False |

## Registered long-regime contrasts

Gains/changes and exploratory paired five-seed t(df4) intervals, in percentage points. Meaningful separation needs magnitude>=5pp and interval excluding zero. No contrast passes.

| Contrast | Mean | Interval |
|---|---:|---|
| DEV024_minus_DEV023_on_DEV023 | -20.21pp | [-42.17, 1.75] |
| DEV024_minus_DEV023_on_DEV024 | -17.71pp | [-39.04, 3.62] |
| DEV024_grid_minus_DEV023_grid_for_DEV023 | -1.46pp | [-7.43, 4.51] |
| DEV024_grid_minus_DEV023_grid_for_DEV024 | 1.04pp | [-2.73, 4.81] |

## Per-seed primary accuracies

Order43/59/71/89/107. All raw predictions, reset/swap behavior and physical controllers are preserved.

| Regime | Cohort on grid | Accuracies |
|---|---|---|
| short | DEV023_on_DEV023 | 100.00%, 100.00%, 100.00%, 93.75%, 98.96% |
| short | DEV023_on_DEV024 | 100.00%, 100.00%, 100.00%, 97.92%, 98.96% |
| short | DEV024_on_DEV023 | 100.00%, 98.96%, 100.00%, 98.96%, 97.92% |
| short | DEV024_on_DEV024 | 100.00%, 98.96%, 100.00%, 98.96%, 97.92% |
| long | DEV023_on_DEV023 | 97.92%, 94.79%, 93.75%, 90.62%, 91.67% |
| long | DEV023_on_DEV024 | 95.83%, 95.83%, 84.38%, 93.75%, 91.67% |
| long | DEV024_on_DEV023 | 94.79%, 63.54%, 51.04%, 88.54%, 69.79% |
| long | DEV024_on_DEV024 | 95.83%, 62.50%, 51.04%, 94.79%, 68.75% |

## Cost and numerical boundaries

DEV024 short on the DEV023 grid predicts99.17% correctly but fails the selected-cost comparison: mean selected cost/entity0.539383680556 exceeds model-only0.539236111111 by0.000147569444. The unchanged strict economic inequality remains failed. Prediction accuracy is not sufficient for combined utility. DEV024 long fails prediction and physical gates on both grids.

All20 diagonals matched identities, predictions, reset/swap predictions, actions, services and realized costs exactly. The largest permitted inference difference observed was2.384185791015625e-6. Brier changes are recorded for every diagonal; these are not accuracy or economic tolerances. All40 new cells reproduced exactly in separate processes. The independent diagonal certificate gated cross-evaluation and bound the final audit to the same20 reports.

## Evidence and interpretation limits

- Scientific source `f0b09f6360457604f3a48f97ebe6bb62b575f702`.
- [Successful11-job run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38926260602), 2026-10-16 15:43:35–15:51:24UTC. All1,117 regression tests passed.
- Complete artifact12056767880,36,776,920bytes. ZIP SHA256 `74c8a34c67148b4c0d53bfa7a648da29fc4dcc0c22ba779cc17fdd530d00f25b`.
- Local archive/source/input/checkpoint/replay bindings, all diagonal comparisons and raw metrics verified. Aggregate summary reproduced byte-for-byte.
- [Audited JSON](DEV-026-AUDITED-SUMMARY.json); [frozen protocol](DEV-026-FROZEN-CROSS-EVAL-PROTOCOL.md).

Known diagonals and some partial DEV-025 raw outputs already existed. This is adaptive diagnostic reuse over two sampled grids of the same fixed-eight-entity generator, not fresh confirmatory or entity-count transfer evidence. Checkpoints were frozen and training steps were zero. Initialization and training-stream namespaces remain bundled in cohort comparisons; numerical validation does not explain the learning discrepancy. DEV-025 remains blocked under its original protocol. DEV-024 long-regime attribution and native TAC overall gates remain failed.

Next justified experiment: retrain a matched initialization-by-training-stream factorial, with all four corners in the same runtime block for each seed. Retrain both historical diagonals as well as the crossed combinations to avoid confounding old checkpoint infrastructure with the new combinations. Keep the long recipe and both evaluation grids fixed. No architecture redesign is justified solely by this cross-evaluation.
