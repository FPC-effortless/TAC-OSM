# DEV-027 audited initialization-by-training-stream factorial

**Adaptive development, fixed eight entities. Draft and unmerged.** All20 freshly trained models and40 evaluation cells completed training, exact checkpoint replay and independent evidence audit.

## Finding

The registered whole-model initialization effect passes: switching A to B reduces pooled primary accuracy by14.90pp, with exploratory interval[-28.44,-1.35]. The training-stream effect is-4.06pp, interval[-13.09,4.97], and does not pass. Initialization is a supported contributor under this long-training recipe; stream contribution remains inconclusive, not absent. Neither result isolates the writer from the encoder/reader or licenses overall TAC success.

## Registered main effects

Whole-model initialization changes encoder, writer and reader together; stream namespace changes public histories and inspection choices together. Primary accuracy averages the two fixed H512 grids before paired-seed factorial analysis. Magnitude>=5pp and exploratory95% t(df4) excluding zero are required. Conditional effects/interaction are secondary and cannot rescue a failed primary.

| Effect | Mean change | Exploratory95% interval | Registered effect gate |
|---|---:|---|---|
| initialization_B_minus_A | -14.90pp | [-28.44, -1.35] | True |
| training_stream_B_minus_A | -4.06pp | [-13.09, 4.97] | False |

Registered main effects passing this development rule: initialization_B_minus_A. This is conditional evidence under the tested recipe, not proof of a universal or module-specific cause. Failed contrasts do not establish equivalence.

## All corners and per-seed reliability

A_A/A_B/B_A/B_B mean initialization A/B (offset23000/24000) and stream A/B (offset5000/6000). Each corner was retrained. Seed order43/59/71/89/107.

| Arm | Pooled H512 accuracy | Pooled per-seed accuracy |
|---|---:|---|
| A_A | 93.02% | 96.88%, 95.31%, 89.06%, 92.19%, 91.67% |
| A_B | 91.04% | 91.15%, 96.35%, 82.29%, 97.40%, 88.02% |
| B_A | 80.21% | 88.02%, 75.00%, 71.35%, 86.46%, 80.21% |
| B_B | 74.06% | 95.31%, 63.02%, 51.04%, 91.67%, 69.27% |

Pooling serves factorial analysis only. The unchanged reliability and economic gates apply on each grid independently.

| Arm | Grid | Accuracy | Prediction/swap gate | Selected physical gate | Selected cost/entity | Model-only cost/entity | Failure rate | Unnecessary service |
|---|---|---:|---|---|---:|---:|---:|---:|
| A_A | DEV023 | 93.75% | True | True | 0.535434 | 0.535937 | 0.001302 | 0.017014 |
| A_A | DEV024 | 92.29% | True | True | 0.526198 | 0.526997 | 0.001563 | 0.015799 |
| A_B | DEV023 | 90.83% | True | True | 0.537491 | 0.538281 | 0.000347 | 0.020486 |
| A_B | DEV024 | 91.25% | True | False | 0.528073 | 0.527344 | 0.000174 | 0.020399 |
| B_A | DEV023 | 79.79% | False | False | 0.639540 | 0.657031 | 0.015365 | 0.073958 |
| B_A | DEV024 | 80.62% | False | False | 0.644453 | 0.665538 | 0.020226 | 0.072656 |
| B_B | DEV023 | 73.54% | False | False | 0.611858 | 0.618750 | 0.000260 | 0.094618 |
| B_B | DEV024 | 74.58% | False | False | 0.603490 | 0.613194 | 0.000087 | 0.095226 |

All gates require every seed accuracy>=.8 and swap drop>=.1; selected failures<=.05, unnecessary<=.10, cost<=.80; mean selected<=mean model-only, without adjusted tolerance. Full per-grid per-seed metrics are in the JSON.

## Secondary conditional effects and interaction

| Contrast | Mean | Exploratory95% interval | Effect gate |
|---|---:|---|---|
| initialization_by_stream_interaction | -4.17pp | [-17.89, 9.56] | False |
| B_A_minus_A_A | -12.81pp | [-20.36, -5.26] | True |
| B_B_minus_A_B | -16.98pp | [-37.08, 3.12] | False |
| A_B_minus_A_A | -1.98pp | [-8.21, 4.25] | False |
| B_B_minus_B_A | -6.15pp | [-20.93, 8.64] | False |

## Runtime controls, evidence and limits

All four corners within a seed shared one process, CPU/runtime block and prospectively seeded order. Initial tensors matched the ten bound historical pre-training references. Within a stream pair, training histories and deterministic receipt rules matched; all components were active and changed. All models used768 memory scalars,8,674 trainable parameters,216 updates and221,184 public events/432 receipts. No old trained checkpoint supplied a corner. Both historical diagonals were retrained along with the crossed corners.

Retraining does not require final weights to reproduce historical checkpoints: numerical runtime may affect trajectories. The historical cohorts remain separate evidence. This experiment controls runtime within each seed; runtime differences across seeds and only five paired blocks limit precision. It does not isolate memory initialization from the encoder/reader, histories from receipt choices, or long-recipe exposure from optimizer behavior. No multiplicity adjustment; all evidence is adaptive/exploratory.

- Scientific source `80c00d0c74c0289283ed16477ba619d89553e391`.
- [Successful run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38928084677). All1,121 regression tests and initial tensor/training smoke checks passed.
- Complete artifact12057464979,11585343bytes; ZIP SHA256 `575455aa70cc5de08a478691445bb0197ea073f9f1b3d99bab1dd9171611eb17`.
- Local source/checkpoint/replay/initialization bindings, block matching and raw metrics verified; aggregate summary reproduced byte-for-byte.
- [Audited JSON](DEV-027-AUDITED-SUMMARY.json); [frozen protocol](DEV-027-INIT-STREAM-PROTOCOL.md).

Only the long recipe and two already-used H512 grids are tested. No H1024 or entity-count transfer claim. Native TAC overall and prior native attribution gates remain failed, regardless of any structured corner passing a development gate. All negative corners and per-seed failures remain in evidence. No further matrix follows this batch.

Both freshly retrained historical diagonal accuracies reproduce the corresponding frozen DEV-026 predictions at the aggregate/per-seed accuracy level. A_A passes prediction and selected-cost gates on both grids. A_B passes prediction on both, but on DEV024 its mean selected cost0.528072916667 exceeds model-only0.527343750000 by0.000729166667; the economic gate remains failed. Both B-initialized corners fail reliability on both grids.

Next proposal: [DEV-028 module-initialization decomposition](DEV-028-MODULE-INITIALIZATION-PROPOSAL.md). No additional matrix was started.
