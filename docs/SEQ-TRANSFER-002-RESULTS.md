# SEQ-TRANSFER-002: frozen DEV-058 checkpoints on supervised next-bit prediction

**Date:** 2026-10-18. **Status:** Local experiment completed; task-specific linear heads trained, pretrained weights unchanged. **Base checkpoint:** DEV-058 draft PR #217, original portable exports for seeds 7581 and 7582. This report records an independent out-of-task transfer study and does not modify prior experiments.

## Hypothesis and test
Does the previously trained TAC-OSM memory provide useful sequence-prediction representations, and do trained features beat untrained representations when evaluated against simple sequence baselines? Binary symbols were mapped to the original public 22-feature event bit; the 32 × 48 memory state and occupancy (1568 values total) were frozen and given to a newly trained, independent L2-logistic next-bit head (C=0.1). Each task and length used 600 training examples and 400 independently generated held-out examples, exactly balanced with complement pairs. Original checkpoint replay first passed 33/33 recorded action choices per seed. No pretrained tensors were updated.

Tasks: constant; alternating; periodic with lengths 3 and 5; third-most-recent bit copying; fifth-most-recent bit copying; truly unpredictable IID next bit. Lengths: 32 and 96 events.

## Held-out accuracy
Each column uses a separately trained, task-specific head on that length; both checkpoint seeds gave identical results.

| Task | 32 events | 96 events | Fit at 32, test at 96 (7581 only) |
|---|---:|---:|---:|
| Constant | 100% | 100% | 100% |
| Alternating | 100% | 100% | 100% |
| Period 3 | 100% | 100% | 46% |
| Period 5 | 100% | 100% | 52.5% |
| Copy lag 3 | 100% | 50.5% | 51.0% |
| Copy lag 5 | 100% | 46.0% | 50.5% |
| Independent next bit | 61.5%* | 48.5% | 51.25% |

*Post-hoc audit on six **new** balanced IID sets, 2,000 samples each, gave 2,920/6,000 (48.67%) at length 32 and 3,140/6,000 (52.33%) at length 96, collectively **6,060/12,000 = 50.5%**. The apparent original 61.5% small-sample anomaly is not evidence of predicting random bits.

## Controls and comparison
These values are **unweighted averages across 14 task–length cells**, not pooled accuracy or model superiority statistics.

| Arm | Average accuracy |
|---|---:|
| Frozen trained TAC seed 7581 + trained readout | 86.179% |
| Frozen trained TAC seed 7582 + trained readout | 86.179% |
| **Trained gate/address + randomly reinitialized encoder** + trained readout | **86.179%** (both seeds) |
| Fully randomized DEV-058 model with forced writes + trained readout, seed 7581 | 68.893% |
| Fully randomized DEV-058 model with forced writes + trained readout, seed 7582 | 65.464% |
| Independently trained 5-gram | **93.0%** |

A non-neural explicit buffer saving the first 31 bits and most recent bit, with the same linear head, also matches trained checkpoint patterns and delayed-copy accuracies essentially exactly. This mirrors the **observed trained writer behavior**: after 32 writes, slot 31 is continually overwritten. The frozen encoder itself therefore does not exhibit demonstrated transferable algorithmic predictive value. The performance advantage over a fully random address writer can be explained by learned *where-to-write* structure; the random-encoder matched-address control isolates this.

Train-at-32 and test-at-96 without retraining is near chance on periods 3/5 and delay-3/5 despite separate task-specific heads reaching 100% on same-length periodic problems. No demonstrated length generalization.

## Boundaries
The model is originally a **trained maintenance candidate-action scorer**, not an autoregressive next-token model. This study probes memory through a newly supervised head; it does not train a native sequence-prediction world model. Periodic synthetic tasks have few distinct templates and are not natural language. The scratch arm is **random frozen features + trained readout**, not a compute-matched fully trained-from-scratch TAC-OSM. An end-to-end scratch control, trainable retention/addressing, and a matched GRU/SSM/Transformer benchmark remain future work. The model should not be advertised as competitive with sequence-modeling architectures.

## Evidence and reproduction
Saved local evidence: unchanged 7581/7582 full portable exports, exact SHA256 provenance, primary frozen-readout script, readout prediction CSV, full arm-level JSON, explicit-buffer/IID robustness script/results, and cross-length script/results. A standalone reproducible archive was generated from this run as `TAC_OSM_Sequence_Transfer_V2.zip` and supplied in the ChatGPT conversation. This GitHub record contains the findings and summary data; it does **not** claim that a new CI workflow ran.

Checkpoint SHA-256:
- 7581: `d3edd648b21c1a8a54575ad3668ddd20a90d9cf75983e2ffc075b9918f88330d`
- 7582: `6000d007c8eeba09d3ce450ac6a23d242fe64d0d75fb79c836157ad406ae7149`

This work should remain **draft/unmerged** pending scrutiny and a fully-trained scratch-versus-pretrained sequence objective test.