# ME039 — native execution and repository-owned model delivery

The new ME038 native fast/slow/associative-memory architecture remains unchanged.
This iteration builds an explicit compiled-core execution path, measures actual
training/serving costs on authentic records, and puts original model bytes and
numerical evidence into the repository. No new accuracy or architectural-novelty
claim; ME038's passed power target and failed compute/legacy gates remain intact.

| Measured engineering outcome | Result |
|---|---:|
| Steady training speed, scripted core vs eager | 1.076x |
| Cold-inclusive 64-step speed, scripted core vs eager | 1.067x |
| Training targets (1.20x steady /1.10x total) | FAIL / FAIL |
| Serving, scripted core vs eager Python | 1.338x |
| Serving, scripted-core export vs original complete ME038 export | 0.963x |
| Original standardized/raw saved validation values exact | 4,224,000 |
| Second serving-freeze construction | REJECTED: exact forecast contract failed |

**Retain original ME038 complete serialized models for serving.** The 1.338x
comparison uses eager Python, which is not the best pre-existing deployment
baseline. The original serialized format is faster than the alternate export.
The training path gives a small measured gain, misses both targets and remains
experimental. No inference/training/default model is silently replaced.

## Complete real-data execution comparison

Eight preselected source/horizon/seed cases, three paired repeats, 64 updates/path,
3,072 optimizer steps total. Source ETTh1/Electricity TRAIN windows; all updates
include the complete forecast/AR/context/native model, original loss, AdamW,
gradient clipping and identical sample-stream prefixes. No selected candidate
capability fits, synthetic benchmark fitting, test rescore or closed-cohort tuning.

| Case | Steady training speed | Cold-inclusive training speed |
|---|---:|---:|
| exposure-electricity-h720-innovation_tac-s38001 | 0.984x | 0.988x |
| exposure-electricity-h720-innovation_tac-s38002 | 1.012x | 1.013x |
| exposure-electricity-h96-innovation_tac-s38001 | 0.995x | 0.997x |
| exposure-electricity-h96-innovation_tac-s38002 | 0.996x | 0.998x |
| exposure-ETTh1-h720-innovation_tac-s38001 | 1.148x | 1.129x |
| exposure-ETTh1-h720-innovation_tac-s38002 | 1.160x | 1.136x |
| exposure-ETTh1-h96-innovation_tac-s38001 | 1.162x | 1.147x |
| exposure-ETTh1-h96-innovation_tac-s38002 | 1.175x | 1.147x |

Compilation/load/optimizer setup and first 16 steps are included in cold costs;
steady costs use steps 17..64. All three repeats and per-step costs are retained.
Source/CPU/quota/version details are in training.json. No full FLOP/energy parity.
Maximum native-gradient relative L2 difference: 3.9839261e-07; acceptance
2.28883146e-05. Maximum final snapshot parameter relative L2
4.99851774e-08. Agreement PASS under the prospectively registered engineering
criterion. Scripted backward rounds differently: this is not bit-exact retraining
or a formal nonlinear optimizer error bound.

## Frozen models and rejected follow-up

All 60 original selected native models replay 2,112,000 standardized and 2,112,000 raw
physical-input validation values exactly; raw exports save/reload correctly.
Those are saved probes, not billions of final forecasts. Weights/state capacity
remain identical; full 96-step replay remains necessary. Original source/weights
are hash checked; no accuracy, lifetime-streaming, controller/VoI or commercial
claim is created by execution speed.

The serialized-baseline comparison is explicitly POST HOC, with its addendum
registered before that timing. It preserves all eight cases. Original all 60
serialized model files total 23,845,178 bytes; alternate
exports total 22,548,422 bytes. These file sizes are not learned
weight compression or total process-memory measurements.

A second construction froze the original inference graph with numerical
optimization disabled. Six models passed before the first failure at
`exposure-electricity-h96-innovation_tac-s38001`: 1,574values differed,
max absolute discrepancy 0.015625 in original
publisher units. The strict contract was not relaxed. It was rejected and the
remaining cases/timing gate were not completed. The failure source/log/array and
original input/output probes remain available. Graph constants still contain
learned weights; zero exposed Parameters never means zero model capacity.

## Repository and delivery

[Original trained native models and all new numerical records](../../model-artifacts/README.md)
are stored as checked repository archives, not just local paths or deterministic
seed claims. All 48 continuation final optimizer/RNG/model states and 1/16/64 gradient/
parameter snapshots are included, split by source into bounded Git files.
All 60 experimental export files/probes are separately labelled; use the original
ME038 serving files. New numerical-record archives contain 0 new capability models.
CI verifies original/archive payload SHA-256 and uploads repository-owned Actions
artifacts. Record actual run/artifact IDs and download verification separately.

The older full 1.05 GB ME038 evidence archive remains local: release metadata writes
worked, but GitHub's asset-upload host rejected configured credentials (HTTP401).
The empty draft was deleted; no manual credential retrieval/login occurred. Do
not claim that full archive is hosted. Current model bytes/new numerical records,
source, failed gates and receipts are all committed on the draft research branch.

Local suite: 1,024 tests +12 subtests passed (55 warnings, including recorded TorchScript
deprecation). Compiler support remains version-specific. This study adds an opt-in
training execution prototype; it does not establish new predictive capability.
