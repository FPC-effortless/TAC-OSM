# REAL-MAINT-003 — a working measured-sensor forecasting component

**All local numerical forecasting gates pass.** The validation-selected RevIN
MLP reduces mean per-room/horizon MAE by **32.1%** versus persistence on the held-out
2019 recording. Every one of its 20 fresh initializations has no point-error
regression on any of the ten rooms at either horizon. The simultaneous bounds
also pass, as does comparison with the validation-selected ridge reference.

**Unit erratum:** the frozen protocol incorrectly says Celsius. The published
measurements and every fitted/evaluated number are Fahrenheit. The corrected
portable API returns `temperature_F`, with explicit input units. Weights, scores,
gates and candidate selection are unchanged. Read [UNIT-ERRATUM.md](UNIT-ERRATUM.md);
the original protocol and superseded mislabelled export remain preserved.

| Horizon | Selected model MAE, °F | Persistence MAE, °F |
|---|---:|---:|
| 30 minutes |0.5137|0.7901|
| 60 minutes |0.8906|1.2753|

These rows average raw errors across rooms and training seeds. The headline is
the average of the twenty cell-relative errors, not an NMSE, failure rate or ROI.
For Celsius MAE multiply these errors by 5/9, without subtracting32.

## Controls and generalization

| Arm | Seeds without regression on every test cell | Mean TRAIN-scaled MASE |
|---|---:|---:|
| Linear, static |20/20|1.1280|
| Linear, RevIN |0/20|1.1945|
| MLP, static |20/20|1.0040|
| **MLP, RevIN; selected before fresh fitting** |**20/20**|**0.8619**|
| Persistence reference |deterministic|1.2690|
| Validation-selected multivariate ridge, λ=.1 |deterministic|1.1881|

All eight reference variants and every neural fit are retained. RevIN is not a
universal improvement: the linear RevIN arm regresses on some cells. This is a
forecasting-recipe comparison, not evidence for CDL, CASM, GRU/SSM superiority or
the entire TAC-OSM architecture.

Selected-model mean relative MAE ranges from .6677 to.6902 across individual seeds.
The largest simultaneous one-sided95% upper raw-MAE difference against persistence
is−.10877°F, still below zero. Mean TRAIN-scaled MAE difference versus the locked
ridge reference is−.32626, with ordinary one-sided95% upper bound−.31495.
Inference uses the registered2000 paired bootstrap draws over20 seeds and circular
two-day blocks. Eleven adjacent experiment days in one building remain a small
physical cohort; blocking does not create independent facilities.

## What was trained and tested

All80 fresh neural fits and 8 validation-only pilots completed 1024 updates, using
final EMA only. Fitting/preparation/reference/pilot CPU cost was133.243 seconds.
Learning rates and candidate were chosen from validation before the fresh matrix.
All20 candidate seeds passed validation before the2019 exposure marker was written.
No post-test checkpoint/arm substitution or fallback to persistence occurred.

There are37 training CSV days/3145 windows,15 validation days/1275 windows and 11
held-out days/935 windows. Only the ten room temperatures, past outside temperature,
past measured-derived RTU power and past logger time enter the model. Fault names,
filenames, future controls/weather, calendar years and simulator truth do not.
These are controlled physical fault experiments, not an independent factory trial.

This is an ordinary64 hidden-unit GELU MLP with a zero-initialized residual head
and reversible normalization from past readings. It learns directly from observed
future measurements, without a Bayesian posterior teacher or maintenance script.
It produces twenty point forecasts in one forward pass. It does not select repair
actions, discover symbolic programs or output calibrated health probabilities.

## Engineering evidence and release scope

All88 neural checkpoints and 8 references replay with zero observed drift across
3,996,436 compared values. Four corrected standalone exports reproduce8 sample
forecasts, retain identical checkpoint bytes and raw predictions to the original
exports, and declare Fahrenheit correctly. The separate semantic unit test catches
the error that numeric self-comparison missed. All1485 regression tests pass.

First predeclared seed 97301 measures0.0705ms median forward CPU time at batch1,
float64/thread1, after 5 warmups/100 repetitions. Parameter tensors occupy68256
bytes, buffers192 bytes, and one supplied context832 bytes. These are tensor
storage and warm forward timings, not total Torch process RAM, physical DRAM
traffic, energy, startup/JSON latency or a production/universal-host SLA.

Portable release: `learned-forecaster.tar.gz`, every arm at seed 97301, never a
best-test seed. Remote replay must pass on the final PR head separately. Source,
all raw/final checkpoints, references, windows, curves, failures and unit erratum
remain SHA256-indexed. The data are CC BY4.0; project metadata declares Apache-2.0.

This is a **raw-F room-forecasting component with a declared unit erratum**. It
does not repair REAL-MAINT-001/002, cover every analog sensor, establish transfer
to a new building, verify real repair success or demonstrate profitable maintenance.
Actual action/cost/independent-outcome and broader world-model gates remain open.
The overall goal stays ACTIVE; MetroPT July B remains unopened.
