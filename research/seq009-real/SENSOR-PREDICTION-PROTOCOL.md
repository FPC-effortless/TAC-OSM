# SEQ009-S — Authentic SCANIA next-readout dynamics baseline

**Preregistered before this study's validation readout values are inspected.**
Date: 2026-10-19. Governing policy: REAL-WORLD-EVIDENCE-RULES
from PR #237; predecessor official-source audit PR #238 passed on
GitHub Actions 39314460663. The source audit verified 2,272
recorded SCANIA training repairs; this separate experiment DOES NOT
predict repairs or measure intervention utility.

## Falsifiable real-data question

When predicting the next *actual* operational readout on a new SCANIA
vehicle, does a learned sensor-wise affine AR(1) baseline outperform
the strong zero-parameter last-observation/persistence baseline?

This is a necessary conventional-model benchmark before evaluating
TAC recurrent world memory, GRU, state-space models or episodic executable
memory on the exact same authentic sequences.

**Dataset:** publisher version-3 SCANIA Component X, source dataset DOI
10.5878/bnh5-ka77, original archive
https://api.researchdata.se/dataset/2024-34/3/file/zip .
CC BY 4.0. Download ZIP with SHA256; extract and hash ONLY
train_operational_readouts.csv and
validation_operational_readouts.csv. Do NOT open publisher TEST file
bodies, TEST labels, future repair records or simulated outcomes.
Published split is approximately 23,550 train and 5,046
validation trucks; actual numbers must be reported from original bytes.

## Inputs and measurement

Training: **all strictly chronological adjacent observations within
each distinct original TRAIN vehicle**, no cross-truck pair. Predict
next real sensor vector after previous real sensor vector, with
time_step / vehicle_id excluded as model inputs. Numeric sensor
features are signed-log transformed identically for both baselines.
Missing previous/target values are skipped symmetrically for each
feature, never converted silently to 0. Refuse duplicate time steps
or decreasing within-vehicle time in the published stream; repair
only through a new registered code revision, preserving the failed
evidence.

Validation: **one last consecutive genuine readout pair per original
VALIDATION truck**, representing a real future observation. No
training on validation readouts. Every eligible truck contributes
one target pair; report eligibility and censoring of trucks with
fewer than two observed readouts. No model selection from the
validation metrics. Verify train/validation IDs are disjoint.

## Models and fixed optimization

PERSISTENCE predicts y=x (last observed values), no parameters.

AFFINE_AUTOREG trains one independent linear next-state function
per observed sensor dimension y_j ~= a_j*x_j + b_j using exact
TRAIN sufficient statistics. Fixed ridge penalty = 0.001*N_j,
with fitted a_j clipped to [-1,2]. Single deterministic fit,
no hyperparameter tuning and no input feature selection based on
validation labels. This baseline tests whether learned recurrence
should be compared against more than naive persistence.

## Metrics and claims

Report signed-log absolute MAE: micro across all valid pairs/features,
macro across sensor features, and macro MAE divided by TRAIN-only
mean absolute readout difference with minimum scale .05. Report
number of trained pairs, genuine validation trucks/readouts, sensor
coverage and feature count, original ZIP/file SHA256, extraction and
CPU usage where possible. Smaller MAE is better.

A successful trained affine baseline is empirical prediction on
real collected sensors, **NOT proof of causal or semantically
correct world representation**. Othello-GPT motivates later
latent-state probes and intervention controls, but those are
not present in a diagonal AR model. This benchmark does not
test unseen identity aliases, repair-risk outcome, stopping
policy, physical intervention economics or commercial readiness.

Even if one baseline beats the other, preserve the result and
promote NO TAC-OSM architecture. Follow-on controlled studies
must train GRU, TAC, selective SSM and BDH/plastic variants on
the same true TRAIN histories, preserve TEST and compare
cost, calibration and action-relevant risk with tuned
tabular/survival baselines, plus independent calendar-time
corroboration.

**Status at registration:** no SCANIA operational-data training
in this study yet. The original small-source audit is distinct.
