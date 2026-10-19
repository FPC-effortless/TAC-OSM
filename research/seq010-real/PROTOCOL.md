# SEQ010 — real SCANIA recurrent memory, train-internal exploratory validation

**Registration:** 2026-10-19, before TRAIN-internal validation scores are observed.
**Status:** mechanism pilot, no native TAC attribution and no published
VALIDATION/TEST exposure. This research honors
`research/REAL-WORLD-EVIDENCE-RULES.md` (PR #237).

## Why this study

The authentic-publisher source audit (PR #238) passed. SEQ009-S (PR #239)
used 23,550 original TRAIN and 5,046 publisher VALIDATION trucks for
real next-readout forecasting. On that publisher VALIDATION set, persistence
achieved 0.047979 macro signed-log MAE versus 0.144903 for a trained
diagonal affine model. This is a legitimate *negative control*, not a
TAC-OSM performance result. Because SEQ009-S already inspected
publisher VALIDATION scores, this new training program **must not
repeatedly tune on publisher VALIDATION**.

## Falsifiable question

On *authentic* SCANIA Component X operational histories from the official
publisher, does a trained GRU residual next-observation predictor
beat persistence on **previously unseen TRAIN-internal vehicles**?
Does a TAC-inspired additional learned gate earn a benefit beyond that
plain GRU? The native deployed TAC-OSM/DEV061.5 agent is NOT these
recurrent models and no native TAC-specific superiority can follow.
This experiment is intentionally small to establish an honest starting
architecture and training pipeline. No synthetic task result qualifies.

## Source / identity / exposed information

- DOI 10.5878/bnh5-ka77, publisher version 3, CC BY 4.0.
- The actual official HTTPS v3 ZIP is SHA-256 hashed.
  ONLY train_operational_readouts.csv body is extracted and used.
  TRAIN repair labels, publisher VALIDATION/TEST sensor data and
  validation/test repair labels are never opened by this study.
- Split TRAIN vehicle IDs deterministically before outcome inspection:
  SHA256("seq010-v1-internal-split-20261019:" + vehicle_id)
  integer mod 5 == 0 => train-internal held-out; all other IDs train.
  No vehicle may contribute to both cohorts.
- Source numeric data are chronological within original vehicle.
  Reject invalid/missing IDs, nonnumeric timestamps, backward or
  tied within-truck observation timestamps.
- Fixed first 24 non-ID/time sensor columns from publisher TRAIN
  schema, chosen by column order before inspecting target values;
  all models see the same subset. Signed-log transform, clip inputs
  to [-12,+12], scale inputs by 1/12, explicit input masks.
  Targets retain un-clipped signed-log values.
- No next-readout time or post-readout information is in predictor inputs;
  irregular sampling is a known limitation.

## Pairs and training

Use at most the 8 immediately previous genuine readouts of the
same vehicle. Left padding is missing, with an explicit observed mask.
Predicted outcome is the ACTUAL next observed signed-log sensor vector.
The predictor produces a RESIDUAL added to the last observed sensor.
Eval requires both last predictor and next target observed, shared
identically among persistence/GRU/TAC-gate.

TRAIN origin selection: hash-based every ~31st eligible within-vehicle
transition, plus the final train-vehicle transition. These rules are
fixed without inspecting outcome. Held-out eval: final genuine
transition of each eligible TRAIN-internal held-out vehicle; never
train on its observations.

Models on exactly the same TRAIN data:
1. Zero-param persistence, next=last.
2. 32-hidden GRU over 8 observations with last-state residual head,
   initialized to zero (initially same as persistence).
3. Same GRU backbone plus one trainable sigmoid-gated local state
   correction at readout ("tac_gated_readout"). It is TAC-INSPIRED,
   not a native TAC-OSM/CASM/CDL implementation.
Both neural arms trained from scratch with fixed seeds
810101 and 810102, AdamW LR 0.001, weight decay 0.0001,
200 updates, batch64, gradient clip 1, smooth-L1 signed-log
residual loss beta0.1 and observed-target masking; no
intermediate state labels, teacher trajectories, repair costs,
simulator laws or oracle family IDs. No training recipe,
feature selection, model choice or stopping rule is determined
from held-out metrics.

## Metrics / primary decision

Report signed-log micro/macro MAE on same observed entries,
and persistence signed-log MAE. Use model-checkpoint SHA-256,
per-model train loss, real vehicle counts, transitions, measured
training wall time, raw source ZIP/CSV hashes and CPU details.
Smaller MAE is better. Per-seed comparison and failed arms
are reported regardless of result.

**Exploratory gate:** A GRU or gated-readout model beating persistence
on internal-heldout macro MAE at BOTH frozen seeds is encouraging
for further registered, stronger real-data study. It is NOT a
confirmed scientific architecture promotion. If neither beats
persistence, record the negative result, retain original weights,
and inspect why stable counters dominate before spending more
compute on BDH/SSM/executable memory.

Published VALIDATION/TEST never opened here. A follow-on
registered native-TAC experiment needs tuned tree/hazard,
trained GRU, SSM/attention comparisons, 3+ seeds for pilots,
bootstrap-by-truck uncertainty, measured CPU/RSS, and a locked
new heldout test protocol. Othello-GPT-style latent-state probes
and state interventions must be done separately and may NOT
be inferred from small MSE/MAE alone.

## Explicit nonclaims

This study does not measure time-to-recorded-repair, censored
failure-risk prediction, known action propensities, economic
intervention cost, alias association, procedural/executable
memory, BDH-scale local plasticity, learned causal dynamics,
safety or commercial readiness. Synthetic fixture unit tests
are permitted only for software correctness.
