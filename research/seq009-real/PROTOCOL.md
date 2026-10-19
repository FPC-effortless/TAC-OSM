# SEQ-009R — Real-world source qualification and longitudinal state learning

**Registration date:** 2026-10-19. **Governed by** research/REAL-WORLD-EVIDENCE-RULES.md
in research policy PR #237. **Status:** acquisition-only first stage, no model
trained and no TEST evaluation. This branch is stacked on the policy branch,
not on SEQ008, and does not modify any synthetic-checkpoint research.

## Scientific hypothesis and primary real task

On independently collected SCANIA Component X truck histories, does a
trainable persistent latent-state model (existing TAC-OSM model, entity
routed GRU, or later a plastic SSM) improve future *observed repair event*
risk prediction and calibration compared with a tuned last-readout
LightGBM/logistic/survival baseline using the **same information at as-of
time**? Full benchmark requires actual authentic sensor readouts and
recorded repair/censoring outcomes. Earlier synthetic SEQ006-008 results
have zero primary promotion weight under the new policy.

**Source:** Swedish Researchdata SND 2024-34 version 3, dataset DOI
https://doi.org/10.5878/bnh5-ka77, CC BY 4.0,
https://researchdata.se/en/catalogue/dataset/2024-34/3.
Published SCANIA fleet measurements have vehicle-held-out partitions:
TRAIN 23,550, VALIDATION 5,046, reserved TEST 5,045 vehicles.
TRAIN has 2,272 recorded Component X repairs in the study and 21,278
censored/nonrepair endpoints. These figures are publisher statements
until original files are independently acquired and checked.

## Stage 0 — authentic-source acquisition gate

Download exactly these small version-3 files from the publisher:
train_tte.csv, train_specifications.csv, validation_labels.csv.
Record original HTTPS source, byte counts and SHA256. Audit actual columns,
unique vehicle IDs, durations, repair/censor codes, validation class counts,
train-specification joins and train-validation vehicle nonoverlap.
Fail closed if raw bytes are unavailable or schema differs from documentation.
The scripts explicitly forbid TEST source names. Training/test readout CSVs
are not yet acquired. Summaries may be archived but raw publisher CSVs
need not be redistributed; keep original DOI attribution.

**No empirical model claims** arise from Stage 0, even if all publisher
row counts match. A fake CSV unit-test fixture only checks code correctness.

## Stage 1 — full authentic longitudinal source (separate continuation)

Acquire and SHA256-verify train_operational_readouts.csv and
validation_operational_readouts.csv (large), archive manifest and
schema/data-quality reports before any model fitting. Keep
test_operational_readouts.csv and test_labels.csv untouched during
model selection. Verify:
- Actual vehicle_id/time_step field names and unique per-vehicle sequences.
- Nondecreasing within-vehicle time after stable sorting; duplicate
  timestamps, missing bins/counters and time perturbation.
- Train TTE event/censor horizon truly matches available readout coverage.
- Fixed train/validation vehicle partition, exact 0..4 publisher labels.
- Counter/histogram semantics and any fields only known after repair.
- Empirical as-of visibility; never use later readouts, endpoint times,
  future masks, validation classes or repair status as predictor features.

### Outcome contract

Task A: binary horizon risk at last validation readout, class_label
in {1,2,3,4} representing repair recorded within a nominal 48 time-step
window, class_label 0 representing no recorded repair in that nominal
window. This is a surrogate event-risk outcome, not true physical failure,
and not evidence that repair was medically/mechanically necessary.
Secondary: the five publisher lead-time classes.

TRAIN train_tte offers one repair/censor endpoint per truck, not directly
the same format as final-readout validation labels. Construct
training prediction origins only from actual within-vehicle readouts;
a positive occurs if a recorded repair happens <=48 steps after origin
and a negative only if there is known follow-up >=48 without repair.
Exclude origins with insufficient observation/censoring horizon.
Document selection weights and time-step anomalies. Avoid creating
labels that presume outcomes after censoring.

Fix exact training-origin sampling, censor semantics, feature extraction,
metric thresholds, classifier/survival details, resource cap and training
seeds **before** unsealing the final test. Any origin sampling informed by
event time, even to ensure label observability, must be disclosed and
reweighting/calibration checked against the publisher validation cohort.

### Baselines/ablation ladder

B0: training prior/no-history; B1: last-readout tree (LightGBM);
B2: last-readout logistic; B3: survival/hazard model that handles censoring;
B4: sequence GRU; B5: TAC-OSM with native recurrent world-state features
and same observation horizon/feature budget. Add learned alias-binding/
role-filler memory only if genuine observations contain identifier
ambiguity; publicly stable vehicle IDs alone cannot show discovered
identity. Plastic SSM/BDH-style and executable memory optional *later*
as separate, equal-information ablations.

### Sequence-prediction world-state objective (Othello-GPT inspiration)

Add auxiliary next-authentic-readout prediction over all observed
training sequences, with explicit irregular time gaps and missingness.
Train without future observations leaking into state used for risk
predictions. Test latent-state *probes* for current operational condition,
future-repair prediction and time-gap robustness. Use latent interventions
and matched ablations to test whether states causally influence predicted
sensor trajectories. Predicting sequences alone does NOT establish
a causally correct world model; all Othello-style probes are secondary
to genuine held-out risk metrics.

### Real outcome / uncertainty metrics

Report PR-AUC for near-event minority, ROC-AUC, per-class recall/precision,
confusion matrix, Brier/log-loss, calibration/expected calibration error,
false alarms vs missed recorded repairs at train/validation-frozen thresholds,
censor-specific cohort coverage and sample counts. Report publisher
expert cost matrix separately as **proxy** (only after validating its actual
versioned values), not dollars saved. Compare end-to-end CPU, peak RSS,
data-preparation wall time, trained checkpoint size and inference latency.
Perform paired bootstrap confidence intervals resampling by truck,
never by individual readout. At least three fixed training seeds for
initial validation, twenty if enough actual headroom is observed to warrant
held-out architectural qualification. Report every fit and failure.

Publisher test split stays sealed until models/thresholds and ledger
are frozen. Randomly partitioned trucks do NOT establish future-calendar
generalization. Backblaze forward-calendar years are an independent task
after the SCANIA primary protocol, not a substitute for its held-out test.

### Promotion and causal restrictions

A novel mechanism must outperform a tuned conventional baseline on
independent real-world held-out vehicles with materially improved
calibrated risk prediction and no catastrophic minority-class regression;
the exact numerical gate is to be frozen before final TEST access.
No observed action effects, independent post-repair function or actual
monetary invoices are supplied in this dataset. Neither a model's
predictions nor expert proxy costs can establish causal economic benefit.

## Current claim ledger

- Real source acquired/hashed: **UNKNOWN until Stage 0 CI completes**.
- Full longitudinal operational histories acquired: **NO**.
- Primary model fits and prediction scores: **NO**.
- Held-out test opened: **NO**.
- Learned dynamic identities/aliases: **NOT EVIDENCED**.
- Commercial / causal repair utility: **NOT EVIDENCED**.
