# SEQ009-S — Real SCANIA next-readout prediction, original results

**Status: COMPLETED, AUTHENTIC REAL DATA, NEGATIVE LEARNED-BASELINE RESULT.**
Original GitHub Actions: https://github.com/FPC-effortless/TAC-OSM/actions/runs/39315568680
Preregistered study source commit:
654a98b5613199b2774a6d1cd7858fd91f120d6c

## Authentic source and cohort

- SCANIA Component X publisher version 3, DOI 10.5878/bnh5-ka77.
- Full original ZIP streamed and SHA256 hashed at
  https://api.researchdata.se/dataset/2024-34/3/file/zip .
- Selected train_operational_readouts.csv and
  validation_operational_readouts.csv are individually hashed.
  Original source and results manifests archived in artifact
  seq009-real-nextreadout-original-results, ID 12218460954.
  Raw publisher CSV/ZIP data not redistributed in the PR.
- Training: genuine consecutive real readouts from 23,550 trucks.
  Validation: final observed readout transition for 5,046
  original publisher-held-out trucks.
- Train/validation IDs checked disjoint. Original TEST data
  and labels not opened. No generated records used for fitting.

## Preregistered model results

Two models evaluated on exactly the same genuine sensor targets,
same signed-log transform and same jointly nonmissing component mask.
Lower MAE is better.

| Metric | Persistence | Trained sensor-wise affine AR(1) |
|---|---:|---:|
| Signed-log MAE (macro) | **0.04797924484** | 0.14490310617 |
| Signed-log MAE (micro) | **0.04798738695** | 0.14500804754 |
| Train-change-scaled MAE (macro) | **0.50805914454** | 1.50448314552 |
| Minimum observed validation targets per sensor | 4,975 | 4,975 |

Trained affine dynamics has about 3.02 times HIGHER macro
signed-log MAE than persistence. **The learned affine baseline fails.**
This is a measured negative result on authentic operational data.
It does not imply that a stronger GRU, LightGBM, TAC-OSM,
selective SSM or any other learned sequence architecture
must perform worse than persistence.

## Verification and exact limitations

- CI source/model preflight and offline unit tests: PASS.
- Original real version-3 publisher acquisition and source hashing: PASS.
- Real model fitting, held-out VALIDATION evaluation and output audit: PASS.
- Original evidence archive: artifact ID 12218460954.
- Independent source+weight replay using a separate implementation:
  NOT PERFORMED. CI pass is not independent scientific confirmation.
- Reserved TEST performance: NOT EVALUATED; untouched.
- This is a next-readout task on irregular real telemetry,
  not a time-to-recorded-repair or censor-aware failure-risk model.
  Train/validation hold-out was by truck, not by future calendar year.
- Sensor counters/histograms and distributional shift may make
  persistence unusually strong. Missing predictor/target values were
  omitted symmetrically for both models.
- The autoregressor has only independent per-feature linear dynamics,
  with no world-state latent probe or attention, no TAC neural
  architecture, no identity-alias inference or inference plasticity.
- No real repair action, verified benefit or economic decision cost
  evaluated; no commercial or architecture promotion.

## Next real-data mechanism gate

**Do not retune the original validation based on its scores.**
Preregister a fresh TRAIN-internal vehicle split for any
hyperparameter search, and preserve the published final TEST.
Compare persistence, robust multivariate/temporal references,
GRU, learned TAC and selective SSM under equal information
on actual truck histories, with memory/latency and time-gap
stratifications. Othello-GPT motivates additional latent
probes and interventions, not an automatic world-model claim.

Recorded repair event risk from SCANIA train_tte and publisher
validation classes is a SEPARATE censoring-aware task with
its own source-verified cohort and five-class evaluation protocol.
Backblaze future-calendar-time validation is separately required
before time-shift generalization claims.

**Outcome: strong simple baseline, negative learned-AR control,
no architecture promotion or causal utility claim.**
