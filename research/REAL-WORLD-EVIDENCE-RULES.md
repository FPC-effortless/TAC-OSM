# TAC-OSM research policy: real-world evidence is the promotion gate

**Effective:** 2026-10-19. **Status:** standing rule for every NEW research
registration, model-selection decision, architecture claim, and research handoff.
This changes prospective evaluation rules; historical experiments, their locked
protocols, failed runs, and evidence remain intact.

## Non-negotiable requirements

1. **Real-world, independently sourced observations are the default training
   and the REQUIRED primary evaluation.** New capability experiments must target
   externally collected records from actual operational equipment, actions,
   people, institutions, or physical systems. Identify the exact source, version,
   licence, dates, entity identity, measurements, event definitions, missingness,
   collection method, and file/checksum provenance before fitting.
2. **No synthetic/simulated benchmark can establish TAC-OSM capability,
   reliability, architectural superiority, world-model performance, or
   commercial readiness.** Generated event streams, scripted worlds, handcrafted
   labels, simulated sensor physics, benchmark oracles and injected rewards may
   only be used as clearly labelled unit/integration tests, negative controls or
   mechanism diagnostics. Do not spend a new standalone research registration
   optimizing synthetic task accuracy. A passing synthetic suite has *zero*
   weight in primary scientific or product promotion.
3. **Physical laboratory measurements are not field deployment.** A real
   instrument with an experimentally injected fault is empirical physical data,
   but any performance claim must say it was measured on controlled apparatus,
   not on naturally occurring operational failures.
4. **Evaluate the exact information available at prediction/decision time.**
   Preserve time order, source entity boundaries, censoring, operational
   missingness, label availability and shift. No future leakage, post-action
   fields, oracle belief, answer-bearing identifiers, hindsight-only labels,
   fabricated alternative outcomes, or arbitrary train/test reassignment.
5. **Separate prediction, decisions and causality.** Predicting a repair label
   or an observed treatment outcome is *not* evidence that a proposed repair
   prevents failure. Estimates of real-world intervention benefit require
   identifiable comparisons (randomized policy/propensities, justified
   observational identification, or prospective authorized measurement).
   Publisher-provided evaluation-cost matrices are **expert-defined proxy
   costs**, not measured savings or actual inspection/repair invoices.
6. **Pre-register a falsifiable real-data protocol BEFORE any held-out labels
   are opened:** hypothesis, approved cohort/exclusions, measurement windows,
   entity/time/site split, access budget, preprocessing fit only on train,
   missingness/censoring handling, baselines, metrics, thresholds, seed count,
   risk stratification and decision rule. Keep test frozen; all attempted runs,
   failures and negative results remain in the ledger.
7. **Compete against the simplest good solution, not only TAC ablations.**
   Depending on the task, include last-observation/persistence, useful
   regression/AR or survival/hazard baselines, calibrated trees, GRU/SSM,
   attention if relevant, and do-nothing/always-act policies wherever actions
   truly exist. Match information, horizon, training budget and realistic
   end-to-end compute; measure memory, latency, calibration and total cost.
8. **Report the limits explicitly.** Real-world *offline prediction* is not
   real-world *policy utility*, causal counterfactual ability or autonomy.
   Do not claim commercial qualification without observed inspection/action
   logs, actual prices and independently verified post-action physical outcomes,
   plus out-of-time/site evaluation. The missing fields are blockers, never
   blanks to fill with synthetic proxies.
9. **Existing synthetic studies remain reproducible historical diagnostics.**
   Freeze their findings and preserve their hashes/checkpoints, but do not
   extend synthetic-only pilots or promote them to primary evidence. In
   particular, the SEQ-TRANSFER-006 synthetic pilot on PR #232 is engineering
   evidence of a narrow, supervised ability only; its reserved synthetic TEST
   cannot satisfy this policy or be treated as a world-model milestone.

## First primary real-world dataset: SCANIA Component X

- **Original operational data:** heavy-duty truck fleet with 33,641
  anonymized vehicles, irregular longitudinal readouts, specifications, and
  workshop records of Component X repair/replacement.
- **Source and fixed version:** Swedish National Data Service / Researchdata,
  version 3 (2025-04-09), DOI https://doi.org/10.5878/bnh5-ka77 ;
  https://researchdata.se/en/catalogue/dataset/2024-34/3
  (11 files, about 1.54 GiB; CC BY 4.0). Document all file hashes after
  actually acquiring them; a catalogue entry is NOT download verification.
- **Publisher split:** train 23,550 trucks / 1,122,452 readouts; validation
  5,046 / 196,227; test 5,045 / 198,140. Retain disjoint vehicle IDs and
  the published validation/test partition. Train has 2,272 observed repairs
  and 21,278 right-censored/nonrepair study endpoints. Validation/test labels
  classify lead time to a recorded repair in five classes, with severe
  imbalance; a no-repair observation is not guaranteed permanent health.
- **Task A (primary):** prospective *as-of-readout* predictions of
  time-to-observed-repair / failure-associated replacement at registered
  lead-time windows, censoring-aware risk, uncertainty/calibration, and
  per-class recall/precision. Never use later readouts or target labels
  for the prediction feature vector.
- **Task B (model-specific):** evaluate whether recurrent/entity-local
  memory improves *the same* true observed-risk prediction against
  last-readout and equal-information time-series baselines; report skill
  by history depth, staleness, missingness and unseen vehicles.
  This dataset does not directly label reversals/TOGGLE semantics or
  two-entity relationships: do not infer those capabilities from it.
- **Decision evaluation:** report the SCANIA challenge's **expert-specified
  asymmetric misclassification cost matrix** as offline decision *proxy
  cost*, always next to raw class metrics, PR-AUC, calibrated loss and
  false-alert/missed-alert counts. Do not call its scores actual fleet
  savings. No actionable proof of repair efficacy, counterfactual
  interventions, or independently verified post-repair function is in
  the released training contract.
- **Important limitations:** published values are anonymized; operational
  counters and rates, absolute dates, readout rates and repair frequencies
  may be perturbed. A random vehicle-held-out split is NOT an out-of-calendar-time
  generalization test. Independently sourced operational data must corroborate
  findings before field robustness claims.
- **Baseline shortlist:** calibrated historical prevalence/constant-risk,
  last-readout logistic or LightGBM/CatBoost, simple survival model,
  GRU/temporal baseline, then TAC-OSM. Choose details by pre-registration.
- **Status:** dataset and schema verified from publisher descriptions;
  no completed local raw-data download, model fit or performance claim
  is established by this policy document.

Publisher: Kharazian et al., *Scientific Data* 12, 493 (2025),
https://doi.org/10.1038/s41597-025-04802-6

## Independent corroboration / action-outcome path

- **Backblaze Drive Stats**:
  https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data .
  Actual data-centre SMART snapshots and drive-failure indicators across
  calendar years. Use untouched forward-time quarters for independent
  operational failure-risk generalization. Hard-drive removal is not proof
  of a counterfactual prevented failure; there are no measured repair actions
  or inspection charges in this dataset. Source attribution/reuse restrictions
  apply.
- **MetroPT-3 (UCI 791)**:
  https://archive.ics.uci.edu/dataset/791/metropt+3+dataset .
  Actual train-compressor readings, four operator-reported failure windows;
  too few independent faults to support broad commercial reliability.
  Sensor prediction/diagnostic external test only. Preserve known ambiguity
  in recorded maintenance times. CC BY 4.0.
- **FHWA LTPP / Malawi waterpoint follow-up**:
  see research/commercial/DATA-SOURCES.md for already audited real maintenance,
  reported costs, measured physical states and their exact incompleteness.
  Potential action/outcome studies; neither qualifies as causal success
  without source-specific timelines, independent verification and confounding
  analysis. Never splice unrelated datasets to fabricate joint actions/costs.
- **Real logged-decision benchmark (separate domain, not maintenance):**
  Open Bandit Dataset, https://huggingface.co/datasets/zozonext/open-bandit ,
  supplies recorded choices, logging propensities and observed click reward
  on a retail platform. Eligible for a *separately registered* offline
  policy-evaluation test with correct support/positivity checks, not for
  long-horizon maintenance or physical repair claims. CC BY 4.0.

## Execution order and acceptance record

1. Source and checksum-verify SCANIA version 3 without using TEST outcomes
   for selection; audit schema, censoring, target construction, leakage and
   irrecoverable perturbations. Preserve existing supplier splits.
2. Register the readout-as-of risk prediction experiment, its real-data
   baseline matrix, calibrated performance and asymmetric *proxy* cost
   interpretation, then fit/train only after registration.
3. Independently verify on held-out trucks; separately test forward-time
   drift/failure forecasting with Backblaze calendar-based splits. Preserve
   every failed fit and all negative findings.
4. Pursue real repair/action-cost/verified physical outcomes as a distinct
   field utility gate. No simulator or arbitrary cost schedule can close it.

**Claim ledger:** engineering diagnostic / real operational prediction /
observational offline decision proxy / identified action effect /
independently verified field cost utility are different evidence categories.
Only the last category qualifies commercial claims, after its own preregistered
reliability and out-of-site/time gates.
