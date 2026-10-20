# REAL-MAINT-004 — actual handpump repair bundles and measured outcomes

Preregistration. No model fitting or outcome-distribution inspection before the
fitting commit. Independent from DEV068; no source or evidence changes there.

## Scope and source

Factual prediction of the reported seconds needed to fill a20L bucket AFTER a
recorded rehabilitation, given pre-intervention functionality and the recorded
repair/chlorination bundle. This is not a causal effect estimate, counterfactual
repair recommender, binary repaired-state certificate, factory agent or economic
qualification. Prices, inspection fees, independent collector/follow-up timing,
labor and actual parts availability are unknown. Supplied actions are recorded
actions, not proven complete pre-authorized plans. Their selection may be
confounded with severity, location and technician behavior. No policy is deployed.

Openwashdata floodchlorinationsurvey0.1.0, USAID response survey, Mulanje Malawi
2019–2020; Emmanuel Mhango and contributors. CC BY4.0; DOI10.5281/zenodo.15846578;
publisher commit `a2755213c841fdfc7b65aaaff0b91016296f2985`.
CSV SHA256 `6062eaa4c758d07820e7a13c4c31f783970dbf5598fa554afee2de5a36d58258`.
Dictionary explicitly names bucket time in SECONDS; do not substitute stroke
count or convert this to liters/second without a separately declared interface.
Do not infer independently measured dates from submission timestamps.

## Sealing and grouping

Only schema, licence, presence counts and date/name metadata have been examined.
No new target distribution or prediction score has been read. Three publisher
preview examples were visible earlier; defensively exclude EVERY waterpoint name
appearing in the first10CSVrows (7normalized names). No names, GPS, photos, comments,
submitter metadata, timestamps or post-repair proxy enters the model.

Group by normalized waterpoint name (case/whitespace folded), admitting possible
collisions as a limitation. Sort groups by maximum reported rehabilitation date;
60% and80% positions define date boundaries. Train groups wholly before/equal
boundary1; validation groups wholly after1 and before/equal2; test groups wholly
after2. Purge spanning groups, preview groups, missing names and invalid dates.
Metadata-only source audit fixes boundaries2019-07-08/2020-02-21:180trainrows /
129groups,70validationrows/45groups,60testrows/39groups;7spanning rows/3groups purged.
These are unlabeled eligibility counts, not final valid target counts.

Do not parse/summarize test targets until the explicit source/selection-hashed
TEST-OPENED marker. The source is public, so this is an auditable procedural seal,
not cryptographic inaccessibility. No re-split or test-dependent exclusions.
Targets must be finite positive numeric seconds; malformed/nonpositive targets
excluded once by that fixed rule with row IDs/reasons retained. No outlier clipping,
residual trimming or missing-target imputation. Unknown replacement entries are
missing, never silently treated as no repair.

## Features, models and controls

Two independently retrained neural arms:
- `prestate`: only explicitly pre-intervention functionality.
- `repair`: same prestate plus all56`replaced_*` fields and chlorination_done.

Timing-ambiguous ratings/depths/condition fields, after-functionality, strokes and
bucket times are excluded. Categories use TRAIN-only vocabulary with separate
missing/unseen indicators. Count/length fields are nonnegative log1p values,
TRAIN-median imputation plus missing flag; all-missing field fallback0 is a
numerical fill, not a factual zero. All transforms are TRAIN-only. Training global
mean/std standardization has unit scale when std<1e-8.

Both neural arms: two16-wide GELU layers, positive exponential residual around
TRAIN-median target, zero output initialization starts exactly at median baseline.
Proper absolute loss in raw seconds scaled by TRAIN median (minimum1); batch32,
AdamW, clip1, float64, deterministic eager, single-thread, EMA .995,2,048updates,
final EMA only for selection. Raw and EMA weights and learning curves preserved.
No best-epoch selection. Candidate is always `repair`, not a post-test best arm.

Conventional references: TRAIN median, TRAIN prestatus-conditioned median, and
log-target ridge on identical repair features at alpha .1/1/10. Select reference
by validation MAE, lexical tie. Keep every reference including failures/nonfinite
predictions; those fail eligibility, never justify dropping rows.

Pilot grid: LR .001/.003 × weightdecay .001/.01 × arms2 × fresh initialization
seeds97400/97421/97422 (24fits). For EACH arm select mean validation MAE over its
three pilot seeds, ties recipe index. Require selected repair mean validation MAE
at least5% lower than selected reference. Otherwise VALIDATION_REJECTED; no fresh
fits and no test parsing. Entry also requires at least80validtraining and20valid
validation targets, five distinct training target values, and20unlabeled test
groups, else SOURCE_HEADROOM_REJECTED. No claim of minimum Bayes oracle headroom.

If entry passes, fit BOTH arms at ALL20fresh training seeds97401–97420. Selected
repair must be no worse than the locked reference on validation in>=18/20seeds
before test opening. Otherwise preserve all fits and close VALIDATION_REJECTED.
Total local CPU cap480seconds, no paid resources. Caps/errors block promotion;
do not silently reduce update/seed budgets. No test-driven recipe repair.

## Sealed assessment and acceptance

If opened, report every valid target, every seed's raw MAE(seconds), baseline-
scaled error and all group/row counts. Candidate must improve mean MAE by>=5%
against BOTH training-median and locked-reference controls, with >=18/20individual
seeds no worse than each. Hierarchical paired bootstrap:10,000resamples of20seeds
and held-out GROUPS with replacement, preserve all rows per sampled group, RNG974004.
Require simultaneous one-sided95% upper differences<=0 (Bonferroni two comparisons,
97.5th quantile). At least20validtestgroups needed; otherwise no qualification.
No20seed claim from pilots, no independent-trial claim for repeat records or seeds.

Report prestate-vs-repair contrast separately: added action information must earn
its complexity; observational predictive contribution is not causal necessity or
CASM attribution. Also report TRAIN-only median/ridge predictions, worst groups,
tail absolute errors, target ranges and feature missingness without dropping bad
cases. No probabilities or reliability guarantees from point forecasts.

Optional90% symmetric interval: calibration uses maximum absolute residual per
validation GROUP and finite-sample ceil((n+1)*.9)rank; lower bound0. Report empirical
row/group coverage, width and sample counts. Exchangeability across dates is not
assumed/proven, so this is a diagnostic, not guaranteed calibrated coverage.

## Reproduction and release

Commit this protocol and real fitting source before fitting. Save hashes, selected
recipes, all raw/final weights, curves, referencefits, full inputs/targets ONLY for
opened splits, source attribution, source-file hashes and environment/CPU usage.
Standalone Torch-only export if a completed model exists; label failed/diagnostic
models unqualified. Replay every saved prediction independently, recheck no ID/time/
post-outcome feature leakage, preserve failed pilots/tests. Existing DEV058–068
and REAL002/003 evidence and MetroPT July B remain intact. Broader goal ACTIVE;
even numerical success here is a small same-survey factual point predictor,
not commercially validated maintenance intelligence.
