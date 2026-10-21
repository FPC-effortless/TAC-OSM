# ME-031 — Three-way FIT-user cross-fitted residual state information

Registered 2026-10-21 BEFORE fits. Real-data mechanism diagnosis, not architecture
promotion or independent qualification. ME030 failed all gates; its joint head is
not adopted. Use the established ME028 native representation without changing its
fast/slow recurrence. No optimization of synthetic tasks. Preserve every attempt.

## Question and observable outcome

Does frozen native state help a finite nonlinear predictor beyond a shared
common-feature predictor, when neither the representation nor the prediction
probes have trained on the evaluated user's records? A positive result is evidence
of finite-family predictive value under this split/recipe; it is NOT a conditional
mutual-information estimate, causal effect, chronology proof, or Bayes-risk bound.

## Source and data-access boundary

Official KuaiRec CC BY4.0, Zenodo18164998; Gao et al CIKM2022,
DOI10.1145/3511808.3557220. Source ZIP SHA256
c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c.
Reuse ME022 authenticated snapshot SHA256
fea9675380d501e81b740a0c1e0cd8c99893863ba6bed9a7e6619a572cc21177;
verify12original checkpoint hashes and canonical fingerprints before fitting.
Only10776 original FIT windows (user%16==0, target before Aug01 2020) may enter
this study. Same128 genuine observed events, next offered item/duration and binary
watch_ratio>=1 target; no padding, new missingness imputation, synthetic labels,
new modalities or outcome fabrication. Inherit original source window exclusions,
entity/timestamp checks and censoring. Metadata is log1p(duration)/12, observed
feedback, gap. The cache validator reads known historical DEV for source-integrity
fingerprints only: DEV is never trained on, scored, selected on or reported here.
No future dates or reserved user cohorts opened.

This diagnostic subdivides original FIT; it does not reassign the program's
frozen train/test boundary or rename used FIT as an independent test. Those FIT
records have influenced previous architecture design. Current within-study user
isolation cannot undo that historical design exposure.

## Locked user roles, all three rotations

Group g=(integer_user_ID//16)%3, independent of outcomes. For rotation r=0,1,2:
representation users R:g=r; readout users S:g=(r+1)%3; evaluation users T:g=(r+2)%3.
Roles are user-disjoint, exhaustive; every FIT window is evaluated once per seed.
Window overlaps within user never cross a role. No user identifier is an input.
Publish exact row indices and user lists before model fitting. Each seed14001,
14002,14003 has fresh representation/base/probes per rotation (9encoders,
9shared bases,18residual probes=36fits). No checkpoint or weights from prior studies.

All36models must be frozen and hash-indexed before ANY outer-evaluation metric is
computed. No tuning, best-step selection, early stopping, retry or model amendment.
Interrupted/failing fits keep partial checkpoints and do not enter the primary pool.

## Representation and feature construction

Representation=ME028 additive nonlinear native TAC,135377 trained parameters.
Same native32fast+32slow, split RMS epsilon.01, original embedding16000x8. Exactly
300updates x128 R histories,4real next targets32/64/96/128, loss weights
1/6,1/6,1/6,1/2. AdamW body.001, head layers.006, decay1e-4, clip1, seed14001–3,
sampler seed+300. Duration mean/std and smoothed item popularity fit on R only.
Save source/role/weight provenance; no T/S outcomes enter representation fitting.

Freeze encoder. Extract four causal features per S/T history:64state h,26common
x, one prior logit. Common channels remain current item embedding8, standardized
stored duration1, EWMAs5, R-only popularity1 and pooled event embedding/meta11.
At cutoffs32/64/96, next item/duration are known query inputs; next response and
later events must not enter these features. Export causal extraction in one pass.
Standardize first90 coordinates using S feature moments only, scale floor.05;
prior logit stays unstandardized. No label in this standardization and no T moments.

## Identical frozen base and two residual arms

Shared base p0: logit(prior32)+26->43 tanh->1 common-only predictor,1205parameters.
Train500updates x128 S histories, four target weights as above. AdamW .003,
decay1e-4, clip1, initial final weights/bias zero. Head seed=representation_seed+
1000*rotation+10000; sampler=head_seed+300. Freeze base, cache exact S/T logits.

Residual arms, both initialized to zero correction and both500updates x128 S,
same targets/weights/optimizer/clip as base, same seed=head_seed+10000 and matching
sample schedules:
- common_residual:26->43 tanh->1 (1205 active trainable parameters).
- state_residual:[h,x]90->13 tanh->1 (1197 active trainable parameters).

Predictions are sigmoid(frozen_base_logit+residual). Param gap8/1197=.6684%.
Both are nonlinear one-hidden-layer predictors; width differs to match trainable
counts. They are NOT identical function classes or proven approximation-equivalent.
Total stored pipeline counts: encoder135377+base1205+residual1197/1205.
Encoder's401 original readout parameters serve pretraining but are unused by the
probe prediction path. The common arm uses the common-feature branch of the shared
encoder, while the state arm also needs recurrence. Report this deployment/compute
asymmetry rather than padding active parameter counts. Shared representation/base
training is fully charged to EACH residual arm; no free pretraining advantage.

Only residual probes differ in state access. This deliberate information ablation
is the diagnostic treatment, not an equal-input architecture ranking. Shared base
and prior32 are unmatched practical reference predictors, not additional selected
arms. No mean-state intervention or permuted/synthetic primary outcome.

## Resources, receipt policy and endpoints

CPU single thread (hardware/software recorded); encoder max180charged CPU sec,
each base/residual max30charged CPU sec, full shared pipeline+residual max240CPU
sec. Same caps, step ceilings/exposure and no tuning privilege for either arm;
unused caps do not imply equal spend. Charge source verification, R preprocessing,
model/optimizer setup, sampling/loss/forward/back/update, all frozen feature building,
S normalization and base fitting; report stage costs and whole-study CPU/wall/RSS.
Record every update FIT loss/CPU, save100/200/300encoder and100/250/500probe progress
as latest partials; only300encoder and500base/residual are final endpoints. Preserve
all per-step traces. No outer-evaluation learning curves: freeze first, score once.
Rotate residual execution order by seed/rotation. Measure inference and profiler
operator-coverage estimates separately; do not call estimates full FLOPs.

Exposure is matched per residual arm: representation38400 histories/153600targets,
base64000/256000, residual64000/256000 per seed/rotation. Actual consumed compute
is expected to differ. This is the exposure-matched diagnostic lane; actual-compute
parity is declared FAIL, no compute-matched or fully matched superiority claim.
The shared stages are computed once but charged fully to each arm. Record total
physical study spend separately to avoid double-counting actual execution.

## Primary analysis and frozen gates

Evaluate ONLY final128-event target per original FIT window in its T rotation;
aggregate predictions in original FIT row order for each seed. Report Brier,
logloss,AUC,ECE10 for prior,base,both residuals, per seed/rotation and pooled.
Primary paired contrast=state_residual minus common_residual Brier. Require:
1. all3 pooled-seed differences negative and mean <= -0.0005;
2. state beats shared base in all3 pooled seeds and mean <= -0.0005;
3. state beats common residual in >=2/3 rotations for EACH seed;
4. mean pooled ECE10 deterioration vs common residual <=0.003;
5. 95% user-clustered paired bootstrap upper bound below0 (1000draws seed31031).
Only all5 gates yields a finite-family state candidate for further engineering.
No promotion even on success: independent future-cohort/parity qualification is
separate. Bootstrap averages losses across fixed seeds, resamples original users
and preserves windows. It is conditional on trained models and overlapping rotated
training roles, not full training/refitting uncertainty. No decision-cost endpoint:
KuaiRec logged factual reactions do not identify intervention outcomes or costs.

Save36 weights/Adam/RNG,36TorchScript exports,9frozen feature caches/normalizers,
role indices, clocks, original predictions and all provenance. Before replay,
verify frozen payload digests, then rebuild features from authenticated original
FIT and frozen encoders, reproduce checkpoint and export predictions, compare
metrics/contrasts and fresh-process exports, test corruption fails before source.
Source and protocol commit must be pushed before first fit. Every negative result
is retained; no post-hoc width/LR/cap/threshold change.
