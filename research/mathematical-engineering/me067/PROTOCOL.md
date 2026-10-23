# ME-067 — Are native fast / slow / associative memory states genuinely useful?

## Scientific status

**Diagnostic-only, original real robot observations; no new model fits, performance claims, or new independent holdout.** Preserve the original ME-066 scientific **FAIL**, including all twenty Git-owned trainable model/optimizer/RNG checkpoints, original publisher outcomes, and replay receipts. ME-064 spray-nozzle and ME-066 charger-installation have already been exposed; they are not new independent confirmation. The last original ME-066 reserved `replace_tissue_box_and_pull_out_first_tissue` must stay SEALED. We should not spend it on an arbitrary unqualified new neural router.

## Physical evidence motivating the investigation

Authentic ME-066 80-episode charger transfer physical MAE (lower better): persistence 0.2453838432, matched GRU 0.3815281845, selective diagonal SSM 0.3989837375, compact RSSM 0.4034837283, native TAC 0.4105902042, FIT servo 1.2005298074. TAC beat GRU on 1/5 paired independent seeds. Its nominal 90% source-CAL whole-episode calibration covered only 77.75%, with physical width 23.18. No causal improvement, safe autonomy or uncertainty calibration is proven.

A capacity-increasing architecture redesign is premature until the current model's fast, slow and 8×8 associative state can be shown to matter in predictions. Existing ME-066 normalized physical action/observation decoder mixes persistence / physical servo / neural candidate; it can mask memory failure.

## Registered immutable hypothesis and interventions

On exactly the original 5 ME-066 native TAC and 5 GRU trained weight checkpoints, for each of the **original 40 exposed source-CAL** and **80 already exposed charger-installation** episodes, replay the exact median eligible 64-frame history and frozen observed next-16 action prefix. Keep original human-takeover censoring and source FIT normalization. Before the forecast head, intervene on **readout state only**:

- `full`: no intervention; its MAE and observation origin must numerically match the published ME-066 score for every model and episode.
- `association_zero`: zero only native associative 8x8 latent.
- `slow_zero`: zero only native slow 32D latent.
- `fast_zero`: zero only native fast 32D latent.
- `fast_slow_zero`: zero both fast and slow latent readouts; retain associative, velocity and innovation.
- `kinematics_zero`: zero observed velocity/innovation reliability traces, retain native memory. This is a same-budget negative/positive diagnostic, not a trainable new module.
- `all_hidden_zero`: zero all latent and kinematic/reliability traces.
- GRU control: full, fast-zero, kinematics-zero, all-hidden-zero.

These are **readout-only lesions**: do not alter the recurrence during genuine past observations; do not retrain, reset model parameters, fit the exposed target, or claim these out-of-distribution hidden-state perturbations are a faithful deployment ablation. The causal inference is only on the frozen network function, not physical world interventions.

## Exact evidence and gating

Verify: 20 original Git-owned training weight SHA bytes, original ME-066 independent numeric replay PASS, each publisher original real trajectory per-episode SHA, same scoring origins and uncensored live steps, full-model replay within predeclared absolute 1e-4. Abort on any mismatch.

Measure for each lesion and each of five original seeds absolute physical MAE and lesion-minus-full error. On the 80 original previously revealed target episodes report paired block-bootstrap 95% CI over **20 four-episode recording blocks**, using fixed seed 67067 / 20,000 block resamples, and count individual independent seed directions. Report the same descriptive per-seed results on the already exposed CAL 40 source episodes. A positive or negative lesion delta is a mechanistic sensitivity finding **only**, not proof of why the memory learned its state or independent generalization.

**Promotion:** none. Further trainable variants require a separate original real-FIT source-only preflight, equally matched trainable parameters and profiler FLOPs, paired data/budgets, independent future test, in-repository original model checkpoint+optimizer/RNG byte artifacts and clean checkout restoration. Concept-CED may be a candidate only after solving its known unmatched compute; it is not qualified by these lesions.

**Product boundary:** The unqualified physical forecasts remain advisory-only in Woyengi; prediction confidence or model access cannot grant authority to execute a requested action. Long-term agent memory/decision cost and causal intervention value require a new independent operational evaluation, not this replay.
