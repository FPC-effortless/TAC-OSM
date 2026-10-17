# DEV-049A: reliability-constrained development results

All 30 fresh fits (three objectives × ten seeds, 216 matched updates) completed. Checkpoint predictions, calibration/certification decisions and physical evaluation outcomes passed replay. These are development findings, not confirmatory or deployment evidence.

| Objective / policy | Mean cost/entity ± seed SD | Failures | Unnecessary services |
|---|---:|---:|---:|
| forecast/cost_tuned_threshold | 0.527127 ± 0.015185 | 61 | 340 |
| forecast/zero_calibration_failures | 0.714540 ± 0.245860 | 0 | 4841 |
| forecast/certified_or_service_all | 0.714540 ± 0.245860 | 0 | 4841 |
| joint/cost_tuned_threshold | 0.530339 ± 0.016234 | 135 | 192 |
| joint/zero_calibration_failures | 0.858594 ± 0.227929 | 2 | 8154 |
| joint/certified_or_service_all | 0.906727 ± 0.196876 | 0 | 9269 |
| failure_penalized/cost_tuned_threshold | 0.527561 ± 0.016311 | 135 | 128 |
| failure_penalized/zero_calibration_failures | 0.810460 ± 0.244810 | 0 | 7051 |
| failure_penalized/certified_or_service_all | 0.810460 ± 0.244810 | 0 | 7051 |
| exact_public_history | 0.504427 ± 0.016263 | 0 | 0 |
| service_all | 1.000000 ± 0.000000 | 0 | 11418 |

Each fixed policy is evaluated on 2,880 worlds arranged into 1,440 paired histories across ten independently fitted seeds (23,040 entity decisions, which are not independent trials). Costs use 4 per residual failure and 1 per service. Compare these costs within this study only.

The forecast certificate policy removed 61 failures relative to its cost-tuned counterpart, but increased average cost from 0.527127 to 0.714540. Its zero-failure thresholds service every entity in four of ten seeds. The exact-history control remained cheaper and recorded no failures or unnecessary service, at 0.504427. Thus the new gate improves empirical reliability relative to its unconstrained counterpart without establishing learned-system superiority.

Calibration alone was insufficient: joint seed 4905 had zero calibration failures, failed the disjoint certification cohort, and would have incurred two evaluation failures. The preregistered rejection rule locked service-all before evaluation, producing zero evaluation failures at the recorded additional cost. Forecast and failure-penalized gates were accepted in all ten seeds; nine joint gates were accepted. The accepted gates are still finite-sample checks, not guarantees.

The fixed failure penalty did not enforce reliability when followed by unconstrained cost-based threshold selection: it recorded 135 failures, as did joint training, with different seed patterns. This tests one penalty magnitude and training/readout recipe, not the impossibility of reliability-aware training. Threshold optimization can also undo conservative training incentives.

The certified joint and failure-penalized interfaces cost 0.906727 and 0.810460 respectively, with zero observed evaluation failures. Their frequent service-all behavior is a costed backup, not evidence of improved memory or counterfactual understanding.

Seed failure vectors (cost-tuned policies):
* forecast: `[0, 0, 4, 9, 0, 24, 5, 0, 19, 0]`
* joint: `[23, 6, 20, 24, 2, 14, 8, 0, 24, 14]`
* failure_penalized: `[24, 6, 11, 24, 9, 24, 7, 6, 24, 0]`

The zero-event one-sided 95% bound is about 2.06% for each seed’s 144 independent generated pairs under the declared sampling assumptions. Do not substitute the 2,304 entities/seed as independent samples, pool repeated policy outcomes, or read zero observed failures as zero risk.

Training and inference wall times, model parameter counts, 200-scalar memory state, raw action probabilities and checkpoint identities are in the archived reports. Matched optimizer updates are not a claim of identical FLOPs; the penalty has additional operations. No integration, Mamba, biological retention, GEPA or ActionOPD improvement is established here.

Evidence: [raw checkpoints and replay records](https://github.com/FPC-effortless/TAC-OSM/tree/evidence/dev049-050-development-2026-10-17). Frozen training source: `84db3f1`. Audit checks checkpoint inference and simulator decisions, not independent repetition of optimization. Previous DEV-015–048 numerical evidence is unchanged.
