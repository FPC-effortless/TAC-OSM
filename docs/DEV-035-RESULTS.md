# DEV-035: frozen MTSK dynamics results

The frozen diagnostic found strong one-step amplification in the independent-retention local MTSK snapshots. Decay below one does not make the complete error-correcting recurrence contractive. This is a mechanistic concern, not evidence that a repaired architecture improves prediction. These measurements concern the DEV-034 MTSK kernel, not the original native TAC writer; they do not replace DEV-021's suffix-transport evidence.

Protocol and implementation were committed at `5a6ac5e` before measurement. The original model and data generator came from frozen DEV-034 source `ec0ff80b058308dad8de57eab7d223ec9a81dcba`. No optimizer steps, validation selection or scientific test access occurred. Six snapshots each received three predetermined training batches of two H512 histories plus the original inspection receipt: 18 batch traces, 18,468 history transitions. These overlapping development histories are not independent statistical samples.

| Snapshot | Maximum conditional row gain | Rows with conditional gain > 1.000001 | Maximum sampled full-state tangent gain |
| --- | ---: | ---: | ---: |
| Initial local independent retention | 0.783 | 0 / 73,872 | 0.783 |
| Local independent retention, pre-update140 | 28.336 | 11,922 / 73,872 | 266.537 |
| Local independent retention, FP64-clipping continuation216 | 33.444 | 14,524 / 73,872 | 571.722 |
| Local tied retention216 | 2.641 | 522 / 73,872 | 12.251 |
| Global independent retention216 | 1.284 | 11 / 590,976 | 3.947 |
| Global tied retention216 | 0.851 | 0 / 590,976 | 3.346 |

The conditional operator fixes input and responsibility weights: `A = alpha*I - beta*r*k*k^T`. Its norm is exactly `max(abs(alpha), abs(alpha-beta*r*||k||^2))`. Key-dependent corrective feedback can overshoot even when decay is below one. This conditional operator is not the full derivative because responsibility weights depend on state.

The full augmented-bank Jacobian includes 72 memory coordinates and 24 responsibility coordinates. Its reported tangent norm restricts input perturbations to zero-sum responsibility changes. SVDs use FP64 on derivatives of the original batched FP32 transition. The maxima in the two gain columns need not occur at the same event. Euclidean gains depend on state-coordinate scaling.

The global tied control matters: its conditional rows never amplified, yet sampled full-state tangent gain reached 3.346. Therefore bounding only the fixed-responsibility feedback would not establish stability of the whole recurrence. Memory-to-responsibility and responsibility-to-memory feedback must also be checked.

All 18 traces remained finite and matched the original complete encoder-plus-receipt state bitwise. This does not contradict DEV-034's clipping failure: finite forward states can coexist with very large gradients. The separate continuation still had poor prediction and physical decisions; it remains a posthoc diagnostic, not a replacement for the failed pilot.

The archive inspection passed: hashes, raw conditional-gain counts, prescribed selected-point census, and independent NumPy SVD recomputation of 176 saved Jacobians. Five algebra/Jacobian contract tests also passed. The inspection does not independently rerun the frozen trajectories, replicate a seed, or audit new physical decisions; none were made here. Raw evidence is recorded in `DEV-035-EVIDENCE-MANIFEST.json`; it is a local workspace archive, not a remotely hosted artifact.

Sampling deliberately includes the largest conditional-gain event per history plus four fixed positions. These are selected one-step diagnostics, not an unbiased Jacobian distribution. They do not establish long-product growth, the cause of parameter-gradient failure, irrecoverable information loss, initialization reliability or architectural superiority. Original DEV-034 outcomes and the full-system capability gates remain unchanged.

## Next controlled intervention

Test a bounded corrective-feedback writer against the unchanged local independent-retention writer, with the same starting parameters, public entity masks, state budget, training streams and optimizer recipe. Normalize the key used in both prediction and correction and parameterize the effective step relative to decay so that the fixed-responsibility eigenvalues are bounded by construction. Register the exact formula and initialization mapping before any new training; normalization alone is not sufficient when the write gain is unbounded.

First verify the algebra and measure the full augmented-state Jacobian, including responsibility feedback. Then run a small paired stability/learning pilot and strict physical replay. A stable but uninformative memory is a failed intervention. Preserve primary all-role prediction, cancellation diagnostics, failures, unnecessary services and total cost. The existing masked-local TAC and schema baselines remain intact. A full tuning/fresh-seed confirmation is still required before any superiority claim.

This candidate addresses a measured recurrence concern. It does not replace the wider optimization study or the separate TAC/PLM–CDL–CASM–OSM integration track, and it is not an implemented or successful repair yet.

Integrity CI [38992884598](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38992884598) passed all 1,168 tests at `f368b85`. It tests code contracts and regressions, not the locally archived frozen-checkpoint experiment. Subsequent changes are documentation only.
