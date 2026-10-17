# What these studies can establish

The central distinction remains representation, prediction and decision. A
successful gradient test verifies a software path; an accurate action ranking
does not by itself verify a low-cost, reliable controller. Read the numerical
results with the per-seed failure counts and the registered guard.

## Writer attribution

DEV-044A tests three predefined gain/retention profiles with trained and frozen
writers. Its development-calibrated selections are base/base in every seed.
Writer learning reduces mean cost from 1.000 to 0.830 per entity, but produces
89, 4 and 45 failures against zero for frozen reservoirs. The failure guard
therefore prevents an improvement claim. The schema control costs about 0.531.
This does not replace longer, equally tuned GRU/delta/reservoir comparisons.

## Readout and decision supervision

DEV-045B exposes actual frozen pre-head readout features as well as original
probabilities. Some new interfaces improve cost in individual seeds. The
three-seed joint head on selected learned memory costs about 0.723 versus
0.823 for the original calibrated threshold, with 82 versus 48 failures.
That is a cost/reliability tradeoff, not a successful replacement. It also
argues against interpreting the old calibration failures as proof that all
information in the representation is unusable.

DEV-046B's joint and one-step simulator-taught arms reduce mean cost relative
to forecast-only training: approximately 0.605 and 0.574 versus 0.656. Their
failure totals are 52 and 55 versus 25. Optimizing expected cost can trade
unnecessary service against missed failure. A deployment criterion that
requires no increase in failures needs an explicit reliability constraint,
not just a positive coefficient on failure cost in an unconstrained loss.

## Generalization and system computation

DEV-047A tests shared local weights and identity permutations at larger entity
counts. The selective diagonal arm is a controlled recurrent equation, not a
Mamba implementation. Removing an early target event measures downstream
damage; no learned memory-item retention objective is established by that probe.

DEV-048A routes before OSM and records actual input entity positions. Inspection
fallback predicts the remaining entities too, so its total OSM positions return
to the full entity count. Exact-history fallback scans history and must be
charged. Unresolved review has incomplete coverage and no invented total cost.
Verification receipts are paid for; truthful observations can correct a model
even when they contradict its forecast.

The experiment makes one query per history. It does not establish amortized
runtime gains over many queries against an already updated persistent state.
The public event-queue controller remains an especially strong comparator:
known deterministic dynamics and reliable service give it an exact solution.
Adding a few queued events does not make that advantage disappear.

## What follows

Preserve negative outcomes and distinguish lower average economic cost from
passing the reliability requirement. A next readout study can preregister
failure-constrained calibration on development data and then evaluate a frozen
recipe on fresh seeds. A next architectural study needs uncertain dynamics,
interactions or partial observations that justify learned prediction, while
retaining appropriately capable explicit-state controls. Those are new studies;
they cannot be inferred by retuning on the development evaluations reported here.

The [reproduction guide](DEV-044-048-REPRODUCTION.md) records remaining gates.
The complete architecture, learned retention objective, real digital twin
validity and untouched-domain superiority remain unestablished.
