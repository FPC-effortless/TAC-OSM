# ME105 — protected sensor predictions: primary coverage recovers, full qualification fails

Original source573f81cb9f2f421d141ddb722525ecce25972edc; all15 original model,
optimizer and RNG artifacts frozen at640cafca55be9b27375fb147e2ae206c7bfb1391
BEFORE REPORT. [TAC draft PR406](https://github.com/FPC-effortless/TAC-OSM/pull/406)
and [PNDS draft PR18](https://github.com/FPC-effortless/pnds-lean/pull/18).

## What was actually changed and trained

A complete100,349-parameter native Temporal model with a bounded median innovation,
source-floor-preserving asymmetric scales, and a missing-channel80% inherited-scale
lower bound. The old29 parameter tensors update at3e-6; four new head tensors at
3e-4. All33 tensors remain trainable and have finite nonzero first-batch FIT
 gradients. Full graph/loss flow is specified in MATHEMATICS.md and independently
executed in Python. PNDS adds explicit finite-weight neural graph definitions,
chronological state folds, both read paths and probability-transform interfaces;
26 limited theorems compile with only standard Lean axioms. Gaussian-law
instantiation, whole-source neural/IEEE/gradient refinement and coverage proofs
remain OPEN. Compiled definitions do not prove trained-model capability.

Exactly3 registered complete objectives ×5 parent lineages; no ablation campaign.
Each fit gets1,536 updates, batch16, contexts8/64/256, freshAdamW. Allfive original
ME103 strong parents are reused in every recipe; these are15 fine-tunes, not15
independently initialized root models. The recipe was selected using exposed DEV
before ME104, disclosed in both protocols; individual favorable seeds not selected.
New sampling seeds105001..5 map to parents102001..5. No ME104 fit or result was
retrained/overwritten. CPU sum955.686s, two one-thread workers maximum. Allrecipe
initializations/plans/masks and profiled operator graphs/dense FLOP estimates match;
unsupported scalar/optimizer/hardware FLOPs remain unquantified. Old/new models
have different size and exposure, so this is not a controlled architecture win.

## Original exposed-2015 results (five-seed means)

Primary PM2.5 metrics; lower errors/scores better. Coverage gate[90%,98%].

| Candidate | Gap MAE | Gap coverage | Six-channel gap NLL | Gap interval score | Ordinary MAE | Full seed gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ME103 strong parents | 37.524168 | 95.66% | 5.151004 | 379.623199 | 33.403120 |1/5 old gates|
|likelihood|36.270334|95.24%|5.137067|403.591461|34.962272|0/5|
|balanced|40.088106|94.69%|5.149118|260.676764|31.066647|2/5|
|point_guard|39.085660|95.52%|5.142799|273.164691|31.121632|2/5|

**Full qualification FAIL: no recipe passes every gate in allfive seeds.** Four
individual models pass every gate, but allfive are required. Primary PM2.5 coverage
passes15/15, versus0/15 in ME104. This does not establish calibration of allchannels.
Balanced improves ordinary PM2.5 MAE7.00% and gap interval score31.33% vs parents,
but gap MAE worsens6.83%. Point_guard has similar trade-offs. Likelihood improves
gap MAE3.34% but ordinary MAE worsens4.67%, and no seed passes the sharpness gate.
Coverage recovery is measured for the integrated candidate; three simultaneous
changes prevent attribution to any one mechanism. No threshold was relaxed.

| Recipe | Coverage passes | Gap point passes | Ordinary point passes | Gap NLL passes | Interval score passes | Full passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|likelihood|5/5|4/5|3/5|5/5|0/5|0/5|
|balanced|5/5|2/5|5/5|5/5|3/5|2/5|
|point_guard|5/5|2/5|5/5|5/5|3/5|2/5|

## Other channels are not all calibrated

Mean nominal90% gap-cohort coverage; not a joint90% region. All metrics for all
six channels/seeds and width tails are preserved in PANEL_SUMMARY/RESULTS.

| Channel | Parents | likelihood | balanced | point_guard |
| --- | ---: | ---: | ---: | ---: |
|PM2.5|95.66%|95.24%|94.69%|95.52%|
|PM10|89.51%|87.41%|86.01%|86.15%|
|SO2|82.55%|82.27%|81.28%|81.13%|
|NO2|74.06%|73.33%|73.91%|74.06%|
|CO|80.14%|79.86%|82.71%|82.14%|
|O3|92.43%|90.14%|87.14%|87.71%|

## Data, verification and interpretation

Authentic UCI501 CC-BY-4.0 archive SHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
FIT2013–2014; already exposed2015:143 gap queries/21 calendar-week clusters,
4,288 ordinary queries/53 overlapping clusters, six-hour targets,256-hour histories.
Seeds and repeated records do not multiply independent observations. Protected
2016–2017 numeric outcomes remain unparsed/unscored. Posthoc paired week bootstrap
conditions on fitted seeds and exposed cohorts, not multiplicity-adjusted or
independent qualification. EWA001/002 physical-cache negatives in the newer master
plan are preserved separately, not evidence for this sensor candidate.

All1,015 Python tests pass. Eight PNDS actual-source tests pass. Every fitted model
passesFP32/FP64 actual-source vs independent serial outputs, state, features,
source floors, bounded median, input/weight gradients, censored NLL/CDF/zero atom,
and32+32chunk checks:30 fitted checks. Fixed tolerances32rtol2e-5/atol2e-6,
64rtol1e-9/atol1e-10. Maximum raw median discrepancy0.000732422 original units;
not all raw absolute errors are below2e-6. See full per-field maxima.
Clean-checkout and GitHub replay status are recorded in restoration evidence,
not inferred from source tests or current working-tree replay alone.

No model promoted/merged/deployed. Six active families remain; only Temporal is
trained here. No autonomous decisions, full joint trajectories, safe control,
embedded hardware latency, customer value or general intelligence is established.

Next prospective architecture/training target: separate median-learning gradients
from uncertainty-shaping gradients and address channel-specific calibration. The
FIT trace confirms active paths and batch-specific conflicts, not a universal
causal explanation. No more ME105 fits; keep alloriginals and unchanged gates.

Clean detached checkoutd2a8bc96 replay passed: all15 original scores/gates,
30 fitted FP32/64 graph audits and allsource-label/summary checks, no retraining.
67 original bundle payload SHA256 values recorded and checked. Original-source
GitHub replay40216493230 was in progress at this metadata write; it is not
claimed passed from earlier source-only CI. PNDS fullbuild/source-math passed.
See RESTORATION_RECEIPT.json and CAPABILITY_PROGRESS.json for exact status.

State-byte audit:632 bytes/sample of actual state tensors (608 FP32 +24 integer),
plus72 model-buffer bytes. The registered736-byte value is an inherited budget
convention, not measured total runtime memory. Python metadata, parameter-version
tuples, allocator, activations and weights are separate. STATE_PAYLOAD.json
supersedes the loose initial MATHEMATICS.md includes-buffers wording.
