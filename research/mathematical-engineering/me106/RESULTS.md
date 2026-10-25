# ME106 — point-priority learning executed correctly; predictive qualification fails

[TAC draft PR408](https://github.com/FPC-effortless/TAC-OSM/pull/408),
[PNDS draft PR19](https://github.com/FPC-effortless/pnds-lean/pull/19).
Sourceadb71873a8ddf9939f9c185b47e24a68927b5122; all15 original weights,
optimizer/RNG/gradient/order/resource receipts frozen before REPORT at
6ae491e7a3608190ac70085b24c4b9705d1eb8ba. No original ME105 artifact changed.

## Completed integrated capability experiment

Same100,349 trainable/deployed parameters and33 active tensors; inherited full
ME105 predictive graph and source-gap floors. New learning graph removes the
new median correction's DIRECT gradient from NLL/interval scoring while preserving
allprimal forecast values. Point and log-location losses keep that head active.
An asymmetric finite-vector projection removes auxiliary gradients opposed to
the weighted physical point gradient. It applies established gradient surgery;
it is not new projection theory or a replacement of TAC by an SSM.

For correction c, ADtangent(c-detach(c))=dc, notzero. Thus mu=base+c and
mu_U=mu-(c-detach(c)) gives dmu_U=dbase. Python/reference/Lean contracts agree.
For p=grad(P),u=grad(U), if <p,u><0 and ||p||²>0 useu'=u-<p,u>/||p||²*p.
Then exact-real <p,p+u'> >= ||p||². This does NOT prove AdamW or finite-step
point descent, calibration or empirical generalization. FP64 reductions and
correction occur before rounding projected arrays to model dtype.

Exactlythree fixed complete recipes ×allfive ME103 strong parents102001..5,
not15 independent root initializations. New sampling indices106001..5; allold
weights train at3e-6, four new-head tensors at3e-4, freshAdamW wd.001, clip1,
1,536 updates each, batch16 contexts8/64/256. No favorable parent/seed selection,
retry, post-score extra fits or threshold changes. Allrecipes share initialization,
real plan/order/augmentation and profiled operator graph/dense training FLOPs per
seed. Unsupported pointwise/reduction/optimizer/hardware FLOPs remain partial.
Recorded training CPU1450.273s, two one-thread workers. Extra backward work is
training-only; forward parameters, state and operations remain unchanged.

## Original exposed-2015 results (five-seed means)

Primary PM2.5 errors/scores lower is better. Allfive seeds must pass allgates.
Old/new recipe sampling and gradient work differ; cross-study tables do not
establish a controlled architectural-superiority or attribution claim.

| Model | Gap MAE | Gap PM2.5 90% coverage | Six-channel gap NLL | Gap interval score | Ordinary MAE | Full seed gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|ME103 strong parents|37.524168|95.66%|5.151004|379.623199|33.403120|1/5 old gates|
|ME105 balanced|40.088106|94.69%|5.149118|260.676764|31.066647|2/5|
|ME106 balanced|38.667545|95.10%|5.154804|304.652338|33.430893|2/5|
|ME106 point_guard|38.043536|95.38%|5.150432|315.503455|33.552188|2/5|
|ME106 point_strong|37.282885|95.52%|5.144077|332.093921|33.622402|1/5|

**Full qualification FAIL: no recipe passes every gate in allfive seeds.** Five
individual models pass allgates (2/2/1 across recipes), but favorable seeds cannot
replace allfive. Primary coverage passes15/15. Stronger point weighting reduces
gap MAE at the cost of interval quality. Point_strong improves gap MAE only0.64%
vs its parent mean, ordinary MAE worsens0.66%, interval score improves12.52%.
Balanced gap MAE worsens3.05%, ordinary MAE is essentially unchanged(+0.08%),
interval score improves19.75%. ME105 balanced's ordinary/sharpness gains are not
retained. More weight tuning is not a demonstrated solution to the capability gap.

| Recipe | Coverage pass | Gap point pass | Ordinary point pass | NLL pass | Interval score pass | Full pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|balanced|5/5|2/5|4/5|5/5|2/5|2/5|
|point_guard|5/5|3/5|4/5|5/5|2/5|2/5|
|point_strong|5/5|4/5|4/5|5/5|1/5|1/5|

Unmatched practical point references (not new neural fits): LOCF gap56.076923,
ordinary34.515625; original FIT-ridge gap43.915070, ordinary32.767415. Allnew
recipe means improve over LOCF and gap ridge, but ridge has lower ordinary point
MAE than allthree. These benefits over simple references existed in the parent
sequence; they do not establish new architecture or decision intelligence.

## Allchannels remain visible

Mean nominal90% gap-cohort coverage; not a joint90% region. Coverage for several
other channels remains below90%. See allper-seed point/interval/width metrics.

| Channel | Parent | balanced | point_guard | point_strong |
| --- | ---: | ---: | ---: | ---: |
|PM2.5|95.66%|95.10%|95.38%|95.52%|
|PM10|89.51%|87.55%|87.69%|87.41%|
|SO2|82.55%|79.86%|80.00%|79.72%|
|NO2|74.06%|71.45%|71.74%|72.03%|
|CO|80.14%|79.00%|79.00%|78.86%|
|O3|92.43%|88.29%|89.14%|89.71%|

## Source, mathematical evidence and replay

Authentic UCI501 CC-BY-4.0 SHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
FIT2013–2014; exposed2015:143 gap queries/21 calendar-week clusters,
4,288 ordinary queries/53 overlapping clusters, six-hour targets,256-hour histories.
Only actual recorded FIT values supervise artificial masking/reconstruction.
Protected2016–2017 numerical records remain unparsed/unscored. Seeds/repeated
queries do not add independent observations. Posthoc paired week bootstrap
conditions on fitted models and exposed cohorts, without multiplicity adjustment.

All1,027 local tests and13 pinned PNDS source tests pass. Thirteen NEW limited
Lean training-role/projection theorems pass with standard axioms, reusing the
explicit full ME105 neural graph. PNDSbuild/source-math at72388e9d passed.
Whole-source neural/gradient/IEEE and Gaussian-law refinement remain OPEN;
compiled definitions/custom-AD algebra do not prove trained-model capability.

All15 fitted models passFP32/64 forward/state/input+weight gradient, source floor,
median/chunk/distribution checks AND independent full loss/projected-training-
gradient audits (30 dtype runs). Largest projected-gradient absolute difference
1.1920929e-6; raw median0.0008544922 original units. Fixed32rtol2e-5/atol2e-6,
64rtol1e-9/atol1e-10; not a universal2e-6 absolute raw-output guarantee.
Original checkpoints/forecasts/scores/failed gates preserved. Independent clean
and GitHub replay are recorded separately when completed, not inferred from tests.

No more ME106 fits. Allsix families remain; this trains Temporal only. No merge,
promotion, customer utility, autonomous policy, hardware advantage or general-
intelligence claim. Next substantive candidate should improve conditional
representations/distribution expressivity (for example source-conditioned forecast
regimes) under a fresh protocol, rather than another point-weight sweep. This is
a proposal, not an implemented success or proved explanation of the failure.

Clean detached checkoute812f655 replayed every original score/gate, all30 fitted
source+training-gradient audits and publisher-label summaries without retraining.
All67 original bundle payload SHA256 checks pass. GitHub original-source replay
40220362323 is in progress at this metadata write; earlier source-only checks
do not substitute for this replay.

Strongest-point candidate gap MAE change -0.24128, conditional posthoc95%week
bootstrap interval[-3.19990,+2.34032], includeszero. This is not independent
evidence or an acceptance gate; allother intervals remain in PANEL_SUMMARY.
