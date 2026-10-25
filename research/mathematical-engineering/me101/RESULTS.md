# ME101 — ordinary forecasting improves; missing-data uncertainty gate fails

Five fresh **97,259-parameter** TAC-Temporal models completed 3,072 guarded updates
on authentic UCI501 station measurements. This is an integrated capability study,
not a four-way ablation or architecture competition. No ME100 model was replaced.

The model adds source-age/evidence-mass current-state variance and exactly
marginalizes a conditional affine emission in standardized log concentration.
Its diagonal Gram contribution is added to strictly positive future noise. The
native scan and 608-byte FP32 state remain unchanged; 24 learned parameters are
added. Extra integer/provenance/staleness/runtime memory remains separately charged.
This implements a declared belief model, not a verified Bayesian posterior.

## Real exposed-2015 results

Means across all five original models, full 256-hour input, six-hour forecasts.
Each seed scores the same cases. Raw concentration units; lower error is better.

| Metric | ME100 | ME101 |
|---|---:|---:|
| Ordinary PM2.5 MAE |35.689034|**32.770872**|
| Natural-gap PM2.5 MAE |38.986917|**37.747018**|
| Ordinary six-channel density NLL |5.010164|**4.963346**|
| Natural-gap six-channel density NLL |**5.769106**|6.234504|
| Ordinary PM2.5 proper interval score |228.490656|**188.780493**|
| Natural-gap PM2.5 proper interval score |**279.347305**|301.220560|
| Ordinary PM2.5 nominal90% coverage |90.81%|87.81%|
| Natural-gap PM2.5 nominal90% coverage |67.69%|68.67%|
| Ordinary PM2.5 mean interval width |196.49|143.45|
| Natural-gap PM2.5 mean interval width |104.94|89.28|

Ordinary PM2.5 MAE improves **8.18%** over ME100; mean errors improve for all six
pollutants. All five new models beat ordinary LOCF individually; the mean is
5.06% below LOCF34.515625 and approximately tied with FIT-only ridge32.767415.
This is useful point-forecast development, not proven superiority to ridge.

Gap PM2.5 improves only3.18% in the mean, with three of five mapped parent-seed
comparisons worse. Gap PM10/SO2 errors regress while other channel point changes
are mixed. Six-channel gap NLL worsens8.07%; PM2.5 gap interval score worsens7.83%.
Intervals become narrower, **not** uniformly more reliable. The proposed belief
formula did not fix the missing-data uncertainty failure.

## Registered gate ledger — FULL FAIL, 0/5

| Seed | Gap PM2.5 MAE | Gap coverage | History | Practical | Daily guard | Interval score | Density | Full |
|---|---:|---:|---|---|---|---|---|---|
|101001|34.774204|63.64%|PASS|PASS|PASS|PASS|PASS|FAIL|
|101002|36.261112|77.62%|PASS|PASS|PASS|PASS|PASS|FAIL|
|101003|33.985218|70.63%|PASS|PASS|PASS|FAIL|FAIL|FAIL|
|101004|40.590847|68.53%|PASS|PASS|PASS|FAIL|FAIL|FAIL|
|101005|43.123707|62.94%|PASS|FAIL|PASS|FAIL|FAIL|FAIL|

Every coverage gate fails the unchanged85% minimum. No favorable seed selection,
posthoc variance multiplier, rounded pass or repeat fit is allowed. A sharper
interval alone is not success.

The within-model long-minus-short gap MAE95% global-week bootstrap interval is
[-17.187783,-0.438010]. This is conditional on these five fitted models and an
already exposed development population, without multiplicity adjustment or
retraining uncertainty. It is not an independent architecture or transfer result.
There are143gap queries in21global calendar-week clusters and4,288ordinary noon
queries in53clusters. Repeated queries and five seeds do not multiply independent
source cases. All six-channel original forecasts, masks, scores and supplementary
width percentiles are preserved directly in Git.

## Training, mathematical engineering and reproducibility

Code and protocol freeze:493032ff7210604d6123a923d0d1a786607552ea.
All original models/optimizers/RNGs/sample schedules/live receipts freeze:
3752a98e302212c3245f394e6c684000bb35152b, **before** ME101REPORT scoring.
Original scores:5430d4b16c076c788e08395406c7ba5315755483.
FIT2013–2014;2015REPORT is retrospective exposed DEV;2016–2017 numerical outcomes
remain closed. Source/preprocessing is unchanged from ME100. Each model sees
49,152 forecast-origin presentations and5,373,952 historical-event presentations,
with replacement and target missingness. These are exposures, not unique records.

ME101 has twice ME100's update budget and a new loss. Actual optimizer-loop CPU
is583.106seconds total across five fits, excluding source loading/startup; at most
two one-thread workers. No controlled architecture, data-efficiency, FLOP-parity
or wall-clock speedup claim. Parent seeds are only mapped by index, not identical
initializations or schedules. All27 parameter tensors receive recorded finite,
nonzero gradients at update2.

**970 local tests pass.** Full serial/parallel outputs, states and all parameter
gradients match in FP32/FP64, including all five original fitted networks on real
FIT queries and chunk/full agreement. Largest fitted state/output discrepancy:
7.629395e-6 FP32 and7.105427e-15 FP64, within declared numerical tests. This is
not bit-exactness or a complete IEEE proof.

PNDS scalar positivity, age-monotonicity under fixed mass/coefficient premises,
observed-zero and read-only proofs compile. Standard Lean axioms are audited;
softplus formal refinement and full-network IEEE proof remain OPEN. A sum of PSD
matrices alone was never treated as a PD proof. The future noise floor supplies
the strict positive diagonal here. Future-only supervision does not identify the
state/future uncertainty decomposition separately.

Independent clean-checkout original score, fitted-graph and sharpness replays
PASS. GitHub original replay40178388335 PASS; PNDS source/math40177277628 and
build40177277596 PASS. Artifact receipts/checksums are repository-owned.

## Capability interpretation and next work

Evidence tier: **MIXED_TRADEOFF**. Ordinary point forecasting improves; calibrated
missing-data belief and the complete capability gate remain unqualified. No
intervention, autonomy, company deployment, independent site or general intelligence
claim follows. Six active model families remain unchanged.

The advisory idea queue and fast experiment loop allow low-cost parallel review
and independently registered studies without demanding attribution first. This
completed protocol ends here. The next integrated candidate should learn a more
useful conditional current-state representation and future distribution together,
with bounded FIT-only selection on point, proper density and interval objectives
before its frozen five-seed run. Authentic observed-label reconstruction and
coherent regime distributions are possible hypotheses, not verified solutions.
Do not spend protected TEST on repeated selection or automatically add geometry,
CED or more retention to compensate for a failed probability model.
