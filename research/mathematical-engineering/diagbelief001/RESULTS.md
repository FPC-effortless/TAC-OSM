# DIAGBELIEF001 — mathematical constraints, learning paths and leakage boundaries

2026-10-26. No-fit audit of original ME110 final models, source49a2ed479c560513e18894b995ab18558d399076. All five seeds110001–110005; original fixed320FIT origins (64 natural-gap stratum plus256 general). No REPORT/protected numerical scoring, optimizer construction, parameter displacement, new candidate or changed historical gate. Scope and questions were committed before measurements. Original ME110 qualification remainsFAIL0/5.

## Main result

A strong measured constraint on interval sharpness at fixed locations is a nearly saturated, fixed minimum width on missing PM2.5, alongside deliberately blocked reconstruction-mean learning. The audit does **not** establish that generic negative-gradient interference is the dominant cause. No forbidden future-label leakage was found at the tested boundaries; complete integrity remainsINCONCLUSIVE.

### 1. Hard floor restricts sharpness in actual trained models

Missing PM2.5 entries among the320monitor origins:66 (includes two in the general stratum). All rows below use those same66cases; seeds are not independent outcome samples.

| Seed | Fixed source variance dominates modeled variance | Left width at hard floor | Right width at hard floor | Minimum/actual mean width at fixed locations |
|---|---:|---:|---:|---:|
|110001|87.8788%|46.9697%|100%|99.6807%|
|110002|87.8788%|56.0606%|98.4848%|99.7514%|
|110003|89.3939%|62.1212%|100%|99.8673%|
|110004|92.4242%|48.4848%|96.9697%|99.7246%|
|110005|90.9091%|71.2121%|100%|99.8601%|

Mean PM2.5 missing-case width1015.777100 versus minimum permitted1013.471762 original concentration units on **this FIT monitor**, not the published2015REPORT population. These widths use different cohorts and are not directly comparable with ME110REPORT width396.649622. Minimum is computed by fixing each recorded location and setting both side scales to its actual protected floor. It is not a minimum over arbitrary medians/parameter changes and is not a counterfactual trained forecast.

For missing channels, source `residual_role_sensor_world.py` implements

    v_floor = FIT_24h_observed_pair_standardized_log_difference_MSE * age_hours/24
    b² = max(v_modeled, v_floor)
    sigma_side = max(b*exp(log(4)*tanh(w_side)), sqrt(v_floor), .8*b*missing)

When the last max selects the floor strictly, the side-width derivative with respect to w_side is zero. Strict floor-selection inequalities are measured in the real fitted graph; the zero width-logit derivative follows from the max operator and is checked separately in a scalar mathematical test. It does not imply every variance parameter or location gradient is zero. More width-head training alone has little room to shrink these intervals at fixed medians.

**Mathematical weakness:** a pooled24hour observed-pair squared-difference estimate is not a lower bound on conditional uncertainty. Linear age extrapolation assumes a diffusion-like accumulation law. A unit-stationary-variance AR(1) with rho=.9 gives proposed240hour floor18.4047, whereas the exact conditional variance given its last state is approximately1. This counterexample falsifies the universal statistical-floor interpretation; it does not establish the optimal replacement on Beijing pollution. A fixed positive algebraic floor prevents numerical collapse, but is not a coverage theorem. Never remove or retune the historical floor after seeing REPORT results.

### 2. Reconstruction can learn variance but not directly correct its mean

The actual reconstruction loss has a nonzero current-mean derivative in every sampled trained model (norm0.272151–0.321097). Its gradients on P coordinates have norm3.413981–4.066384, versus0.330295–0.402968 on U (exact per-seed values inRESULTS.json). ME110 deliberately drops reconstruction's P gradients; the current-mean output rows and their input representation are in P.

Therefore the statement “reconstruction trains the current-state mean” is false for this typed update rule: reconstruction may tune current variance, but current means are shaped by the point objective through the future decoder. This is an intentional routing restriction, not an autograd disconnection or evidence that reconstruction has no effect. The mean still changes during point learning.

Point learning also modifies encoder/memory/context features consumed by uncertainty. Coordinate separation is one-way; it is not functional independence of the two tasks. A nonpositive local point-gradient dot certificate does not ensure finite-step calibration or interval-score improvement.

### 3. Measure conflicts where updates actually survive routing

One fixed16origin augmented FIT batch per seed, all five final checkpoints. Existing Research Observatory measures full-source module gradients; a separate audit applies the **actual P/U masks** before computing cosines. This distinction matters.

| Pair | Actual retained U-coordinate cosine across five seeds |
|---|---:|
| Future NLL vs interval score | +0.598815 to +0.762008 |
| Future NLL vs reconstruction | −0.022133 to +0.016546 |
| Interval score vs reconstruction | +0.005480 to +0.099469 |

By contrast, full common-gradient NLL/reconstruction cosines are −0.300437 to−0.077515; many conflicting coordinates are discarded by the optimizer. Thus generic PCGrad is not supported as the first fix by this sample. These are local gradients on one batch/seed, not a trajectory-wide proof of no interference, nor sufficient to attribute prior gate failures. Objective magnitudes, group imbalance, finite steps, feature drift and changing regimes remain relevant.

Point losses balance channel/availability groups; auxiliary losses average available labels. Rare missing groups thus receive different relative weighting across objectives. This is a declared design mismatch to evaluate prospectively, not a proven averaging error.

### 4. Probability law and uncertainty terminology

The censored median-preserving split-normal-in-log1p law has a valid CDF/density, including its atom at zero and the log1p change-of-variables term. No normalization bug was established.

For the uncensored standardized-log law, with left/right scales l,r:

    E[Z] = median + (r-l)/sqrt(2*pi)
    Var[Z] = (l²+r²)/2 - (r-l)²/(2*pi)

Consequently trace `total_variance` is a **pre-width construction value**, not the final predictive variance; censoring and exponentiation change moments again. A mathematically allowed (.8,4) pair has variance6.690253 times the base value. Actual FIT aggregate ratios are retained per channel/seed inRESULTS.json; these are ratios of summed variances over all320origins per channel, not just the66missingPM entries. This is a semantic/identifiability limitation, not a detected consumer bug.

A sum of positive-semidefinite Gram terms remains PSD, but need not be positive-definite. Strict PD needs a positive diagonal term or a full-rank condition. Likewise bounded recurrent state does not ensure finite expm1 quantiles for arbitrary weights. Full neural/IEEE verification remainsOPEN.

### 5. Leakage and data-integrity findings

**PASS at named boundaries:**

- Original archive SHA matches declared publisher-source bytes. Actual FIT loader instrumentation checked4,400,584 numeric-field accesses, all before2015;12first-cutoff rows were read only far enough to stop. FIT length16104hours/site. No protected numeric values were accessed.
- Independently recomputed actual FIT-only mean/std match frozen loader values bit-for-bit under its float32-log/float64-aggregation contract. Independent all-float64 mean differs at most5.457684e-6; that is a precision distinction, not leakage.
- Actual CSV/transform fixture: poisoning all later REPORT values preserves FIT inputs/statistics and earlier REPORT inputs; direct+6target poisoning changes labels but not the first six augmented model inputs. An excluded2016row contains nonnumeric sentinels and is never converted. Fixtures are engineering evidence, not authentic capability data.
- Model forwards consume only values/masks/ages/clocks/IDs/times. Reconstruction/future labels are outside forward arguments in inspected code.

**Not cleared:** inherited transform/cache/source-join provenance end to end, all-source near duplicates, nuisance controls, original ancestor fit receipts, every REPORT truth/mask join, or full IEEE information flow. The older poison probe fixed normalizer caches and therefore did not certify raw-source-to-transform isolation. Phase strings alone do not authenticate arrays: a helper will accept a longer mislabeled raw array. The actual caller uses the cutoff-enforcing loader; no actual contamination was established.

FIT monitoring excludes current update target origins, but ancestral models and FIT statistics saw the fullFIT partition, and histories overlap. The2015REPORT cohort has been repeatedly exposed. Neither is an untouched holdout. Eligibility also conditions on a recorded+6hour PM2.5 target; this evaluates an observed-target population, not all future outages. Reading compressed archive bytes for hashing is distinct from parsing protected numeric values.

## Research-informed next engineering step

SeePRIMARY_LITERATURE.json andLITERATURE_AND_NEXT_DESIGN.md. The priority is a **new, prospectively registered coherent belief/forecast factorization**: preserve the improved point path and native affine scan; give hidden-current mean reconstruction a controlled learning path; learn conditional tail/scale behavior using matched availability groups; separate a small numerical positivity constraint from empirically justified staleness uncertainty; retain all-channel proper-score/coverage gates. No post-hoc floor relaxation, threshold changes, additional ME110 fits or protected-cohort tuning.

The audit does not add another architecture or claim improved intelligence. It supplies a reproducible reason to prioritize those changes over another generic gradient-surgery or larger-memory experiment.
