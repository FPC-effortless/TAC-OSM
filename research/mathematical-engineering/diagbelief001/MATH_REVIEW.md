# ME110 uncertainty mathematics audit (read-only, no fit)

Read AGENTS.md, fpc-universal SKILL.md, current ME110 MATHEMATICS/RESULTS, and actual source. Worktree research/me110-typed-role-learning-2026-10-26 initially clean. No frozen source, checkpoint, train/eval data, or model modified. Numerical probes instantiate an untrained model and differentiate once; no optimizer step, training, TEST access, or result rescoring.

## Main conclusion

No demonstrated probability-normalization or censoring implementation bug was found. There are concrete representational/optimization restrictions that plausibly explain weak interval scores; they are not established causal explanations of the scored failure. Existing results explicitly acknowledge complete gate failure and all-channel coverage gaps. Priority should be no-fit diagnostic accounting of floor activation, scale movement and discarded calibration gradients, rather than a new fit.

## 1. Hard missing-scale floor creates exact width-gradient dead zones (verified mechanism; causal relevance OPEN)

`src/tac_osm/variants/residual_role_sensor_world.py:83-100`: missing sides are at least max(sqrt(FIT rate*age/24), .8*sqrt(total_variance)). For multiplier exp(log(4)*tanh(raw)) below .8, width-head derivative is exactly zero, even if that side is statistically too wide. Threshold raw < atanh(log(.8)/log(4)) ~= -.1624 when the .8 floor dominates. If the source floor dominates, the dead region can be larger. Both sides can simultaneously have dead width gradients. Calibration may still reduce underlying modeled variance unless the source floor dominates; do not call the entire distribution frozen.

Executed scalar PyTorch falsifier: base=1, raw=-1 gives final scale .8000000119, dscale/draw=0, dscale/dbase=.8000000119. raw=.2 has nonzero derivative 1.7515804. This follows actual operators at lines 88-100. Cheap actual-checkpoint diagnostic: per seed/channel/availability count split_side<=protected_floor, and sum desired interval/NLL scale derivatives on those entries; separately count floor_variance>=modeled_variance. If dead-side share is negligible, this mechanism is falsified as dominant explanation. Do not alter historical thresholds/floors after evaluation.

## 2. One-way coordinate separation does not protect uncertainty against point learning (verified restriction, explicitly acknowledged in math record)

`typed_role_learning.py:19-23,35-50` permits uncertainty losses only on U. Encoder, source admission, memory, state_response, current_residual.input, median parameters remain P. Those P variables influence scales and reconstruction errors as well as medians (`residual_role_sensor_world.py:53-72,77-90`). Thus point updates can move calibration features and affine variance propagation while U cannot train those features to encode uncertainty-specific information. The local point-dot guard supplies no calibration guard (`typed_role_learning.py:60-63`; `inward_role_guard.py:18,41-59`).

Executed actual-source untrained-model probe: differentiating left_scale.sum()+right_scale.sum() produced 95,205 nonzero coordinates outside U on a sample fixture. This is a sensitivity count, not a trained loss gradient or proof of harmful interference. The predeclared math record already states cumulative point learning can alter uncertainty features: do not report this as an undisclosed implementation bug.

Cheap falsifier: on frozen existing monitor batches compute gU_full, discarded P*gU_full, and J_scale * proposed point displacement (finite difference plus analytic JVP); compare uncertainty objective before/after hypothetical proposal without committing weights. If discarded gradients and harmful scale motion are tiny, this explanation loses support. Handle both actual routed proposal and finite-step loss because a nonpositive point dot does not prove loss descent.

## 3. Diagonal scale/age variance is not identifiable as physical uncertainty decomposition (mathematical limitation)

`belief_sensor_world.py:27-31` gives age/evidence variance; `residual_role_sensor_world.py:72,79-90` adds learned current residual variance and future variance, then width multipliers. Multiple combinations give the same final side scales. Width can compensate for overestimated latent/current/future variance; interpretation of any one component as measured physical uncertainty is unidentifiable from predictive NLL alone. `state_response` influences both location and variance but receives only point gradients. Reconstruction loss (`reconstruction_sensor_world.py:119-124`) constrains current variance only on selected hidden-current targets; it does not identify future-vs-current split globally or supply covariance terms.

Cheap falsifier/diagnostic: record component ratios and local Jacobian singular values for final scales with respect to variance-head biases, state_noise and width biases, stratified by missingness/age. Near-null compensation directions support nonidentifiability. A general covariance omission is a modeling hypothesis, not a bug: source explicitly declares a conditional diagonal affine emission, not a full-network uncertainty propagation theorem.

## 4. `total_variance` is a pre-width construction value, not final split-law variance (verified semantic distinction)

For uncensored standardized split law Z=mu+sigma_side*N with each half mass .5:
E[Z]-mu=(right-left)/sqrt(2*pi),
Var(Z)=(left²+right²)/2-(right-left)²/(2*pi).
Thus trace `total_variance` (`residual_role_sensor_world.py:107-112`) only defines base_scale². Allowed left=.8*base,right=4*base gives final variance 6.6902533827*base²; executed scalar formula. Raw censoring/exponentiation changes moments again. Neither location nor `current_mean` should be described as final predictive arithmetic mean. Source correctly calls split location median (`median_split_sensor_world.py:1-7`). Not a forecast bug unless downstream consumers treat trace total_variance as final variance.

Cheap falsifier: inspect actual consumers and numerically integrate CDF moments vs this formula before censoring. Report side scales or derived true moments rather than rebranding base variance as final uncertainty.

## 5. Objective weighting targets unequal populations and fixed-location scale optima (verified design; superiority remains hypothesis)

`me110_objective.py:6-13,20-24`: point objective equally weights present channel/availability cells per minibatch; uncertainty NLL/interval/reconstruction inherits label-count averaging (`me106_gradient_roles.py:14-24`). Rare missing groups can dominate point representation learning but contribute proportionally little scale supervision. Group presence makes point group weights random per minibatch rather than one fixed dataset-weighted objective. This is deliberate availability balancing, not an averaging coding bug. Raw MAE+.1 standardized-log MSE also generally has a compromise optimum rather than the exact conditional raw median. The conditional width optimum given that compromise location need not achieve symmetric 90% tail coverage.

Cheap falsifier: from original batches/monitor arrays tabulate actual effective point and auxiliary weights per channel/availability; compute groupwise derivative contributions and censored PIT/tail misses. If weights are near equal or scale errors do not concentrate in underweighted cells, reject weighting imbalance as dominant cause. Avoid tuning a replacement weighting on frozen REPORT.

## Probability-law sanity

`median_split_sensor_world.py:80-84,101-108,118-127` uses sigma_left for negative normal quantiles and sigma_right for nonnegative, with each half mass .5. Positive density includes side.log(), target_scale.log(), and log1p Jacobian. At zero NLL uses the censored atom via log_ndtr. This is a valid median-preserving split law despite a density jump at mu when sides differ; adding the usual continuous two-piece-normal normalization would change the law and median. No such correction is justified. Tail exponentiation may overflow for extreme allowed parameters (`raw_quantiles:83`); no finite-output guard exists, but no observed historical overflow was established here.

## Executed checks and limits

Executed source reads, git status, and one PYTHONPATH=src:. Python no-fit scalar/actual-source differentiation probe. All reported probe values above are actual outputs. Did not run full tests, score replay, checkpoint perturbations, PNDS verification or data diagnostics. No claim of real-model causal diagnosis, trained-model calibration proof, full IEEE equivalence, or model superiority.
