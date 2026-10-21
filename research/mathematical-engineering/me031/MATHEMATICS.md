# ME031: What cross-fitted residual prediction can establish

Write C for the shared available causal context, H for learned native state,
Y for genuine subsequent feedback and q0(C) for a frozen common-feature predictor.
The residual predictors are

\[
q_C=\sigma(\operatorname{logit}q_0+f_\alpha(C)),\qquad
q_{CH}=\sigma(\operatorname{logit}q_0+g_\beta(C,H)).
\]

Both corrections begin at zero, so the initial functions are exactly q0. The
common and state corrections are both one-hidden-layer tanh networks, with
1205 and1197 active coefficients respectively. Equal coefficient counts do not
make the function classes identical: the common network is wider and the state
network has more input coordinates. This approximation/optimization difference
must remain part of the interpretation.

For binary log loss and logit correction f, the gradient with respect to a
parameter theta is

\[
\nabla_\theta L=E[(q_f-Y)\nabla_\theta f].
\]

At zero correction, this measures correlations of the REAL response residual
Y-q0 with the correction features. No synthetic outcome labels or teacher oracle
are used. With unlimited function classes, the optimal correction is
logit(E[Y|C,H])-logit(q0(C)), where conditional probabilities in(0,1);
probabilities0/1 are approached as limits. Finite tanh probes only approximate it.

Under binary squared loss, Bayes-optimal predictors m_C=E[Y|C] and
m_CH=E[Y|C,H] obey

\[
R_C^*-R_{CH}^*=E[(m_{CH}-m_C)^2]\geq0.
\]

Finite fitted Brier difference adds the two models' approximation/estimation/
optimization excess risks. Our log-loss-trained probes are evaluated by Brier
and other metrics; their empirical difference is not this ideal Bayes quantity,
not a conditional mutual-information estimator, and not a causal intervention.
A common-only residual control is necessary because q0 can simply be underfit.

## Three-way entity isolation

For each rotation, mutually disjoint user sets R,S,T train the representation,
fit base/probes, and evaluate respectively. Preprocessing duration/popularity uses
R; probe feature mean/scale uses S. Nothing is fit on T, and all windows from one
user share a role. Every T user's state comes from a representation trained
without that user's records. S feature generation is also out-of-user with
respect to representation training, avoiding representation in-sample vs T
out-of-sample feature mismatch. Each rotation has its own encoder, normalizers
and heads; latent coordinates from different encoders are never pooled as features.
Only final scalar predictions are pooled in original row order.

Each user's observed past feedback is a legitimate prediction input; exclusion
from parameter fitting does not require erasing their observed history. Causal
prefix extraction excludes next response and subsequent events while retaining
known next item/duration. Crossfit cannot remove previous architectural design
exposure to these FIT records; it is not an independent new dataset evaluation.

## Resource and uncertainty bounds

Every residual arm receives identical representation/base fitting and S sampling
exposures. The shared stages are charged fully to both arms, while the physical
study computes them once. This separates experimental resource accounting from
actual total cost. Actual head FLOPs/time differ, and the ablation deliberately
changes state access; no fully matched architecture winner is declared.

Bootstrap intervals resample users and preserve their windows and fixed-seed
paired predictions. Models are not refit in bootstrap, and rotated training sets
overlap. Intervals are conditional on these fitted models, not training-process
uncertainty or a generalization guarantee. All endpoint thresholds are frozen in
PROTOCOL.md; no statistically favorable subset is substituted after scoring.
