# ME101 — source-aware probabilistic belief memory

A complete integrated TAC-Temporal candidate; exactly six model families remain.
This is capability development on previously exposed 2015 observations, not an
architecture-superiority study. Original ME100 models and all failed gates remain
immutable. No attribution ablation campaign is a prerequisite for this run.

## Conditional distribution and complete graph

ME100's masked encoder → source sum/mass scan and evidence prefix → fast/slow
schedules → native associative scan → causal retrieval → forecast decoder is
inherited exactly, including station ownership, masks, clock/version checks and
full/chunk semantics. All inherited parameters train from scratch in ME101.

The source anchor a_c is the current observed standardized-log measurement or,
if missing,8*S_c/max(C_c,1e-6). Define a current-state latent belief with mean a_c
and diagonal variance

v_c=(1-observed_c)*(softplus(b0_c)+softplus(b1_c)*age_hours_c/24)
    *(1+softplus(b2_c)*(1-C_c)).

Age is recovered from the existing causal log-age encoding, capped256hours.
At fixed evidence mass and coefficients the variance is nonnegative, increases
with age and decreases with mass; observed channels have exactlyzero state
variance in this ideal observed-value model. Real instrument measurement noise
is not identified separately. Neither age nor evidence mass is calibrated
confidence by itself. Learned gate/mass changes can confound across-case
monotonicity; the theorem fixes mass and coefficients.

The conditional emission is affine in that latent current state:

L_future,c = mu_base,c + j_c*(X_current,c-a_c) + eps_c,
j_c=1+.25*tanh(response_c), eps_c~Normal(0,sigma_future,c^2).

Latent current channels and conditional future noises are independent under the
model. Integrating X out is exact for this declared conditional affine model:

mu_total=mu_base; variance_total=sigma_future^2+j^2*v.

J=diag(j) is the derivative of this conditional affine emission, NOT an asserted
Jacobian of the entire nonlinear neural network. It is a structured diagonal
Gram contribution J diag(v) J^T. Positive definiteness follows from the strictly
positive future-noise diagonal (sigma_future>=.05), not from PSD addition alone.
It does not automatically preserve proofs of the older ME098/099 trajectory law.
No new cross-channel covariance or multi-horizon path is claimed in this study.

Physical concentration Y=max(0,exp(L)-1), with FIT standardization inverted.
Median, ordered5/95quantiles, censored zero atom and positive-density Jacobian all
come from this SAME marginalized distribution. Training uses its proper NLL,
plus .1 standardized-log mean-square loss and .1 proper90interval score of the
censored standardized-log quantiles. No posthoc DEV interval multiplier.

## Computational reuse and dimensions

18learned variance coefficients and6response coefficients add24 to ME100's
97,235 parameters: **97,259 active trainable/deployed parameters**, no padding.
The new readout consumes the existing final source mass and causal age and does
not mutate the observed-world state. There are no additional persistent tensors:
152FP32values=608bytes, plus24integer bytes for owner/count/hour and104bytes for
external age bookkeeping. Python/model version metadata, weights and temporary
forecast arrays remain additional. The variance branch runs after the scan,
vectorized across independent channels, examples and entities; it does not add a
sequential recurrence or make an end-to-end GPU speed claim.

## Proof and source refinement

PNDS/BeliefSensor.lean proves observed-zero, nonnegative/age-monotone latent
variance, positive marginal variance and read-only trace preservation under
explicit coefficient positivity premises. The source-trace and matrix-memory
kernels are inherited. Softplus implementation, stochastic independence,
calibration, full-network IEEE behavior and superiority are not proved by these
kernel theorems. No `sorry`; standard Lean axioms are disclosed.

Independent Python evaluates the entire inherited graph chronologically with
rank-one matrix updates, then separately evaluates the new marginal law. Tests
compare FP32/FP64 outputs, state, new variances, gradients, masks and chunks.
Real fitted-source checks are required after training, not inferred from unit
examples. This candidate implements the NEXT-SHARED-BELIEF hypothesis without
claiming that ME100 proved cross-channel interference or a unique failure cause.

## Interpretation limit

The state/future variance decomposition is part of the declared generative model.
Future-only supervision does not separately identify a true posterior over a
missing current measurement and true process noise; their contributions can trade
while giving the same marginalized prediction. This is not demonstrated Bayesian
parameter uncertainty, optimal filtering or a guarantee of calibrated latent state.
The directly testable outcome is the forecast distribution on authentic futures.
