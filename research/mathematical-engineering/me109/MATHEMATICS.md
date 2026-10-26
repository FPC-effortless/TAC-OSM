# ME109 — identity-preserving, availability-conditioned residual roles

Status: prospective integrated capability candidate. No predictive result follows
from the architecture, the initialization identity, or its local optimizer guard.
Native TAC's recurrence and six active model families remain unchanged.

## Question and inherited negative evidence

ME108 trained five originals and passed its computed local update contract, but
failed the full predictive gate in every seed. Mean natural-gap PM2.5 MAE worsened
37.524168 ->41.132003 (+9.6147%). General MAE and gap interval scores improved.
Replacing the parent's learned current affine map with fresh narrow FFNs is a
possible contributor, not an identified cause. ME109 preserves that map exactly
at initialization and conditions learned corrections on actual availability.

## Typed input-to-output graph

Inputs: normalized observations V[B,T,13], legitimate Boolean observed mask
M[B,T,13], capped causal log-age A[B,T,13], calendar K[B,T,6], known owner IDs
I[B], strictly increasing recorded hours H[B,T]. Missing numeric slots are masked
before encoding; no targets, future masks, future calendar or total trace length
enter the predictor. Targets are recorded six-pollutant values six hours ahead.

The unchanged source-aware native TAC scan computes source_sum[B,13],
source_mass[B,13], evidence_sum[B,62], memory[B,8,8], count and ownership/time
metadata. Input-only affine transitions compose chronologically; readouts do NOT
feed back into the same scan. This is native TAC, not a Transformer or replacement
SSM. Fixed FP32 float payload152scalars=608bytes; metadata, parameters, activations,
training memory and reference execution costs are additional. Sums/counts and
capped age do not imply indefinite numerical boundedness or unlimited recall.

After the scan, re-encode the legitimately masked current observation, reuse the
write-key projection to form the normalized query, and read memory@query. The
features phi[B,102] concatenate source means13, source masses13, evidence means62,
current age6 and inclusive associative read8. Append current Boolean mask6 to make
c[B,108]. The input age already appears in phi; no duplicate age or hidden
mutable mask cache is introduced.

For roles r=current,width,median with output dimensions12,12,6:

    d_r(c) = W2_r tanh(W1_r c + b1_r) + b2_r
    u_r(phi,c) = L_r(phi) + d_r(c)

Each hidden dimension is12. All three L_r are retained independent affine maps
102->12/12/6. Current L is copied from the ME103 parent; inherited width/median
L start at zero. All W2,b2 initialize to EXACT zero; W1,b1 use seeded ordinary
initialization. Every parameter remains trainable, no padding. Parameter total:
100349 inherited + 3*(108*12+12)+(13*30)=104663;45parameter tensors.

Current mean = observed/retained-source anchor +2*tanh(u_current[:6]); current
variance=.0025+softplus(u_current[6:])^2+inherited latent-state variance.
The unchanged response, future variance and FIT-only age floor yield base mu,sigma.
Width factors exp(log4*tanh(u_width)) multiply the two scales. Median correction
.5*tanh(u_median) shifts mu. Both final scales retain the source floor and, for
missing channels, at least.8*base sigma. The coherent censored split-lognormal
marginals, raw quantiles, zero atom, NLL and raw-unit Jacobian are inherited.
No calibrated coverage theorem is asserted by a positive scale or a floor.

## Initialization and gradient contracts

With W2=b2=0, every d_r is zero, preserving all three parent raw readouts and
full deterministic decoder values for identical state. Actual FP32/64 source
tests require exact forward/state equality. Parameter/input gradient agreement
uses precision-specific existing tolerances, since graph sharing changes floating
accumulation order. Initially residual hidden parameter gradients are zero;
output affine gradients can be nonzero. This is staged residual activation,
not disconnected parameters or an all-gradients-nonzero claim. After training
starts, parent predictions are NOT frozen or guaranteed preserved.

Width heads cannot directly change the median; uncertainty-view NLL/interval
losses remove the direct median-correction tangent. Shared ancestors remain
coupled. The same13-block actual-displacement AdamW guard as ME108 protects the
current-minibatch six-channel average point gradient. It does not protect every
channel/regime, finite-step loss, held-out sample or future coverage.

## Integrated training and real-data gates

See frozen FAIR_COMPARISON.json for five seeds109001..109005, exact4096updates,
B16,8/64/256 contexts, all original losses and coefficients, optimizer rates,
FIT-only masking and unchanged per-seed gates. Parent3072updates count toward
retained-parameter exposure; all4096newupdates and repeated presentations are
reported separately. No matched architecture-superiority claim is eligible.

All originals/optimizer/RNG/receipts must be Git-frozen BEFORE REPORT.2015REPORT
is exposed development evidence; protected2016-2017 numeric rows stay closed.
Failure cannot be rescued with best-seed selection or threshold changes.

## Prior art, proofs and OPEN boundaries

Residual learning (He et al,CVPR2016), explicit missingness/time inputs
(Che et al,Scientific Reports2018), task-specific heads, AdamW and gradient
surgery are established methods reused here. Their combination does not prove
new intelligence. PNDS/ResidualRoleFFN.lean states limited exact-real residual
identity, role separation and weight-dependent bounds. The independent Python
serial reference reconstructs the actual native graph and new head equations.
Full source-to-Lean neural AD, nondifferentiable branches, IEEE transcendental
refinement, concrete CDF implementation, population calibration, useful memory,
transfer, control and product utility remain OPEN.
