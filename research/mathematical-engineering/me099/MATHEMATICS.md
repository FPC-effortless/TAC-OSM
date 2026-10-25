# ME099 — coherent latent uncertainty regimes, bounded correction cycle

Read ../me098/MATHEMATICS.md for the unchanged native scan, conditioned trajectory
law, source and typed input/output graph. ME098's five originals and failed
coverage gate remain immutable. This is ONE new five-seed correction, with a
new registration before fitting; no further fits/threshold changes in this cycle.

## Why this new integrated model

ME098learned useful1/4/16frame means but nominal90%intervals cover only~75–83%,
and its jointNLL is worse than the FIT-moment reference. Correct the learned
predictive distribution and objective, rather than add another retention gate.
Do not claim which memory module caused this failure. Preserve all original data,
fit attempts, scores and the negative result. DEV is already exposed/adaptively
studied, not an independent cohort.

## Computational graph and probability law

Keep all103,322ME098weights active. Add only58learned values:
Linear(56,1) uncertainty-regime logit plus one global broadening parameter.
Regime features reuse ALREADY computed v0/delta_scale, terminal velocity/scale,
retention and log(amplitude/scale). No repeated encoder/NNfuture rollout, no
additional persistent observed-state, no seventh family.

Let the old coherent coordinate-factorized joint-horizon df4Student-t path
measure be P(S). Conditional covariance response S is derived from the same
stable physical trajectory and shared innovations. New distribution is

    P = (1-p) product_coordinate t4(mean,S) +
             p product_coordinate t4(mean,k^2 S),
    p=sigmoid(RegimeGate(features)), k=1+softplus(raw_k).

One regime variable is shared across ALLfuture coordinates and horizons. This
preserves projective horizon marginals and the unchanged forecast mean semigroup;
it permits dependent tail events across channels. It is a distributional regime,
not a physical movement probability, alternative action or epistemic certificate.
The two components share their mean; they express differing uncertainty.

Total covariance=2*((1-p)+p*k^2)*S (df4S is scale, not covariance). Cross-coordinate
linear residual covariance remains zero under equal-location component products,
although shared regimes couple higher-order tail dependence. No claim of a full
learned cross-channel covariance or counterfactual world distribution.

Global jointNLL uses log-sigmoid weights and logsumexp. Both components reuse one
Cholesky: scaling S by k^2 changes logdet by2Hlog(k) and Mahalanobis by1/k^2.
This reuses computation instead of factoring two matrices. Scoring reports joint
log density normalized percoordinate for comparability, not independent marginals
mistakenly labeled a global joint density.

## Derived marginal quantiles with correct gradient

For standarddf4, u=z/sqrt(4+z^2), F4(z)=1/2+3u/4-u^3/4,
f4(z)=12/(4+z^2)^(5/2). The positive95thpercentile q solves

    (1-p)F4(q)+pF4(q/k)=.95.

The bracket is [t4(.95),k*t4(.95)] for k>=1.48FP64bisections give a bounded
root error (plus floating rounding), with first-order implicit derivative

    dq/dp = (F4(q)-F4(q/k))/f_mix(q),
    dq/dk = p*f4(q/k)*q/k^2/f_mix(q).

Source attaches the implicit first-order gradient to the detached root. It does
NOTclaim correct second/higher derivatives. Independent Python uses80bisections,
a trigonometric CDF identity and explicit derivative attachment; density uses
slogdet/linear-solve instead of source Cholesky-whitening. Unitgradcheck verifies
p/k derivatives. Full NNsource values, state, quantiles and gradients are checked
inFP32/64. No empirical DEVerror quantiles or posthoc interval inflation are used.

Marginal bounds=mean +/- q*sqrt(diagonal(S)). Valid masks preserve evidence and
produce a degenerate zero-horizon/invalid forecast; densities are not scored on
those degenerate rows. Uncertainty is unbounded in support; no invented maxhours.
No finite-sample coverage guarantee is asserted from this mathematical law.

## Training, gates and scientific status

103,380trainable/deployed params,5original seeds99001–5, fixed corresponding
ME098parents98001–5 with SHA preregistered.512additional updates perseed,batch8,
64history,targets1/4/16. The old core lineage totals1,280updates, original
trajectory7448values1,024updates, new58values512updates. Repeated real windows
are exposures, not new independent events. RetainedcoreLR.00005, oldtrajectory
headLR.0002,newregimeLR.0006,AdamWwd.001,clip1, no sweep.

Loss: same .6/.2/.2FIT-RMS-normalized MSE +.05properglobal jointNLL/42 +
.1proper90%intervalscore/FIT-RMS. Intervalscore=(upper-lower)+20penalties for
missed endpoints. It targets quantiles properly; direct empirical coverage is
not substituted as an improper training objective. All labels/statistics come
from the same59FITepisodes.13previously exposedDEVepisodes,104windows; TEST not
scored. Mean/scale source contract exactlyME098. OriginalNNbytes/optimizer/RNG/
schedule/preproc committed BEFORE any score via hard Git-byte freeze. All5live
parameter receipts and clean independent original replay remain compulsory.

UNCHANGED numerical criteria: EVERY seed/horizon MSE<=.95persistence,
H1<=1.05its same-case frozen parent (now ME098, a stricter parent), 90%coordinate
coverage>=.85eachhorizon, intervalscore<=1.05sameFIT-moment-t reference. Do not
change baseline/gates/eligibility if this corrected model fails. Record ALLfive
original fits and all case/horizon losses, calibration and regressions. Single
architecture development, no matched-data/parameter/compute architecture winner.

PNDS formalizes conditional CDF convex-mixture range/monotonicity and positive
variance inflation, inheriting ME098fixed-mean semigroup/Gram proofs. These use
standardLean axioms, no sorry; they are NOT a full neural/t-distribution measure
or IEEEroot-error proof. Exact NNLean refinement, global numerical guarantees,
independent-site/long-memory/calibration qualification remain OPEN.

Established prior art: mixture density networks (Bishop1994,
https://publications.aston.ac.uk/id/eprint/373/), proper scoring rules
(Gneiting/Raftery2007, https://doi.org/10.1198/016214506000001437), and the
Mamba-2/GatedDeltaoperator references inME098. This is an engineering integration
of known probability tools into TAC's native shared graph, not novel probability
mathematics or proof of Mamba competitiveness.

Capability-to-value: sharper but reliable read-only physical future forecasts.
External operator/control utility, Woyengi deployment, paying customers, independent
pilots and ROI NOT_MEASURED. Otherfivefamilyheads unqualified. Remain draft/unmerged.
