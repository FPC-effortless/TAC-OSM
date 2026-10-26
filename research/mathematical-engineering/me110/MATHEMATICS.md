# ME110: coordinate-typed native predictive-state learning

One integrated104663parameter ME109 forward function is retained exactly:
observed/masked13-channel inputs+sourceage+clock+station -> encoder48 -> causal
source/evidence traces -> input-conditioned affine8×8native associative scan ->
fast/slow features and retrieval -> retained affine heads + availability-conditioned
108→12→12/12/6 residual FFNs -> censored split-lognormal six-channel forecast.
The152FP32primary-state scalars and chronological scan semantics do not change.
No Transformer/SSM, additional learned parameters, authority or action outcome
is introduced. Existing Python/source FP32/FP64 graph reference remains inherited;
new optimization/reference correspondence is checked separately.

## Typed coordinate masks and gradients

Let U_i∈{0,1} identify uncertainty coordinates and P_i=1−U_i. U consists of all
state_noise,width_head,width_residual coordinates and rows6: in readout.2,
retrieved_delta.2,current_head,current_residual.output. Other coordinates—including
shared encoder/memory/retrieval, current mean and median corrections—belong toP.
Actual live counts: |P|100253, |U|4410, total104663.

At each actual batch, separate observed and missing current channels by their
post-augmentation observed mask (not sampler position). For each present
(channel,availability) group average future raw median error/FITrawstd and
standardized-log location MSE. Average the present groups equally:
L_P = group_mean_MAE +0.1group_mean_MSE.
L_U = censoredfutureNLL +0.25rawproper90intervalscore/FITrawstd
      +0.3hidden-currentGaussianNLL.

Compute gP=∇L_P and gU=∇L_U. The actual source must have U*gP=0 on every batch;
any violation blocks training. Route rP=P*gP, rU=U*gU. Clip the two vectors
independently at norm1 (denominator norm+1e-6, as in the source), then
r=P*clip(rP)+U*clip(rU). Auxiliary gradient magnitudes do not change point-role
clipping. AdamW moments are coordinatewise and start fresh; the retained/new
rates are3e-5/3e-4 with original betas/eps/decay.

Each full rate/decay/preconditioned candidate is protected with the already
source-verified inward guard, using gP and13exhaustive mathematical blocks.
Conditional realized-dot checks happen before any weight/moment commit. gP is
zero onU, so the point projection does not change those uncertainty coordinates.
The certificate is local/first-order, not finite-loss descent or generalization.

PNDS TypedRoleTraining proves binary-mask algebra and conditional congruence for
a decoder that factors through point coordinates. Real original-source probes
check uncertainty-only parameter perturbations leave medians/native state
bit-identical. These are not complete proofs of all neural/AD/IEEE operations.
Cumulative point learning can still alter features used by the uncertainty head;
calibration needs genuine held-out scoring. Neither task conflict nor all
statistical failure is universally prevented by coordinate algebra.

## Source/monitor and numerical scope

Original licensed UCI501FIT2013–2014 only. Monitor64natural-gap+256disjoint-general
origins excluded as new update targets; ancestors/stats already sawFIT and history
windows overlap, so this is internal monitoring rather than independent evidence.
Sampler2gap+14general uses shuffled cycles;57344general update origins distinct.
Current artificial-gap reconstruction masks follow the original causal producer.
Targets only supervise losses; six input fields remain the prediction boundary.
Scoped future-target poisoning holds normalization caches fixed; uninstrumented
source/normalizer/nuisance paths remain INCONCLUSIVE. Protected2016–2017closed.

Original full checkpoints/optimizer/RNG and monitor arrays at0/1024/2048/4096;
only4096model is REPORT-scored. No selection, repeatedfit or gate relaxation.
InheritedME1033072updates plus4096newupdates are charged separately. Fixed-stage
curves can reveal remaining learning but are not equal-budget architecture wins.
