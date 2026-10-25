# ME106 — source-equivalent forward, explicit training roles

## Complete predictive graph

`PointPrioritySensorWorld` inherits `ProtectedSensorWorld.forward` without a new
parameter, state coordinate or forward operation. The complete explicit equations
and typed shapes remain those of ME105 MATHEMATICS.md (source573f81cb), and
`PNDS.SensorGraph105` (PNDS52d65e76): masked49→48 encoder and station embedding;
source/evidence prefixes;88-coordinate fast/slow proposals; causal controlled
rank-one8x8 associative writes; distinct pre-event original decoder and inclusive
current-belief read; reconstruction/source floor; bounded median and asymmetric
protected scales; six conditional censored-lognormal half-distributions.
No decoder feeds back into the same scan. It is not an SSM replacement.

The independent Python formulation `me106_graph_reference.serial` delegates to
the unchanged independent ME105 evaluator, which traverses source equations
chronologically without calling any actual model forward/prefix kernel. The new
`loss_parts` and `priority_gradients` independently reconstruct alllosses and
flattened finite-vector projection without invoking production helper methods.
This is source numerical evidence, not Lean full-network refinement.

## Training values and tangent roles

Let mu=base_location+c(theta), with c=.5tanh(median_head(phi)). For uncertainty
NLL and interval scoring use
  mu_U=mu-(c-stop_gradient(c)).
Its primal equals mu exactly (the subtracted pair is an exact FP zero). Its AD
tangent is dmu-dc; because dmu=dbase+dc, the direct new median-head contribution
cancels. IMPORTANT: tangent(c-stop_gradient(c)) is dc, NOTzero. Stop_gradient's
custom tangent is zero; its primal is identity. A frozen-snapshot formulation
mu-(c-c_snapshot) at c_snapshot=c gives the same value, with snapshot held fixed
when differentiating. This is not a classical derivative identity for the primal
identity function. The inherited base location, scale and features still learn.
Point and log-location losses use the live mu. No changed CDF, likelihood or
quantile values are reported from this training-only view.

With mask-aware averages on actual labels, P=w*MAE(rawmedian/FITrawstd).
U=raw-space properNLL+.1*standardizedloglocationMSE
  +.3*maskedcurrentGaussianNLL+.25*raw90intervalscore/FITrawstd.
Loss values P+U are logged but their naive summed derivative is deliberately
replaced by a declared multi-task update. All33 tensors remain active; no frozen
encoder, unused padding or hidden learned model.

## Exact finite-vector projection

Flatten actual model gradients in named-parameter order (100,349 scalars):
p=grad(P), u=grad(U), d=<p,u>, n=<p,p>.
If d<0 and n>0, u'=u-(d/n)p; else u'=u. Combined g=p+u'.
For n>0, <p,u'>=0 when conflicting, otherwise <p,u'>=d>=0, hence
<p,g> >= ||p||². Coordinates with p_i=0 are unchanged by projection, so widths
are not arbitrarily assigned point gradients. Zero point gradient returns u.
These are exact-real directional properties, not a theorem about AdamW or
finite-step decreases. Additional group learning rates, adaptive moments,
clipping and weight decay can change the applied direction.

Production uses TWO autograd.grad passes at the same pre-update model: P with
retain_graph, then U. Unused paths get explicit zero arrays. FP64 dot and norm
aggregation; denominator replaced by1 when its computed norm iszero; negative
component is removed in FP64, projected arrays rounded to model dtype, then
combined with p. No higher-order gradients are taken. IEEE/nonsmooth/underflow
refinement remains OPEN; numerical tests compare FP32/64 actual values and
gradients at immutable tolerances. Independent reference uses one flattened
vector and a conditional formula, not the production tensor-list reduction.

Two backward passes and O(parameter_count) reductions add training work; all
recipes measure that same graph. The input-only forward schedules still admit
two chronological source/memory scans of O(T) fixed-width work/O(logT) span;
post-scan decoder queries are read-only. Streaming payload remains608B FP32+24B
integer state/sample, model buffers72B separately, runtime metadata/weights/
activations excluded. No new embedded-device speed claim.

## Formal correspondence ledger

PNDS.PointPriorityTraining imports the full ME105 typed neural graph. It declares
the objective-role interface, frozen-correction identity, correct custom-AD
composed tangent, finite-vector projection and exact-direction theorems. Proofs
are limited to those statements. Existing whole-network real/IEEE/gradient and
NormalLaw instantiation obligations remain OPEN; no axiom, sorry or scalar proof
certifies statistical calibration. Python fullsource/ref tests, fitted audits,
authentic outcomes and checkpoint replay are independent requirements.
