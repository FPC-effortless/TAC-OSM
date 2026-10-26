# ME108 — train the integrated role model with protected AdamW proposals

Native forward graph is exactly ME107's102713parameter role-specialized TAC,
with its three FFNs, fixed primary state and inherited input-conditioned scan.
No new attention, state-space recurrence, external memory or learned model is
added. All39 tensors train jointly. Allfive ME103 strong backbones are retained;
their current linear head is discarded and three FFNs initialize freshly.

The registered loss uses real FIT labels only:
`L=NLL+.5pointMAE/FITstd+.1logLocationMSE+.3maskedCurrentNLL+.25intervalScore/FITstd`.
The existing exact-primal uncertainty view stops the median-correction tangent
for NLL/interval only; shared-ancestor conflicts are still possible.

Let `g=clip_1(grad L)` and `p=grad(.5pointMAE/FITstd)`, computed from the same
forward with two VJPs; p is not clipped. Adam moments use g:
`m'=beta1*m+(1-beta1)*g`, `v'=beta2*v+(1-beta2)*g^2`, `t'=t+1`.
Bias corrections, sqrt and epsilon give the AdamW proposal including decoupled
weight decay and each parameter group's learning rate. The complete proposed
parameter displacement is protected in13 exhaustive blocks using ME107's
operator: if `<p,d>>0`, subtract `<p,d>/||p||^2*p`.

Python evaluates the proposal in model precision, guard reductions inFP64,
rounds the displacement, rechecks, then forms the candidate parameters and
checks `<p,candidate-old>` inFP64 again. A residual opposing move is rejected on
nonzero-point-gradient coordinates; truly zero-point-gradient coordinates remain
free. All candidate/moment validation precedes actual parameter mutation.
**Moments advance using the clipped raw combined gradient even if a weight move
is projected/rejected.** Fresh moment state starts empty, no parent optimizer is
reused. The point snapshot is consumed exactly once per step.

There is no optimizer differentiation, line search, extra training step, rejected
batch retry or state cache reuse across changed model weights. `propose()` is a
read-only candidate evaluator for independent original-checkpoint audits and does
not commit moments/weights. Synthetic arithmetic tests compare to independent
tensor equations and unopposed TorchAdamW; they are not extra TAC fits.

`PNDS/RoleGuardedTraining.lean` defines exact-real moments, proposal, protected
step, unchanged forward wrapper and parameter arithmetic. It proves nonpositive
point dot for the ideal protected displacement and unchanged feasible/zero-point
coordinates. Clipping and externally computed gradients are explicit boundary
inputs. IEEE reductions/rechecks, full automatic differentiation and neural
source refinement remainOPEN. A nonpositive computed first-order dot is **not**
finite-step MAE descent, calibration, optimization convergence or generalization.

One fixed five-lineage candidate receives4096new updates each; 65536 sampled
examples and7163648 repeated history-event exposures/model. Inherited backbone
has3072parent+4096new own-lineage steps, fresh FFNs4096. No new trained control;
larger capacity, learning-rate and exposure changes prohibit historical
architectural-superiority claims. CPU and partial profiled dense FLOPs are
reported honestly; optimizer/unsupported operations are excluded from that FLOP
estimate. All exact original models/optimizer/RNG/receipts must be frozen inGit
before already-exposed2015 scoring. Protected2016–2017 numerical rows stay closed.

Allfive seeds must pass the unchanged ME106 engineering criteria. No threshold
adjustment, favorable-seed selection or extra ME108 fits is allowed after the
fixed panel. Original failed gates and all prior source/models remain intact.
