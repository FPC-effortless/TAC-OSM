# ME095 — context-conditioned native state with an affine filter branch

Whole-model capability development, not attribution or a matched architecture win.
All original predictive thresholds remain. ME094 still failed quietH1/stricterCI;
its five originals/scores and successful independent replay are preserved.

## Known inputs and context

Same120FIT/40previously exposedREPORT, original14state channels, lag1recorded
reference, positive observed capture elapsed and exact Git-owned gray32camera.
Add known requested operation type (five planned tasks), plus six sin/cos features
of the CURRENT recorded capture clock modulo1/4/16seconds. Periods fixed before
fits. No episode identity, curator episode age, total duration, ending, outcome,
future clock or same-frame reference. Public metadata and existing source records
qualify these fields; no protected source is opened. Numeric source timestamps
are absolute clocks; only periodic phases enter, not epoch scalars/record IDs.

Old models did not receive this context. Extra information and different model/
training costs are disclosed; passing this practical same-target gate cannot
establish an equal-information architectural winner.

A5x64learned operation embedding initializes native fast/slow priors and enters
the encoded event. A6to64clock projection adds current phase context. Unknown
operation uses a zero vector, not an unused learned embedding row; no unseen-task
accuracy is claimed. Both context paths start atzero, all parameters trainable.

## Explicit composable physical observation state

Let e_t be current numeric/CNN/context encoding, independent of the new filter
state. A64to28projection yields nonnegative rates lambda=softplus(W e_t+b).
For observed rho=15*elapsed>0, a_t=exp(-lambda*rho). Let u_t concatenate current
normalized qpos and raw observed velocity. The new28scalar state obeys

k_t=a_t*k_(t-1)+(1-a_t)*u_t.

This is an input-selective diagonal affine branch, not a replacement for native
fast/slow/associative memory. Every component is a convex combination of its old
estimate and a factual observation; in real arithmetic it remains within their
historical range for arbitrarily changing gains. Physical safety/accurate memory
or meaningful long-history prediction does not follow from this numerical bound.

Affine segments compose associatively:

(A2,B2) o (A1,B1)=(A2*A1, B2+A2*B1).

The implemented inclusive prefix scan uses doubling offsets and elementwise
products/additions, without cumprod/division underflow tricks. O(TlogT)work and
O(logT)dependency depth for this small diagonal branch. Independent serial values
and parameter/input gradients agree in numerical tests. The coupled native TAC
recurrence is STILL serialT: the complete architecture is not advertised as an
associative scan or all-parallel sequence model.

Level/velocity evidence receives a28to32projection and enters the common encoding;
filtered velocity enters the physical expert. Addresses/input projections and
local causal filters remain shared over batch/time, then native nonlinear states
update. Added15field explicit state has1,359scalar values/entity (original1,331+
28filter values), excluding weights/cache inputs/outputs/containers. Whole/chunk/
streaming states and predictions agree under the declared numerical contract.

The retained learned median, causal value retrieval, coherent translated full
multimodal distribution and proper interval/density scoring stay unchanged.
No old latent mixture weight is called a calibrated physical-motion probability.
All outputs depend only on factual prefix inputs and known requested context.

## Budget, objective and decision

104,689learned values=101,173retained ME094+1,820filter rates+928filter evidence+
320operation embedding+448clock projection. No dummy padding/frozen learned encoder.
ALL5indexedME094parents, no selection, optimizer reset.2,048newupdates/seed,
retainedLR.0001/newcontext-filterLR.0003, AdamWwd.0001clip1. SameCRPS+.2NLL+
20medianMAE+.2factualmotionBCE and original3taskPCGrad; no new teacher labels or
quiet-only penalty. Parent pipeline17,408+2,048=19,456updates. Retainednative/visual
15,360ownlineage,location11,264,oldlocalnoise9,216,olddecision6,144,new3,516values
2,048updates. All attempted work remains charged; no matched-compute winner.

All original numerical gates plus original95%and additional99.64286%paired CI for
14declared recipes, and strict independent original replay, remain compulsory.
Allfiveoriginal weights/optimizer/RNG/receipts must be Git-frozen beforeREPORT.
Adaptive exposedDEV only. No protectedCAL/TEST/tissue-box, architecture superiority,
causal intervention, autonomous policy, deployment, generic intelligence or customer
value is established. References: EMA/linear-state filtering, Blelloch affine-prefix
scan, input-selective state-space prior art, original native TAC and proper scores.
