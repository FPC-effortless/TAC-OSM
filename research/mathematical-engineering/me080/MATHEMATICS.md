# ME080 mathematical and execution contract

## Native state

State `S=(f,s,M,x_prev,v_prev,seen)` has32fast+32slow+64associative+14previous
state+14previous velocity+1flag float values: **157float32values =628bytes per
entity** excluding tensors/allocator/container overhead, model weights and
transient activations. Stored physical values are normalized by fixed FIT-only
statistics; no rolling-window normalization invalidates the cached coordinates.
Fixed nominal frame spacing1/15s is not learned irregular-time integration.

Observed features are current14Dstate, known previous command mapped into state
coordinates, asinh-scaled observed velocity and acceleration, nominaldt. The
encoder, memory address and observed contribution to the fast update are pure
functions of those features. They are evaluated once per chunk. Explicitly,
`W_fast[obs;f;s]+b = (W_obs obs+b) + W_state[f;s]`.
This algebraic factorization preserves the update in real arithmetic; float32
operation ordering can differ. Unit tests compare it to the frozen original
native full linear expression. No old checkpoints are modified.

The fast and slow updates are the original bounded convex native updates. The
8×8memory uses retention and delta error-write gates, normalized shared address,
read-before-write, and radius4projection. `f,s` remain within[-1,1] from zero
initial state, and `||M||_F<=4`. Bounded state does not imply predictive accuracy.

## Conditional coordinate distribution

For each horizon h in{1,4,16} and coordinate j, displacement d has density

`p(d|history)=(1-sigmoid(l))*Laplace(d;0,b0)+sigmoid(l)*Laplace(d;delta,b1)`.

`b0=max(1e-6,delta_scale*h*.1*(.01+softplus(u0)))`;
`b1=max(1e-6,delta_scale*h*(.01+softplus(u1)))`.
The learned motion delta is bounded, and combines observed velocity, FIT servo
and a nonlinear residual under learned gates as inME078. The zero-centred
component is a modelling prior, not a hard oracle label or forced no-motion
prediction. Learned scales are positive, with no arbitrary postfit upper cap.

The objective is proper marginal mixture negative log likelihood, physical-unit
Jacobian included, plus0.2binary cross entropy of the separate factual global
movement event. Targets and importance weights are authentic and shared with
the existing registered dense-prefix supervision. No latent component labels
are manufactured. A low latent moving-component probability is not the same
as a low probability of global actual movement. The latter has its own head.

Coordinate medians and nominal90%intervals come from the same mixture CDF;
40bisection iterations in float64, bracket40times the larger scale beyond both
component centres. For probabilities.05/.5/.95 this bracket contains the desired
quantile. Ordering follows CDF monotonicity, not a separately learned constraint.
Marginal mixture likelihoods do not claim calibrated cross-coordinate or joint
future trajectory dependence. Event probabilities can still be miscalibrated.

## Prospective numerical contract

Neural arithmetic and cache float32. CDF inversion and all scalar physical
scores float64 BEFORE arithmetic, not after summing float32terms. Raw replay
rtol1e-5/atol1e-6 fixed before fitting. This addresses numerical amplification in
new scoring without relaxing historical ME078/079limits or claiming universal
hardware agreement. Clean-clone and external runs must still pass independently.

## Scope of reused computation

`step` processes ONE observation and prior-known command using caller-owned
state. `forward` optionally accepts/returns the same state. Chunk boundaries
are mathematically transparent, and callers may detach gradients explicitly.
No global memoization, hidden entity table or state mutation. Different weights,
normalization or stream restart require cache invalidation. Carrying state across
40test inputs proves execution equivalence only. Real training/evaluation resets
at8frames; persistent longer-history forecast accuracy remains unqualified.

No GPU throughput, fused kernel, associative scan, O(log T)parallel native
training or microsecond claim is made. The nonlinear TAC recurrence retains
sequential dependence; observation-only work is batched, while new inference
observations can reuse prior state instead of replaying the prefix.
