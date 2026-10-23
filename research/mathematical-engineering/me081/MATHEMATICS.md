# ME081: shared horizon-query predictive distributions

## State and causal context

The original native32Dfast/32Dslow/8×8associative state and bounded transitions
are preserved. Observed32frame chunks have full gradients across the whole chunk.
Their actual per-prefix observed features remain state14D, known lag-one command
mapped to state coordinates14D, asinh-scaled observed velocity/acceleration28D,
and nominal1/15s time. No future command, state or takeover indicator is a feature.
Batched encoding and horizon queries operate independently at each causal prefix.

At prefix t, the native core emits z_t. The decoder constructs

q_(t,h) = [encoded_observation_t ; z_t ; observed_motion_context_t ; b(h)]

where b(h)=[h/16, log(1+h)/log17, sqrt(h)/4, exp(-h/4)]. One shared SwiGLU map
and one shared85-output projection predict the distribution for each registered
horizon h in{1,4,16}. There are no independently parameterized horizon heads.
This is shared conditional forecasting, not a guarantee of coherent joint paths
or extrapolation to unseen horizons. The inference interface still exposes the
three trained horizons. Original normalization is fixed FIT-only, not rolling.

## Coordinate distribution

Each coordinate/horizon predicts mixture weight p, bounded motion location m,
positive quiet scale b and positive moving scale c:

F = (1-p)Laplace(0,b) + p Laplace(m,c),
b = max(1e-6, delta_scale*h*.1*(.01+softplus(raw_b))),
c = 2*b + delta_scale*h*(.01+softplus(raw_c)).

The quiet component is zero-centred; no oracle motion label chooses a component.
Moving scale is wider by construction, not capped to an arbitrary business limit.
The conditional median and5/95%quantiles come from this same CDF. A separately
supervised global factual displacement event is not a latent component identity.
Marginal coherence does not establish cross-coordinate/temporal dependence.

## Proper distribution scoring

For one Laplace variable, E|X-y| = |m-y|+b*exp(-|m-y|/b).
For independent X~Laplace(0,b), Z~Laplace(m,c), with d=|m|,

E|X-Z| = d + [c^3 exp(-d/c)-b^3 exp(-d/b)]/(c^2-b^2).

The model enforces c>=2b, so the divided difference never reaches its singular
c=b case. E|X-X'| for equal independent Laplace variables is1.5b. Expanding
both terms with mixture weights gives exact marginal CRPS:

CRPS(F,y)=E_F|X-y| - .5 E_F|X-X'|.

The code is independently checked against the integral
integral (F(z)-1[z>=y])^2 dz, using deterministic numerical quadrature. This is a
software mathematical reference, not synthetic model-training evidence.

The real-data loss averages valid dense prefix/horizon scores:

CRPS * physical_scale/(FIT_displacement_normalizer*h)
+ .02 * (negative_normalized_log_density + log(physical_scale))
+ .2 * BCE(global factual motion probability, observed physical event).

Positive fixed combinations of proper marginal scores are proper over the
unrestricted marginal distribution class, conditional on the registered
population weighting. The restricted mixture family and finite optimization
need not recover the true distribution. BCE pertains to a separate observable
binary event. No guarantee of calibration, a global joint likelihood, causality,
intelligence or architectural advantage follows from the objective alone.

## Data and resource scope

Five cold starts, all100,165active/deployed parameters, original source120FIT/
40exposedREPORT, 2048updates×24windows×32supervised prefixes per fit. Window
sampling uses nominal uniform/motion mixture importance weights. The target
population is eligible task-balanced windows with their dense-prefix losses,
not a claim of uniform unique transitions. Finite component task counts differ
by at most one; no exact finite-population unbiasedness assertion.

Training uses four times ME080's observed-prefix exposure. Different context,
loss and sample pools mean there is no matched-compute architecture claim.
Primary readout keeps the exact original eight-frame task; secondary readout
uses only registered, actually available32frame autonomous histories. Its matched
short-history subset is reported separately to avoid comparing different query
populations. Longer source-independent memory utility remains unqualified.

Float32neural parameters; float64distribution inversion/scoring; fixed original
raw replay rtol1e-5/atol1e-6. Code/checkpoints/optimizer/RNG and all case-level
predictions must be Git-owned and independently replayed. No old model bytes,
registration, tolerance or negative result may change.
