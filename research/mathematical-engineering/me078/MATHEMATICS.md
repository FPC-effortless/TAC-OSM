# ME078 — native kinematic forecasting and controlled computation

Native state is `(f,s,M)`:32fast,32slow,8×8value-by-key association. For each
observed state/previously issued command, compute state-coordinate command error,
observed velocity/acceleration and their FIT-scale asinh representations. No
future commands, states, target movement flags or takeover labels enter the
encoder. Every training-prefix forecast is causal, including the Transformer.

The native updates are convex bounded fast/slow steps:

`f' = lerp(f,tanh(c),sigmoid(g))`

`s' = lerp(s,tanh(s+tanh(innovation)),sigmoid(rate))`.

Thus both coordinate sets remain in[-1,1]. Normalize a learned8Dcurrent address
`k`; read the prior matrix via`M k`. Retain and correct:

`D=alpha M`, `M'=D+beta (tanh(v)-D k) k^T`, projected to Frobenius radius4.

Retention/write are separate learned gates. Shared write/read address prevents
arbitrary key/query coordinate mismatch for the same observation, but does not
prove relevance, retention, learned memory benefit or semantic identity binding.
This is established gated-delta math adapted to a native dual-state predictor,
not a newly invented memory rule or an SSM conversion. Each actual eight-frame
window starts its own recurrent state; unbounded streaming memory is unqualified.

All models share a kinematic ordered-quantile interface. For horizon`h`,

`delta_h=B_h tanh((h*(sigmoid(gv_h)*velocity+sigmoid(gs_h)*FIT_servo)
                    +B_h*tanh(r_h))/B_h)`

`median_h=x_t+delta_h`

`lower_h=median_h-D*h*(0.01+softplus(l_h))`

`upper_h=median_h+D*h*(0.01+softplus(u_h))`.

`D` and`B_h=max(4*FIT_servo_step_limit,0.05)*h` come from FIT only. All raw
outputs and per-coordinate gates are trained. Quantiles are ordered, but neither
joint-distribution consistency across coordinates/horizons nor guaranteed
coverage follows from this parameterization. A separate event head is trained
on factual mean14Dphysical movement>=0.025; it is not presented as a CDF implied
by the quantiles. No latent reasoning, action policy or counterfactual effect
is implied by more direct future readouts.

Loss is a physically scaled pinball score at5/50/95%, normalized by FIT typical
displacement and horizon, plus0.1event BCE. All eight prefixes receive actual
1/4/16-frame future supervision wherever the source future exists before human
takeover. Missing/censored futures contribute no target loss. Target validity
is never an inference feature.

To improve rare-transition gradient coverage without grossly enriching the
training objective, the proposal combines half uniform and half moving origins.
For within-task genuine movement prevalence`p`, use balance weights
`w_m=1/(0.5+0.5/p)` and`w_stationary=2`. These are the exact nominal half-mixture
importance ratios. Finite round-robin task/component counts differ by at most
one draw per task over49,152windows; therefore an exact finite-schedule
within-task equality is not asserted. The nominal proposal correction, frozen
schedule and tiny balancing discrepancy are shared by all arms; no target-aware
routing is used at inference.

Computation matching uses an explicit algorithmic scalar-operation estimate:
ATen dispatch arithmetic/reductions/transcendentals/softmax/backward/gradient
clip, plus analytical AdamW16operations/parameter and fixed bookkeeping.
Every observed floating operator must be classified or preflight fails. The
independent PyTorch matrix profiler verifies the dominant GEMM subtotal.
This does **not** prove hardware-instruction, runtime, energy or universal FLOP
parity; the comparison is qualified only under the stated estimation convention.
No dummy parameters or work are added to force a matching profile.
