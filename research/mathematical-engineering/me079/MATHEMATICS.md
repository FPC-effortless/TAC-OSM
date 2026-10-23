# ME079 — risk-conditioned rational residual shrinkage

Keep the complete native/GRU/Transformer physical predictor. For each horizon
and joint, let its raw median correction be`d`, its learned global-motion
probability be`p`, FIT mean absolute displacement scale be`D`, and horizon be`h`.
Add42active learned scales, initialized by`softplus(s)=0.25`:

`tau=D*h*softplus(s)*(1-p)/(p+0.05)`

`d_final=d*d^2/(d^2+tau^2+1e-12)`.

All quantities available at inference come from factual past observations,
known prior commands, FIT-only scales or learned outputs. No actual future
movement flag, oracle expert selection or TEST calibration. Each architecture
gets the same added mechanism, training budget and selection opportunity.

For fixed raw weights and inputs, the gain lies in[0,1], so correction magnitude
cannot increase; zero correction remains zero. This is a mathematical amplitude
constraint, **not** a guarantee of factual accuracy or calibration. End-to-end
training changes the raw model as well, so comparison of two separately trained
models is not governed by the fixed-weight bound. Empirically ME079 still fails
the stringent original stationary-preservation and learned-rival gates.

Shift median/lower/upper together by`d_final-d`. Their ordering is preserved;
width is unchanged by this transform at fixed weights. Training is the same
proper marginal pinball loss plus separate event BCE. No coherent joint future
trajectory distribution, simultaneous coverage or action utility follows.
The event probability is separately supervised rather than derived from quantiles.

Adaptive shrinkage is established prior art (e.g.Donoho & Johnstone,1994,
[Ideal spatial adaptation by wavelet shrinkage](https://doi.org/10.1093/biomet/81.3.425)).
This rational gate is an engineering design, not that paper's estimator or a
new optimality theorem. Native gated-delta, GRU and causal attention foundations
remain as cited in ME078. Both study recipes and every original failure remain
visible; no retrospective gate relaxation or further fit is authorized here.
