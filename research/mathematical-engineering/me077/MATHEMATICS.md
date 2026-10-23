# ME077 — native dynamics and a coherent conditional physical distribution

All observations are available at prediction time: normalized current state
`x_t`, prior-known logged action `a_(t-1)`, already observed displacement
`v_t=x_t-x_(t-1)`, and nominal elapsed grid time. First history-frame velocity
is zero because no earlier state is supplied. No actual future state, movement
flag or future takeover is an input.

The encoder maps `[x_t,a_(t-1),v_t,log(1+dt)]` to 96 coordinates. Two native
candidate/admission and residual gates update the entity-local fast state:

`h_t = tanh(lerp(h_(t-1), tanh(c_t), sigmoid(g_t))
             + 0.05 sigmoid(rg_t) tanh(r_t))`.

Thus every state coordinate stays in [-1,1]; this is a bounded native
recurrence, not an associative parallel scan, retention theorem or policy.
The decoder consumes the updated state and observed encoding. For each joint:

`pi = sigmoid(g)`

`d = B tanh((sigmoid(g_v) v_t + 0.1 B tanh(r_d)) / B)`

`b0 = max(1e-6, 0.1 D (softplus(r_0)+0.01))`

`b1 = max(1e-6, D (softplus(r_1)+0.01))`.

`D` is FIT-only mean absolute genuine normalized displacement (floor0.0005);
`B=max(4*FIT_servo_step_limit,0.1)`. Both come from source FIT, not REPORT.
The distribution over next displacement is

`p(delta|history)=(1-pi) Laplace(delta;0,b0) + pi Laplace(delta;d,b1)`.

This is a **continuous near-stationary/moving mixture**, not an atom at exact
zero. Component identity is latent: `pi` is not itself the probability that
mean physical14D movement exceeds0.025. A separately supervised global event
head predicts that authentic threshold event using BCE. Its separate scoring
must not be confused with mixture-component calibration.

Optimize mean coordinate proper mixture NLL using stable log-sum-exp plus
0.2 times factual event BCE. Uniform eligible-within-task FIT sampling avoids
unweighted target enrichment. Predict the **mixture median**, appropriate for
absolute error; obtain5%,50%,95% quantiles from the same CDF. The Laplace CDF
uses `0.5+0.5*sign(z)*(1-exp(-abs(z)))` to avoid overflow. Fixed26-step bisection
brackets both means by40 times the larger scale, sufficient for registered
interior quantiles; replay tolerance is fixed prospectively. This algorithm
is a bounded-precision inversion, not an exact symbolic quantile formula.

90% intervals are marginal per coordinate. Proper interval score is
`(upper-lower)+20*(lower-y)_+ +20*(y-upper)_+`, converted to recorded units.
No observation or target is clipped; bounding the moving mean does not bound
the distribution's tails or guarantee actual coverage. No uncertainty-based
hard-switch policy or factual counterfactual outcomes are implied.

This reuses mixture density modeling, proper scoring and native TAC updates.
The scientific output is a measured integrated development result, with a
failed full5% material-gain gate; no claim of a new underlying mathematical rule.
