# Next architecture direction — proposal, not a fitted improvement

ME024 tests two bounded changes to native TAC rather than another architecture
leaderboard. Its tied slow feedback is not a selective state-space model: the
native gates and proposals depend on the current hidden state, so no affine
associative scan follows. Adding normalization does not change that fact.

A distinct next hypothesis is a selective diagonal memory lane with an exact
continuous-time discretization. This should receive its own frozen real-data
protocol and capacity-matched TAC controls before any fits. It is not included
in ME024's evidence or primary gate.

For observed event input u, define positive bounded decay rates and intervals:

```
lambda_j = lambda_min + (lambda_max-lambda_min) sigmoid(theta_j)
delta_tj = delta_min + (delta_max-delta_min) sigmoid(d_j(u_t))
a_tj = exp(-lambda_j delta_tj)
b_tj = tanh(B_j u_t)
m_tj = a_tj m_(t-1)j + (1-a_tj) b_tj
```

For input held constant during the interval, this is the exact solution of
`dm/dtau = -lambda (m-b)`. Compute `1-a` with `-expm1(-lambda*delta)` to retain
precision for small intervals. Positive finite lower/upper bounds are required
in an eventual contract; values must not be chosen after viewing development
scores. Observed elapsed time and a learned effective interval must be clearly
distinguished. Source timestamps and the forecast boundary must stay causal.

Starting within [-1,1], every coordinate remains bounded by convexity. With
input-independent-of-state selectors, the recurrent derivative is exactly
`diag(a_t)`, and over k events it is
`diag(exp(-lambda * sum(delta_t)))`. This yields an explicit, auditable
retention-timescale parameterization. It does not establish useful memory,
whole-model contraction or a predictive advantage. Very slow poles also weaken
instantaneous injection, which the training/readout design must address.

Each event represents an affine map `(a,c)`, with `c=(1-a)*b`. Composition is

```
(a2,c2) compose (a1,c1) = (a2*a1, a2*c1+c2)
```

This is associative in real arithmetic, enabling a parallel prefix scan for
this lane. Floating-point scan/window/streaming agreement needs an explicit
measured tolerance; mathematical associativity is not bitwise equivalence.
Making the selectors depend on m would invalidate this direct scan argument.

Keep native fast/slow paths and this candidate lane distinct in the next
factorial. Predeclare whether a lane replaces native state capacity or adds it;
match total trainable capacity and target exposure for the primary contrast.
Do not simultaneously add query attention, extra losses and learned gates.
Use the same four actual causal prediction targets initially. Register stable
pole derivatives, input perturbation causality, scan/stream equality, runtime,
real-data state-use diagnostics and a meaningful accuracy gate. Previously
exposed development outcomes remain development; no test cohort may be opened
merely to rescue an unsuccessful architecture hypothesis.

LLM-scale capability is a separate question requiring a suitable training
objective, data, budget and independent evaluation. Architectural complexity
or a familiar SSM equation cannot substitute for those measurements.
