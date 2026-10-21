# Native architecture engineering in ME035/ME035B

The work changes and trains the predictive architecture. Mechanism explanation
is limited to implementation decisions and numerical correctness, not a separate
state-value research program.

**Joint prediction.** A direct causal-context path and SwiGLU([context;fast;slow])
path train together. The common predictor is not frozen and the nonlinear path
is not forced to vanish at zero state. This avoids restricting task adaptation
to a correction around an immutable, potentially weaker predictor. SwiGLU is a
known gated feedforward mechanism (GLU Variants Improve Transformer,2020), not
a new TAC invention. Here it is used as a continuous outcome readout.

**Causal normalization with level retained.** For each observed96x7history,
mu=mean(history), sigma=sqrt(var(history)+.0025), z=(history−mu)/sigma.
Only preceding observations determine those quantities. The context retains mu
and log(sigma), so normalization does not discard level/volatility information.
Decode local predicted changes by multiplying by sigma and adding the last
observation. This adapts the RevIN principle (Kim et al.,ICLR2022) to native
memory with trainable context and preserved absolute conditioning. Global source
mean/std are separately fitted on original training rows only.

**Residual slow state.** Original native fast update is retained:

```
q = tanh(Au+Tf+b)
g = sigmoid(Gu u+Gf f+bg)
f_next = f+g*(q-f)
v = sigmoid(V[f_next;s]+bv)
innovation = tanh(W[f_next-s;s]+bslow)
candidate = tanh(s+innovation)
s_next = s+v*(candidate-s)
```

With a zero innovation map (W=0,bslow=0) and s=0, derivative of the slow proposal with respect to s
is identity; the slow update's direct Jacobian is also identity. The old
zero-proposal update has direct derivative1−v. Small slow-matrix initialization
starts near the identity case while retaining learned nonlinear updates. This
does not guarantee global nonexpansiveness or useful long-term memory. Bounded
gate interpolation of bounded candidates preserves |f_i|,|s_i|<=1 from zero.
Innovation coordinates [f−s;s] are an invertible reparameterization of [f;s];
the residual identity term is the structural change, not extra raw information.

**Multiscale initialization.** Bias−log(tau−1) gives initial gate1/tau at zero
other logits, with fast tau2..32 and slow32..256. Fast transition initializes
orthogonal with norm.9. These are learned starting conditions, not fixed decay
kernels or verified empirical half-lives. State readout uses split RMS eps.01.

**Working backbone integration (ME035B).** The first round's stronger ridge
reference motivates replacing the summary-only direct path with a full-history
trainable linear path:

```
prediction = last + AR_theta(flatten(global_normalized_history))
             + local_sigma * nonlinear_TAС_correction.
```

Initialize AR using FIT-only ridge; inherit each seed's trained native core and
SwiGLU features, reset the correction output to zero, then train ALL coefficients.
Anchor the backbone with25*mean((W_AR−W_AR_initial)^2) while using a smaller
learning rate on it. This keeps a working representation/predictor available
while memory learns a residual. It does not force the base to stay immutable.
Initial forecast equals ridge within declared floating-point tolerance; selected
trained models use nonzero updates. Residual training here does not by itself
prove that memory contributes unique temporal information.

The first round has18924active coefficients per stage; the composite has32154.
Its capacity, FIT-only ridge warm start and inherited training differ, so its
lower development error is not a parameter/resource-matched architecture win.
All development access, parent hashes, budgets and attempts are recorded.

The model currently normalizes/reprocesses a whole96-step window per forecast.
Changing window statistics would change old normalized inputs, so cached native
states cannot simply be reused for a shifted window. Core state is64float32
coordinates (256B), but this is **not** a constant-time streaming serving claim.
The physical-input wrapper stores the14training-only normalization constants and
inverts output scaling. Faster online normalization is a future engineering task.
