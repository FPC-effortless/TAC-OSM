# ME-055 mathematical architecture

All physical vectors below are normalized using FIT-only statistics. For state
`x_t` and proposed action `a_t`, the shared identified servo prior is

```text
p_t = clip(g * (T(a_t) - x_t), -ell, ell).
```

The learned physical prediction is identity anchored and bounded:

```text
x_hat_(t+1) = x_t + p_t + d * tanh(r_theta(z_t, a_t)).
```

Here `g`, `ell`, and `d` are positive FIT-only vectors. The formula prevents a
single learned residual from creating the unbounded physical jump seen in
ME-052.

The native belief is `(f_t, s_t, m_t)` with fast and slow states in
`R^32` and a compact associative trace in `R^8`. A normalized address `q_t`
reads the trace by elementwise content interaction:

```text
c_t = m_(t-1) * q_t.
```

A fused projection produces the fast proposal/gate and write address. A second
synchronous projection produces the trace value, write strength, slow proposal
and slow gate from one state snapshot. The bounded updates are

```text
f_t = f_(t-1) + sigmoid(g_f) * (tanh(u_f) - f_(t-1))
m_t = tanh(rho * m_(t-1) + sigmoid(beta) *
           (tanh(v) - rho * m_(t-1)) * k_t)
s_t = s_(t-1) + sigmoid(g_s) *
      (tanh(s_(t-1) + tanh(u_s)) - s_(t-1)).
```

The diagonal associative trace is the principal ME-055 recurrence change. It
retains relevance-conditioned content addressing while avoiding a serial tiny
matrix read/write path.

For observed physical innovation `e_t` and takeover innovation `b_t`, one
cross-modal projection produces a bounded joint-state correction. Its mixing
strength is

```text
i_t = 1 - (1 - mean(abs(tanh(e_t / d)))) *
            (1 - abs(tanh(b_t))).
```

so zero innovation produces an exact no-op. The action plan is rolled forward
for 16 steps without future observations. Physical and onset losses use only
labels available before the first human intervention.

