# ME-056 anchored predictive-state equations

For non-boundary observation `x_t`, only fast evidence state changes:

```text
u_t = tanh(E x_t)
[p_t, g_t] = W_f [u_t; f_(t-1); m_(t-1)] + b_f
f_t = f_(t-1) + sigmoid(g_t) * (tanh(p_t) - f_(t-1)).
```

Slow state `s` and relevance trace `m` remain unchanged by this observation.
They can still receive bounded factual physical/takeover corrections after an
outcome is observed.

At fixed, outcome-independent 128-frame anchors, the model applies its learned
address and synchronous state projection:

```text
[k, r, a] = W_b [u_t; f_t; recall_t] + b_b
k_t = address(k + sigmoid(a) tanh(r))
[v, beta, i, g_s] = W_s [f_t; s_(t-1); recall_t] + b_s
m_t = tanh(rho m_(t-1) + sigmoid(beta)(tanh(v)-rho m_(t-1)) k_t)
s_t = s_(t-1) + sigmoid(g_s)(tanh(s_(t-1)+tanh(i))-s_(t-1)).
```

On every second observed FIT transition, the same `W_b` and `W_s` maps receive
dense factual targets derived from bounded physical delta, action-state error,
control authority and takeover onset. This gives the slow pathway direct
supervision in two large batched operations. It adds no parameters and does not
read DEV, TEST or post-takeover counterfactual targets.

