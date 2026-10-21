# Native innovation memory

ME038 changes TAC's state transition. It does not simply increase the forecast
head or insert an independent Transformer/SSM in place of native memory.
The proposed integration is new to this repository; rank-one delta writes,
normalized addresses, gated recurrence and multichannel pooling are established
ideas. No first-in-literature claim is made without a separate literature audit.

For each observed eight-step patch u and the same-time mean patch across channels
g, let a=[u;g]. Each channel has f,s in R^32 and M in R^(8x8), initialized at zero.
All channel-shared calculations use observed values; there are no future inputs.

```
q = normalize(Wq [f;s])
r = M q
b = sigmoid(Wb [a;f;r])
f' = f + b * (tanh(Wa a + T f + Wr r) - f)
k = normalize(Wk [a;f'])
v = tanh(Wv [f';s])
beta = sigmoid(Wbeta [f';s])
rho = .95 + .0499 sigmoid(rho_parameter)
e = v - rho M k
M' = rho M + beta e k^T
r' = M' q
nu = sigmoid(Wnu [f';s])
j = tanh(Wj [f'-s;s;r'])
s' = s + nu * (tanh(s+j) - s)
```

`normalize(x)=x/sqrt(sum(x^2)+1e-5)`, so norm<=1. Recall influences the fast
proposal/gate and slow innovation; the matrix is part of the coupled recurrence.
The head sees RMS-conditioned[f';s'], not an independent dense matrix decoder.
The same query is used before/after the write within each patch; new queries are
formed at the next patch. All weights including retention are trained end to end.

Conditioned on fixed k,v,beta,rho, the memory transition can be written
`M'=rho M(I-beta k k^T)+beta v k^T`. Since beta in(0,1), norm(k)<=1,
`0 < eig(I-beta k k^T) <= 1`. Thus its operator norm is at most rho<.9999.
The entire coupled Jacobian also contains derivatives of queries, keys, values,
gates and fast/slow dynamics. The conditional bound is not a global contraction
proof or a forecast/long-history guarantee. Affine composition for fixed signals
does not give associative execution of the coupled nonlinear TAC recurrence.

A state-magnitude bound does hold without fixing future learned signals:
`norm(M')_F <= .9999 norm(M)_F + sqrt(8)`. Consequently from zero,
`norm(M_t)_F <= sqrt(8)*(1-.9999^t)/(1-.9999)` in exact arithmetic. Fast and slow
coordinates remain in[-1,1] by convex combinations of tanh proposals. These are
numerical engineering bounds, not proof of useful information or robust control.
No new tolerance is chosen from observed forecast discrepancies.

The forecasting wrapper retains ME037:
`Y=mu+sigma*(AR(z)+alpha_h(context)*tanh(R_h))`, with its monotone horizon gate
and unchanged amplitude envelope. Channel pooling is permutation equivariant
(up to floating-point reduction order). No channel identity or privileged labels
are embedded. All neural controls receive the same shared-channel observations.
The matrix adds64coordinates to native64:128float32coordinates/channel, not one
128-vector for a whole multichannel system. Serving normalization requires the
complete96-step history, and the current trained model resets state per window.

The core processes normalized tokens in order. Window statistics and the dense
context branch use the complete observed96-step window, so raw-history prefix
states cannot be reused as a causal streaming transform. Unit prefix checks
apply to the core given tokens, not to separately normalized raw prefixes.
