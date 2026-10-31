# Conventional controls: exact implemented equations

All models map observed `Long[B,T]` byte IDs in 0–255 to 256 logits per byte.
Owners are unique explicit metadata and never predictive features. No optimizer
or dataset is executed by these files. Unknown forward keyword fields reject.

The common head is `z=LayerNorm(h, eps=1e-5)` and
`logits=W_o((W_u z+b_u) * SiLU(W_g z+b_g))+b_o`. LayerNorm uses population
variance and learned scale/bias. Every counted learned tensor lies on the scored
pathway. Changing its real FFN width allocates capacity rather than padding.

| Control | Learned parameters | Core dimensions | Head width | Persistent payload |
|---|---:|---|---:|---|
| GRU |100346|byte embedding32, hidden64|189|64 floats/stream|
| Selective diagonal recurrence |100518|embedding32, state64|211|64 floats/stream|
| Causal Transformer |100454|embedding32,4 heads×8, FFN64|259|all observed Long bytes,8L bytes/stream|

Maximum relative parameter gap across these and original098100370 is below0.2%.
This does not establish matched training compute or any architecture winner.

## GRU

For `e_t=tanh(E[x_t])`, split input and recurrent affine terms into reset,
update and candidate coordinates in the native PyTorch `(r,z,n)` order:

`r=σ(W_ir e+b_ir+W_hr h_prev+b_hr)`

`z=σ(W_iz e+b_iz+W_hz h_prev+b_hz)`

`n=tanh(W_in e+b_in+r*(W_hn h_prev+b_hn))`

`h=(1-z)*n+z*h_prev`; output the common head. State is the final64D `h`.

## Selective diagonal recurrence

`e=tanh(E[x])`; `a=.999*σ(W_a e+b_a)`;
`offer=tanh(W_b e+b_b)`; `s_t=a_t*s_prev+(1-a_t)*offer_t`.

`h=tanh(W_s s+b_s+W_e e+b_e)`; output the common head.
This is a simple input-dependent affine/diagonal recurrent control, not an
implementation or reproduction of Mamba. The parallel pair-reduction scan and
literal serial recurrence are checked numerically. State is the final64D `s`.

## Transformer

Concatenate all legally observed prior bytes with the current invocation and
recompute one causal block. Input is `sqrt(32)*E[x]+PE(position)`;
`PE_(2i)=sin(t*10000^(-2i/32))`, `PE_(2i+1)=cos(...)`.

With `z=LayerNorm(h)`, each of4 heads forms8D learned `q,k,v`, computes
`softmax(q k^T/sqrt(8)+causal_mask)*v`, concatenates heads, applies the learned
attention output projection and adds the residual. A second pre-norm residual
FFN is `W_2 GELU(W_1 LayerNorm(h)+b_1)+b_2`, with exact-erf GELU. The common
head scores only positions from the current invocation. Dropout is0.

State contains the actual observed Long token matrix, with width equal to
clock. This is a source-faithful full-prefix attention baseline, not an
optimized KV-cache decoder; costs and actual raw-token state bytes must be
charged. It has no fixed maximum context or learned positional table.

## Evidence and outstanding proof obligations

`conventional_reference.py` independently implements these forward equations in
NumPy without invoking any PyTorch model forward. Checks cover full numerical
output/state correspondence in FP64, future poisoning, chunk/prefix consistency,
actual parameter count, every parameter tensor receiving finite nonzero gradient,
and one independent central-difference coordinate per parameter tensor.

These finite constructed checks do not prove global AD correctness, IEEE
refinement, all-shape numerical equivalence, useful learning or full PNDS
certification. Lean typed-graph records/source reciprocity, full proof and actual
source/data/training-budget preflight are OPEN and owned by the main study.
