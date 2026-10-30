# LANGACQUIRE097: acquisition-to-query learning contract

This study changes supervision and episode exposure, not the inherited TAC086
transition. It uses the exact `CompactDualClippedTAC086` graph, original GRU094
and diagonal selective SSM094 graphs, and an explicitly implemented standard
two-layer causal Transformer. Counts are respectively 100521, 100661, 100404
and 100484 active learned parameters. No dummy or inactive padding is permitted.
The model/output/source and actual parameter checks are executable in
`models.py`; all source pins must be sealed before an optimizer step.

## Typed causal sequence and objective

Each original-source episode is an acquisition byte string A, an observed query
Q and a future target Y. The dataset builder owns labels, source identities,
split assignment, lengths and selection. The model accepts only observed
`Long[B,T]` byte IDs, complete owned state and stream-owner strings. The
categorical alphabet is the same 256 bytes for all arms. Loss masks and future
targets are not arguments to the model.

For training only, form R=A||Q||Y, X=R[:-1] and Z=R[1:]. At position t, the model
has observed X[:t+1] and predicts Z[t]. Its query-target mask is

`m[t] = (t >= len(A)+len(Q)-1)`, intersected with the valid non-padding mask.

Thus the first active prediction is the first Y byte after the last observed Q
byte. Both objectives compute every valid chronological input and the entire
cross-entropy array. The query objective averages only active Y terms; the
full objective averages every valid next-byte term. The number of observed
inputs and supervised targets are separate resource/exposure quantities.
Right-padding must be contiguous, query masks nonempty and targets legal byte
indices. Padding does not create an observed-history entitlement in evaluation.

\[
 S_A=F_\theta(A;S_0),\quad
 S_{AQ}=F_\theta(Q;S_A),\qquad
 \ell_t=\log\sum_j\exp(z_{tj})-z_{t,Z_t},\quad
 L_m={\sum_t m_t\ell_t\over\sum_t m_t}.
\]

Acquisition is unrolled in the same differentiable graph. A process/chunk
boundary is not a stop-gradient. A shared parameter has a direct query path and
an acquisition-state path, schematically

\[
 {dL\over d\theta}=
 {\partial L\over\partial\theta}\bigg|_{S_A}
 +{\partial L\over\partial S_A}{dS_A\over d\theta}.
\]

The direct loss gradient of inactive acquisition logits is zero; their *state*
can still receive later query gradients. Local tests require a finite, nonzero
query-loss gradient into acquired state and distinguish a deliberately detached
negative control. This proves local connectivity, not that the optimizer learns
useful acquisition. Inference receives A then Q and generates its own bytes;
training's teacher-forced Y must not cross that boundary.

## Actual states and transition graphs

Every `State097` has `arm`, unique `owners`, integer `clock`, and named floating
tensors whose first dimension is batch. Reset/donor/corruption interventions
replace content while preserving recipient identity and legitimate clock.
The wrapper changes metadata/layout only; neither recurrent source is replaced
by an operator reference.

| Arm | Actual per-stream state, excluding clock/owner metadata |
| --- | --- |
| TAC086 | fast/slow matrices [8,8] each; aligned [2,8,8]; wave [4,2]; basis [48]; concept seeds [4,8]; lexical [3,32]; previous writer [8]; upper [32]. 480 FP32 values, 1920 bytes. |
| GRU094 | Two layers of hidden state [2,83], exposed batch-leading by a reversible transpose. 166 FP32 values, 664 bytes. |
| SSM094 | Hidden state [128]. 128 FP32 values, 512 bytes. |
| Transformer097 | Two layers, each with K and V [4,clock,12]. 192*clock FP32 values, 768*clock bytes. The actual cache grows; it is not hidden or padded to a fictitious fixed state. |

TAC086 computes 32-dimensional byte/position features and three observed lexical
lags. A normalized learned writer is retained for the next observed value; a
separate normalized query consumes current lexical features and previous upper
state. The observed value combines the fixed eight-bit byte code with a learned
residual before normalization. The fast matrix uses a decayed clipped-error
rank-one write; the slow matrix uses a decayed weak error write. Their query
reads have a learned mixture. Two time-aligned matrices, four rotated waves,
a cumulative basis mean, four concept seeds and an upper recurrent state feed
the actual decoder. Fast/slow feature projections are not additional independent
recurrent memories. The concept linear solve is the source four-node
Laplacian solve followed by nonuniform evidence-conditioned pooling.

TAC's 160-dimensional normalized decoder input passes through the source gated
FFN (width94) and 256-byte head plus concept readout. The source also owns an
8192-byte deterministic byte-code buffer; this is not a learned encoder or free
per-stream state. All encoder, writer, query, gate, recurrence and decoder
parameters count toward the 100521 total.

GRU094 uses a 256x32 embedding, two genuine GRU layers of width83 and a learned
256-byte head. At each layer the PyTorch source GRU uses
`r=sigmoid(W_ir*x+b_ir+W_hr*h+b_hr)`,
`z=sigmoid(W_iz*x+b_iz+W_hz*h+b_hz)`,
`n=tanh(W_in*x+b_in+r*(W_hn*h+b_hn))` and
`h'=(1-z)*n+z*h`; the second layer consumes the first layer's current output.
SSM094 uses a 256x32 embedding, 128-dimensional input-dependent
diagonal update
`h'=a*h+(1-a)*tanh(offer(e)+input(e))`, with
`a=exp(-softplus(raw_decay)*sigmoid(gate(e)))`, and a width154 source gated
FFN/head: `logits=W_out((W_up*h')*silu(W_gate*h'))`, including all biases.
This source control is not a Mamba-family performance/equivalence claim.

Transformer097 uses a 256x48 embedding, deterministic sinusoidal positions, two
pre-normalized blocks with four 12-dimensional attention heads and FFN width290,
then normalization and the 256-byte head. Attention admits exactly source keys
at positions no later than the current query. Cached K/V tensors remain
differentiable across acquisition. No dropout, learned task label, target length
or evaluator function enters inference. The maximum registered cache length is
4096 bytes and is enforced by the source implementation.
For each block, `u=LayerNorm(x)`, `(q,k,v)=W_qkv*u+b_qkv`,
`a=softmax(q*K^T/sqrt(12)+causal_mask)*V`,
`x1=x+W_projection*a+b_projection`, and
`x2=x1+W_down*gelu(W_up*LayerNorm(x1)+b_up)+b_down`.
The new k/v join the old complete cache in chronological order. Multihead
reshape/concatenation preserves four heads of width12; final normalization and
the output linear map produce [B,T,256] logits.

## Limits, cost and correspondence

Serial chunking satisfies `F(A||Q,S)=F(Q,F(A,S))` in exact mathematics when the
complete state and time are carried. This holds for arbitrary causal nonlinear
transitions; it is not a proof of an efficient associative parallel scan.
Actual full/chunk FP32 forwards, gradients, state shapes, ownership and masks
require the separate source tests with declared tolerances.
The local native-state query-gradient fixture compares AD with a central finite
difference in each of the four actual models using FP64, absolute tolerance
1e-7 and relative tolerance1e-4. This is a local source check, not a proof of
every parameter derivative or finite-precision trajectory.

The inherited normalized eight-dimensional addresses remain rank-limited;
clipped writes do not establish eight-item universal capacity or reliable
retrieval. Bounded state norms do not imply useful information survival. The
source upper direct retention is capped below0.995, while other memory paths
have different retention and interference. No uniform state bound for every
component or full-model stability theorem is inferred from the masked-loss
proofs. Changing the objective does not by itself prove procedural discovery.
For the isolated native matrix paths, the source's clipped fast write gives
`||F'||F <= .99*||F||F+.35`, hence a zero-initialized norm envelope35. The slow
delta step has `rho<=.999`, `0<=eta<=.01`, unit-bounded writer/value and
`||I-eta*k*k^T||2<=1`, yielding the loose zero-initialized envelope10.
These are derived component bounds under the normalization/source assumptions,
not new kernel-checked097 full-model theorems. The aligned matrix's independently
gated forgetting/write cannot borrow the LANGMATH095 coupled-write envelope.

TAC/GRU/SSM source recurrence cost grows linearly in observed sequence length
at fixed geometry; the actual Python implementation may incur substantial loop
and solve overhead. Transformer prefill attention has quadratic context work
and linear growing KV state; cached generation reads the retained context.
Both acquisition forward/backward and query/generation/verification must be
charged. Equal bytes, equal supervised targets and equal training FLOPs are
distinct quantities. Report matched regimes honestly rather than equating them.

`probe_adapter.py` registers all four actual graphs. Local acquired floating
state derivatives are measured; categorical byte/clock derivatives and candidate
permutation semantics are not declared. Growing-cache shape differences have
unknown delta norms and remain `INCONCLUSIVE_SHAPE_CHANGED`. No adapter fixture
can establish source authenticity, training success or real capability.

The PNDS companion `PNDS/LANGACQUIRE097.lean` defines serial unroll, categorical
NLL and masked objectives, with scoped chunk/selection/chain-rule theorems.
`foundation/v1/langacquire097_reference.py` independently checks objective
gradients, chronological chunking and a shared-parameter scalar acquisition
counterexample. It is explicitly not a reconstruction of any complete model.
The full typed source graph is recorded in `LANGACQUIRE097_GRAPH.md`; complete
IEEE PyTorch/source AD equivalence, full tensor derivative proofs, autonomous
verification, continual improvement and general intelligence remain OPEN.
