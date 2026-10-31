# LANGMATH098: contextual, normalized dual associative TAC

Untrained architecture revision following LANGACQUIRE097's failed capability
gates. No old weights/scores are changed. This preserves native associative
key/value banks, rather than substituting an SSM as the whole model. A small
diagonal context process supplies contextual writes. The nonlinear fast delta
correction, positional alignment banks, waves, concept solve and upper feedback
are intentionally replaced in this candidate. The unchanged086 is the control.
This is a substantive changed graph, not a frozen-checkpoint optimization.

## Full causal graph and types

Observed bytes X:Long[B,T], alphabet256; unique owner strings; clock>=0.
No targets, task labels, loss masks or gold lengths enter forward. All floating
source state is FP32/FP64, same device/precision as model. No mixed precision
claim is made before separate correspondence checks.

Persistent state per entity (432 floats,1728B FP32):

| Component | Shape | Meaning |
| --- | --- | --- |
| context | [48] | multiscale observed-context summary |
| lexical | [3,32] | three previous observed lexical embeddings, oldest first |
| previous_key | [2,8] | last observed context's two learned write addresses |
| memory | [2,16,8] | two native value/address matrices |
| mass | [2,8] | matching positive address mass |

1. `e=tanh(Embedding(X))`, width32. `local=tanh(e+sum_l tanh(lag_l)*past_e_l)`.
   Lexical lag features use only previous observed bytes and current byte.
2. Context offer `u=tanh(Wc*local+bc)`, width48, and
   `c_t=delta*c_(t-1)+(1-delta)*u_t`, `delta=sigmoid(-raw_decay)`.
   Initialization spans time constants8–4096; these are design choices, not
   empirically optimized horizons. The context read is **c_(t-1)**.
3. Concatenate `[e32,local32,c_previous48]` into112 features. One fused affine
   map produces two8D write-key logits, two8D read-query logits, two16D value
   logits, and two scalar gate logits. Keys/queries use softmax; values use
   tanh. There is no fixed byte-code constraint or inverse-norm address division.
   Fast alpha is in[.97,.995], slow alpha in[.995,.99995], up to FP rounding.
4. Write previous contextual key with current contextual value, then read with
   current query. Cold first write is disabled. For bank h:

   `M_t=a_t*M_(t-1)+(1-a_t)*v_t*k_(t-1)^T`

   `z_t=a_t*z_(t-1)+(1-a_t)*k_(t-1)`

   A false legal write mask sets `(a,b)=(1,0)` for this bank; it does not remove
   the observation from lexical/context state. The mask is a write entitlement,
   not the training objective mask. The default admits every observed write.
5. `evidence_h=z_h dot q_h`,
   `read_h=M_h*q_h/(evidence_h+1e-4)`, width16 each.
6. A16D sigmoid comparison gate consumes
   `[local32,c_previous48,read_fast16,read_slow16,abs(read_fast-read_slow)16,evidence2]`.
   It produces a mixed read. Both original reads are ALSO passed separately
   to the decoder, so comparison does not erase the read pair.
7. Concatenate `[e32,local32,c_previous48,fast16,slow16,mixed16,evidence2]`.
   LayerNorm162 (epsilon1e-5); gated FFN width138:
   `logits=Wo((Wu*z+bu)*silu(Wg*z+bg))+bo`, alphabet256.

There are100370 active/deployed learned parameters and16B of fixed buffers.
All input/state/decoder/gate parameters count. No dummy, inactive or frozen
pretrained parameter allowance is used. State ownership/shape/precision/time
are validated; validity theorems concern zero/reachable states, not arbitrary
externally injected finite matrices that violate the memory cone.

## Invariants and numerical conditioning

The invariant is `z_j>=0` and `|M_ij|<=z_j`, initially true at zero. With
`0<=a<=1`, `k_j>=0`, `|v_i|<=1`, the actual coordinate update preserves it.
If `sum z<=1` and `sum k<=1`, total mass also stays<=1. Positive queries give
`|sum M_ij*q_j|<=sum z_j*q_j`, so normalized read magnitude is<=1 for every
positive epsilon. The read denominator never divides by zero; derivative
bounds still depend on epsilon. Mass normalization avoids simply attenuating
rare retained values with `(1-alpha)` writes, but does not guarantee useful
retrieval: conflicting writes can still average and small evidence suppresses
reads. Underflow/rounding and epsilon bias remain measured numerical concerns.

Context/lexical values stay in[-1,1]. Previous keys are zero at cold start and
subsequently positive simplex vectors. Each valid persistent tensor is bounded.
The comparison gate is convex, so its mixed read is bounded as well. Decoder
and parameter-gradient conditioning is distinct from this state invariant.

The Lean file proves the full persistent tensor invariant over arbitrary observation
streams, activation-derived encoder bounds, finite tensor affine composition/action, zero/nonwrite,
matrix coordinate cone preservation, mass bound, normalized read bound,
activation ranges, read-pair exposure and halving-scan work. No assumed matrix
norm inequality replaces the actual update proof. The Python reference covers
the complete encoder/recurrence/read/comparison/decoder graph. Full equivalence
of PyTorch's IEEE kernels and automatic differentiation to real-number Lean
semantics is **OPEN**, as is learned useful retention and capability.

## Efficient chronological execution

Both context and banks use `s'=a*s+b`. Composition of first(a1,b1) then(a2,b2)
is `(a2*a1,a2*b1+b2)` coordinatewise, associative in exact arithmetic. Context
is scanned first; contextual addresses/gates are then computed in parallel;
the two banks are scanned next. This is a **two-stage** algorithm, not an
assertion that the combined nonlinear context-to-address graph is one affine
operator. No bank state is fed back into its own key/value/gate coefficients.

The implementation reduces adjacent pairs, recursively scans them, and
reconstructs even states from preceding odd states. State-width work obeys
`W(T)=T+W(floor(T/2)) <=2T`; depth is logarithmic. It does not use an
O(T log T) all-prefix-doubling scan or unstable division by cumulative products.
Streaming step: fixed state/parameter geometry, O(1) history work and memory.
Prefill/training: O(T) work and O(log T) temporal span, with O(T) state-width
activation storage, not O(T^2) pairwise context attention. Two heads process in
parallel. Actual hardware speed, kernel-launch/allocation overhead and gradient
cost require measurements; no globally fastest architecture claim follows.

Prior-art reuse: normalized positive-feature associative retrieval follows
[Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html),
with contextual feature generation and separately retained fast/slow banks.
Pair-reduction scan is a standard associative-prefix technique. These kernels
are reused mathematical tools, not a claim of novel general intelligence.
The simpler convex update trades delta-error correction for an exactly
scannable bounded update; it may lose useful overwrite behavior. Softmax keys
also restrict signed addressing. These tradeoffs require controlled learning
experiments rather than assuming this candidate is better.

## Controlled objective, tests and gates

`objective.py` fixes each domain weight to1/2 and each case's future mean to
coefficient1, then adds `lambda*prefix_mean` per case. Both domains and nonempty
future/prefix in each case are required. Varying lambda changes only explicit
prefix coefficients; it does NOT guarantee invariant effective update shares
after joint gradient clipping and AdamW. No training is enabled here.

Constructed checks compare source full logits/all state against an independent
NumPy serial implementation, Torch serial versus parallel, chunk continuation,
all-parameter gradients, acquired-state AD/finite differences, causal future
poison, entity isolation, ownership/precision, state invariants, write identity,
context-to-address/value connectivity and read-to-comparison connectivity.
FP64 forward tolerances2e-11, FP32 tolerances2e-5; gradients1e-10 absolute and
1e-8 relative in FP64, local finite differences1e-7/1e-4.

No original data, protected TEST, optimizer steps, pretrained-model predictions,
new training or capability scores are authorized by this no-fit record. A new
fit needs full authentic-source FAIR_COMPARISON preregistration, fresh splits,
all controls, numerical diagnostics and the usual real research gates.
