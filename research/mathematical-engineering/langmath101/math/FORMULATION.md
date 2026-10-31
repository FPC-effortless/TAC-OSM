# LANGMATH101 source-grounded mathematical formulation

This record specifies the **actual candidate source graphs**, not a validated
theory of intelligence. The source files live in `langmath101_candidates` and
`langmath101_adaptive`. Exact file SHA256 values are in SOURCE_MAP.json. A TAC
commit and reciprocal PNDS commit must be pinned by the publisher; this local
draft is not a substitute for those immutable remote records.

## Typed inputs, time, state and outputs

Every model receives observed `Long[B,T]` byte IDs in0..255 and produces
`Float[B,T,256]` unnormalized next-byte logits. Position t predicts the byte
after the observed byte at t. Targets, the evaluator, true next bytes, task
strata and future outcomes are not forward inputs. Teacher-forced training may
use already observed earlier answer bytes, but free-running task success is a
separate score. Padding is not an observed event: losses must exclude it, and
padding-derived final states must not be carried into a later query.

Owners are unique nonempty strings checked at API boundaries. Owner strings do
not enter neural arithmetic; owner binding is procedural isolation, not an ACL
or authentication theorem. Clock is an integer count of observed bytes and is
preserved in reset/donor interventions unless the control explicitly starts a
fresh invocation. Kernel equations below describe one stream; source batches
are independent replicas. States contain unitless learned values, except raw
token IDs and discrete metadata. Returned state is explicit and may be copied
or serialized; no useful memory or state sufficiency follows from this API.

The exact-real Lean graph is `PNDS.LANGMATH101`; actual FP32/FP64 tensors,
source validation, error paths and automatic differentiation are checked by
executable tests and remain separate from a universal refinement proof.

## Shared primitives

For a learned affine map, `L(x)=Wx+b`. LayerNorm uses population variance,
learned gamma/beta and epsilon1e-5. Define `sigma(x)=1/(1+exp(-x))`,
`tanh(x)=2sigma(2x)-1`, `silu(x)=x sigma(x)`. Every BudgetHead is

`z=LN(x); logits=W_out[(W_up z+b_up) ⊙ silu(W_gate z+b_gate)]+b_out`.

No head concatenates a label or externally inferred answer. Widths are active
readout capacity, not unconnected parameter padding. Their actual counts are
measured from instantiated source models in CANDIDATE_MANIFEST.json. Native
buffers are copied with their actual source dtype: the native bank constants
were registered as FP32, so converting the model to FP64 does not replace them
with exact FP64 decimal literals. This distinction was found by the reference
test and corrected without changing its tolerances.

## A0/A2 native098

The exact source has d32,c48,k8,v16 and FFN138. Its learned parameter count is
100370; native tensor state is432 floats/stream. A2 `factorized098` is an
explicit **source-identical alias**, not a second architectural mechanism.
Factorized interventions expose fields of the same graph. A separate new fit
using the alias must not be represented as independently different architecture.

For byte b_t, `e_t=tanh(E[b_t])`. Store the three preceding embedding vectors
in chronological order. The local input is

`l_t=tanh(e_t+Σ_(j=0..2) tanh(w_j)⊙e_(t-1-j))`.

`delta=sigma(-raw_decay)` and
`c_t=delta⊙c_(t-1)+(1-delta)⊙tanh(W_context l_t+b)`.
Keys, queries, values and retention are generated from
`[e_t,l_t,c_(t-1)]`, not from a future byte. Two banks have independent
softmax8 keys/queries, tanh16 values, and retention in the declared fast/slow
intervals. The matrix and address-mass update for each bank is

`M_t=alpha_t M_(t-1)+(1-alpha_t) v_t k_(t-1)^T`,
`mass_t=alpha_t mass_(t-1)+(1-alpha_t)k_(t-1)`.

The first observation cannot make a lag-key binding; alpha=1 there. A false
bank-write mask also sets alpha=1, but context, lexical history and previous
keys still change. The read is

`r_t=(M_t q_t)/(mass_t·q_t+1e-4)`.

The comparison gate sees `[l_t,c_(t-1),fast,slow,abs(fast-slow),evidence]`.
It produces a per-coordinate sigmoid mixture, while both original reads also
remain exposed. The terminal head input is
`[e_t,l_t,c_(t-1),fast,slow,mixed,evidence]` (162 coordinates).

The executable reference serializes both affine recurrences and uses no source
`affine_scan`. The source uses a work-efficient pair-reduction scan. Their
finite-precision correspondence is tested, not assumed to be bitwise at all
lengths/precisions.

## A1 current/window4

These are independent parameterized gated MLPs, not native098 with unused
memory weights counted toward a budget. They have no persistent tensor state,
and reject nonzero carried clocks. Current-only uses `tanh(E[b_t])`. Window4
concatenates the current embedding and up to three preceding embeddings from
the **same invocation**, with zero vectors before that invocation began.
No acquisition bytes are available across a hidden-history boundary. A
four-byte observation window is not a full-context Transformer.

## A3 stacked contextual blocks

The common stem is the 32D three-lag local process above. A repeated block
receives its preceding layer's h_t, not the original byte embedding again.
Each block has c16,k4,v8 and its own source state. Its context update uses
`tanh(W_c h_t+b)` and diagonal learned decay. Address generation uses
`[h_t,c_(t-1)]`. Its two banks obey the same normalized lag-binding law.
Comparison uses `[h_t,c_(t-1),fast,slow,abs(fast-slow),evidence]`.
With z formed by replacing abs-difference with mixed read, the residual is

`h'_t=h_t+tanh(W_out LN(z)+b_out)`.

Nominal depths are1,2,3,4,8; the extra depth3 enables a like-depth hierarchy
control. The final learned BudgetHead consumes h at the last layer.
Associative stack state contains96 lexical floats plus96 floats/layer.

The non-associative controls retain the same c16 affine context process but
replace the banks with a real gated FFN:
`h'=h+W_out[(W_up LN([h,c_prev])+b_up)⊙silu(W_gate LN([h,c_prev])+b_gate)]+b_out`.
Their stored state is96 lexical floats plus16 floats/layer. Parameter-matched
width allocation changes active decoder capacity; parameter parity alone does
not imply equal FLOPs, exposures, model expressivity or state bits.

## A4 three-level hierarchy

Each level additionally learns `u_t=sigma(W_update h_t+b_update)`.
The effective context retention is `1-u_t(1-delta)`, and effective bank
retention is `1-u_t(1-alpha)`. Initial horizons are multiplied by4^level;
initial update biases are -level. These are **declared temporal priors**,
not discovered concepts or a guaranteed learned timescale ordering.

Learned tanh upward projections precede levels after the first. After computing
all levels, downward readout is `h=level_i+tanh(W_down_i h+b)` from top to bottom.
**Downward projections affect output representations, not next temporal-state
updates.** This is a readout hierarchy, not full recurrent top-down predictive
feedback. A claim about the stronger feedback hypothesis remains OPEN.

There are matched-depth associative and non-associative hierarchy candidates.
Functionality must be tested by held-out outcomes and path interventions;
clustering a state or naming a layer does not establish abstraction.

## Conventional controls

GRU: tanh32 byte embeddings, a conventional64D PyTorch GRU with r/z/n gates,
then a gated head. The reference evaluates the gate equations directly,
including `n=tanh(input_n+r⊙hidden_n)` and `h=(1-z)⊙n+z⊙h_prev`.

Selective diagonal SSM: tanh32 embedding, `a=.999 sigma(W_a e+b_a)`,
`s'=a⊙s+(1-a)⊙tanh(W_b e+b_b)` and
`h=tanh(W_s s'+b_s+W_e e+b_e)`. It is explicitly a simple affine control,
not a reproduction of Mamba. The reference evaluates a serial recurrence.

Transformer: one pre-norm causal self-attention block, embedding32, four
heads of8 dimensions, sinusoidal positions, no dropout, then residual GELU
FFN32→64→32 and gated output head. State stores the complete observed raw byte
IDs and recomputes the full prefix; this is **not a KV-cached implementation**.
Attention is `softmax(QK^T/sqrt(8)+causal_mask)V`; the independent functional
reference explicitly forms Q/K/V, masked scores and weighted values rather
than calling `MultiheadAttention` or its source forward method. A separate
NumPy reference and coordinate finite differences are supplied by the
conventional-control implementation. Native recurrence and attention are not
equivalent merely because both predict bytes.

## A5 byte/discrete/latent/controller/BPE

The adaptive source uses embedding24 and hidden sizes130/127/128/127/129
for byte/discrete/latent/controller/BPE. The learned readout receives **only
hidden state**. It does not concatenate raw bytes, local byte features, pending
patches or token IDs. This closes the explicit raw-byte readout bypass, while
allowing the declared BYTE branch to process bytes normally.

Fixed patch span is4. Pending observed embeddings are padded to4 and mapped
through a learned96→24 affine+tanh pool. A latent patch supplies that pool.
A discrete patch selects one of16 learned24D vectors by minimum squared
distance to the pool. Forward uses a hard code identity; training uses the
declared straight-through soft assignment. One-item flushes use the byte
embedding directly. Incomplete patches leave h and therefore logits unchanged
until commitment, which may harm prediction quality.

The adaptive three-way controller sees
`[h_prev,current_embedding,pooled_pending,current_predictive_entropy/log(256)]`.
The real implementation uses the recorded decimal entropy normalization.
Routes are BYTE, DISCRETE, LATENT. BYTE flushes any older pending patch through
the GRU before advancing the current byte. Token routes buffer up to4 observed
bytes and commit the chosen representation. The training surrogate evaluates
extra candidate GRU calls; sparse inference executes chosen calls. Every call
must be counted. A reduction in committed updates is not a FLOP/speed result.

Straight-through derivatives are not mathematical derivatives of argmax. The
Lean definitions cover primal forward behavior; source/autograd equivalence is
checked by the independent functional reference, not asserted as a theorem.
Actual FIT profiles must check natural route coverage and gradient activity.
Forced-discrete fixtures do not prove that codebook weights are active under
the natural learned controller on training data.

BPE uses exactly32 ranked merges fitted solely to authenticated FIT bytes,
plus byte IDs and BOS288. It retokenizes the whole observed prefix. An exact
common-token-prefix hidden cache avoids recomputing unchanged GRU prefixes;
parameter object/version/dtype/device and tokenizer guards invalidate/rebuild
the cache after updates or reload. This consumes raw O(history) bytes,
O(tokens×hidden) hidden-cache floats and token metadata. No constant-memory or
tamper-proof KV claim is made. The independent reference naively recomputes
each full prefix, so cache behavior is tested against the uncached graph.

## Mathematical analysis and proof boundaries

For nonnegative keys with unit mass, |value_i|≤1 and0≤alpha≤1, the normalized
bank invariant is `mass_j≥0` and `|M_ij|≤mass_j`. It yields `|read_i|≤1` for
nonnegative queries and positive epsilon. The generic real-number bank theorem
already proved for native098 applies to this algebraic update; an explicit
mapping and source/reference evidence is still required for every new block.
Bounded bank magnitude does not prove retention, separation, calibration or
state sufficiency. Colliding addresses and overwrites remain possible.

Context is a convex bounded update because effective retention
`1-u(1-delta)` lies in[0,1]. Associative residual outputs can grow with depth;
they are not globally bounded to[-1,1]. Recurrent FFN residuals need additional
weight/norm assumptions for a uniform stability theorem. No global contraction
or optimality theorem is asserted. The A4 ordering priors can change in training.

A fixed S-float FP32 state has at most2^(32S) bit patterns (fewer under finite
validation), so it cannot injectively preserve every unrestricted256-symbol
history longer than4S bytes at a fixed length/clock. This is an information
capacity bound, not a useful-task recall bound. Transformer/BPE explicit raw
history has different storage growth and must be counted.

Native and per-block affine scans have linear state-width work and logarithmic
ideal temporal span. This does not eliminate dense/head work or real kernel
overhead. Prefix-recompute attention can be O(T²) per growing invocation; BPE
retokenization/cache maintenance has separate costs. Exact active parameter
counts are not a hardware performance or equal-training-compute proof.

Lean currently contains actual typed forward definitions and four narrow
structural results: deterministic fold chunking, terminal-head independence
from native state updates, fixed-parameter GRU prefix-cache equivalence, and
hidden-only A5 readout noninterference. These do **not** prove complete
IEEE/AD/source equivalence or learned memory usefulness. Local Lean compilation
is NOT RUN successfully because the installed launcher fails even `--version`
with `failed to locate application`; a PNDS CI kernel build remains required.
No sorry/admit or new unreviewed axiom is counted as proved.

## Executed numerical verification

`test_reference.py`:19 tests PASS, comparing actual source logits, every state
field and all parameter gradients to the independent functional graph in FP64.
Tolerance: outputs/states atol1e-10,rtol1e-9; gradients atol1e-10,rtol1e-8.
The native buffer precision issue was fixed in the reference, not tolerated.
The adaptive source supplies its separate five-mode functional reference and
42 executed implementation/reference tests. Source snapshots/hashes must be
checked again by the final runner after the merged source is frozen.

OPEN: Lean kernel compilation/axiom audit, all-source finite-precision proof,
complete decoder/readout stability and gradients for all inputs, external
owner/ACL authenticity, protected independent replication, learned sufficiency,
retention/transfer/composition, hierarchy and scientific capability gates.
