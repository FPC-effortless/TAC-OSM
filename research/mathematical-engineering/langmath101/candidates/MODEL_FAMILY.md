# Compact LANGMATH101 candidate family

These are source implementations for prospective internal FIT/DEV experiments,
not trained results. No optimizer or dataset scoring ran in this directory.
`CANDIDATE_MANIFEST.json` records actual instantiated parameter counts/state
fields. `factorized098` is an interface alias of `native098`; it must not be
counted as a second independent neural fit or new architecture.

## API and boundaries

```python
model = make_model(name)
state = model.start(('explicit-owner',))
logits, state = model(tokens, state=state, owners=('explicit-owner',))
```

Only observed byte IDs0..255 enter the model. The output is256 next-byte logits
at every supplied position. No target, loss mask, judge, future outcome or
candidate answer list is accepted as a forward argument. Stream owners are
metadata, not embeddings. FP32/FP64 only. State schemas are declared rather than
guessed from another architecture's matrix geometry. New generic State has
`owners`, `clock`, `tensors`; conventional states additionally have `arm`.
The model's administrative `arm` is assigned by the factory.

Stateless candidates return clock0 and no tensors. They operate solely on the
provided current-input byte or four-byte causal window. Chunking a window at a
boundary without supplying its earlier bytes intentionally changes information
access; chunk invariance is not claimed for them. Stateful models preserve
clock and all declared tensor fields across calls. The Transformer baseline
stores and reprocesses the observed raw prefix; its memory grows with history,
and it does not claim to be KV cached.

The generic intervention helper accepts exact fields or layer prefixes, plus
`fast/slow/banks` for native098 and `<layer>.fast/slow/banks` for new stacks.
Matrix and address mass are reset/replaced together. Previous keys are separate
write-link state and must be explicitly selected. Wrong-state donors require
matching clock/schema/shapes; the original owner's metadata is retained.

## Exact native reference and A2

`native098.py` is byte-for-byte the original ContextualAssociativeTAC098 source,
SHA256 `e59c0aec80a39d84a6f01a7ca662d0c8baf3e2c2d33901057f017cb23c999498`.
Its100,370 learned parameters and432 FP32 state values are unchanged. A2's
factorization does not require new weights because the native tensor fields
already expose recurrent context, lexical cache, previous keys, memory and mass.

## Stateless A1

Let `e_t=tanh(E[x_t])`, with learned byte embedding width32. For window `w=1`
or `w=4`, concatenate `[e_t,e_(t-1),...,e_(t-w+1)]` from the supplied sequence;
missing leading positions are fixed zero vectors. No vector is carried to the
next invocation. Apply the gated decoder below. The window4 model is a bounded
direct-context control, not a recurrent state disguised as statelessness.

## Residual contextual associative A3 block

Common hidden width32, context width16, two banks with key width4 and value
width8. For a current layer input `h_t`, form:

1. `a_t=sigmoid(-raw_decay)` in ordinary stacks.
2. `c_t=a_t*c_(t-1)+(1-a_t)*tanh(W_c h_t+b_c)`.
3. From `[h_t,c_(t-1)]`, one affine projection produces two keys, two queries,
   two values and two retention logits. Keys/queries use separate softmaxes;
   values use tanh. Retention lies in `[.97,.995]` and `[.995,.99995]`.
4. At the first-ever token no bank write occurs. Otherwise, with previous key
   `k_(t-1)`, update matrix and address mass together:
   `M_t=alpha_t*M_(t-1)+(1-alpha_t)*v_t*k_(t-1)^T`,
   `z_t=alpha_t*z_(t-1)+(1-alpha_t)*k_(t-1)`.
5. `read_t=(M_t q_t)/(z_t dot q_t+1e-4)` for each bank. A sigmoid gate sees
   `[h_t,c_(t-1),read_fast,read_slow,abs(read_fast-read_slow),evidence]`.
   Form the convex mixed read from that gate.
6. Concatenate `[h_t,c_(t-1),read_fast,read_slow,mixed,evidence]`, LayerNorm it,
   and return `h_t+tanh(W_out normalized_features+b_out)`.

Context is one affine scan; bank evolution is a second affine scan whose
coefficients are determined after the context scan. Both support serial and
work-efficient pair-reduction/reconstruction execution. The input stem has the
original three-lag learned linear mixture followed by tanh and a96-value recent
embedding cache. Repeated blocks use independent parameters and state.

## Non-associative temporal A3 controls

The same32D hidden/16D context recurrence is retained, but no key, matrix, mass
or associative read exists. Instead:

`u_t=LayerNorm([h_t,c_(t-1)])`

`h'_t=h_t+W_out((W_up u_t+b_up)*SiLU(W_gate u_t+b_gate))+b_out`.

Each block has a real width36 gated FFN. The associative and recurrent control
blocks have nearly equal parameter counts without adding unused coordinates.
Depths1,2,4,8 are requested principal arms; depth3 paired controls support a
memory×hierarchy factorial against the three-level hierarchy.

## Input-gated temporal hierarchy A4

Three bottom-up levels use either the associative or recurrent block above.
Between levels, `h_(l+1,input)=tanh(U_l h_(l,output)+b_l)`. Each level learns an
input-dependent update rate `r_t=sigmoid(W_update h_t+b_update)`, and changes its
context retention to `1-r_t*(1-a_t)`. Associative bank retention is similarly
`1-r_t*(1-alpha_t)`. Initial context horizon scales are multiplied by `4^l`;
update-weight initialization is zero with bias `-l`. These temporal priors are
explicit design choices, not evidence of discovered human concepts.

After the upward pass, let `y` start at the highest-level output. Descend via
`y=h_(l,output)+tanh(D_l y+b_l)`. This learned downward pathway affects the
current readout. It does **not** feed back into the already-updated recurrent
state, so the implementation must not be described as prediction-conditioned
top-down state correction. Higher-level abstraction, invariance, causal utility
and generalization remain empirical questions.

## Common active capacity allocation

For a final feature dimension `d`, the output head is:

`u=LayerNorm(features)`

`logits=W_o((W_u u+b_u)*SiLU(W_g u+b_g))+b_o`.

Choose its positive hidden width by rounding the remaining parameter budget
to the nearest real decoder width. Every counted learned parameter lies on the
prediction pathway; there is no dummy/frozen padding. All nineteen exposed
names are100,196–100,518 parameters (about0.321% maximum relative gap), including
the A2 alias. Decoder width necessarily changes with depth, so these are
capacity-allocation comparisons, not pure depth effects with a fixed decoder.
Parameter matching does not imply identical compute. Counts and actual
training/resource/exposure preflight must be audited separately before fitting.

Conventional GRU, selective diagonal recurrence and causal Transformer exact
equations/reference tests are in `CONVENTIONAL_FORMULATION.md` and
`conventional_reference.py`.

## Tests and mathematical status

Constructed checks cover every parameter tensor receiving a finite nonzero
gradient, native source identity, actual counts, state ownership and causality,
serial/scan/chunk outputs/state, explicit intervention selection, stateless
information boundaries, independent conventional NumPy correspondence and one
central finite-difference coordinate per conventional parameter tensor.
These checks do not establish useful language/memory/reasoning capability or
prove every floating-point/AD operation. Independent new-family reference and
Lean definitions/refinement are maintained separately; all uncompleted formal
obligations remain OPEN. No training or mathematical release is claimed here.
