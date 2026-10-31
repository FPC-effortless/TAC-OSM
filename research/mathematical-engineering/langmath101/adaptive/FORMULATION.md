# LANGMATH101 optional A5: causal committed representations

This is a new graph candidate informed by SENSE-TOK004/005/006. It does not
replace their original source, results, no-fit registrations or mathematical
claims. No optimizer or dataset access is implemented here. The inherited
100K rule is met by live objects; full training/data/compute gates are separate.

| Arm | Trainable/deployed parameters | Recurrent width |
|---|---:|---:|
| a5_byte | 100520 | 130 |
| a5_fixed_bpe | 100201 | 129 |
| a5_discrete | 99917 | 127 |
| a5_latent | 100632 | 128 |
| a5_adaptive | 100448 | 127 |

All counted weights are on useful reachable paths. Natural controller-path
activity must still pass the actual original FIT gradient/profile audit; forcing
a route in a unit test does not prove that the natural model visits it. The
pairwise maximum gap is below1%, including native100370 reference geometry.

## Inputs, output and state

The only observed input is Long[B,T] byte IDs0..255. Output is [B,T,256]
logits: after observing byte x_t, predict x_(t+1). There is no unobserved EOS.
State ownership is explicit. Native state holds hidden h and up to3 pending
byte embeddings/source bytes and their proposed token kind. Pending source
bytes are audited input metadata, not decoder features. The model-owned
last_receipt contains counters only and does not enter subsequent prediction.

Both fixed patch modes commit exactly4 observed bytes. Before commitment their
decoder state is unchanged, so they issue identical distributions for some
successive positions. This is a normalized causal byte model, but could be a
poor one. It is not a conventional token LM that conditions within-token byte
readouts on raw partial token content.

## Selected representation equations

Byte embeddings e=E[x] have24 coordinates. For a nonempty observed pending
patch e_1..e_n, n<=4, concatenate them with zero padding to width96:

    z=tanh(W_pool [e_1;...;e_n;0;...;0]+b_pool).

The zero padding has no learned unused embedding row. BYTE uses r=e_t.
LATENT uses r=z. DISCRETE uses16 learned code vectors C_k:

    q_k=softmax_k(-||z-C_k||²), k*=argmax_k q_k
    r=C_(k*) (forward)
    weights=onehot(k*)+(q-stop_gradient(q)) (backward surrogate).

One-byte residual patches fall back to their exact byte embedding. A committed
four-byte discrete patch is lossy in general:16 codes cannot preserve all
possible four-byte strings. Exact spelling and rare strings must be measured;
no lossless codebook, semantic token discovery or compression claim is made.

Recurrent update is the standard GRUCell equation, and output is only

    h'=GRU(r,h), logits=W_head h'+b_head.

The head NEVER receives the current byte, pending patch, code probabilities or
the unquantized pooled vector as an additional input. Code interventions hold
the selected code fixed while changing raw source bytes and embeddings; the
resulting forced-discrete logits remain bit-identical. Replacing the selected
code changes logits. This closes that specific byte-head bypass, not every
possible learned shortcut or an externally certified information boundary.

## Adaptive controller

The three-way controller sees [h,e_t,pool(pending+e_t),entropy(head(h))]. Every
quantity is available after x_t and before x_(t+1). It chooses BYTE, DISCRETE
or LATENT. BYTE flushes earlier pending content in its last selected token kind
then updates with e_t. A token choice buffers data and commits at span4. The
selected token kind is finalized from legally observed data at commitment.

Sparse inference executes only selected GRU calls. Surrogate training computes
the BYTE candidate (including a possible pending flush) and, at full span, both
token candidates. Hard forward selection uses

    w=onehot(argmax a)+(softmax(a)-stop_gradient(softmax(a))).

This is a biased straight-through estimator, not exact differentiation through
the argmax, and not evidence of training-compute reduction. Candidate work and
selected updates are reported separately. Actual adaptive work depends on
learned choices/pending lengths and cannot be declared constant before fit.

## Fixed conventional BPE control and hidden-prefix cache

FrozenBPE.fit consumes only explicit FIT bytes, records their content SHA256s
and fits32 deterministic ranked byte-pair merges. Its256 byte IDs,32 merged
IDs and one observed-start BOS embedding give289 embedding rows. Tokenization
has exact byte roundtrip, including unseen strings through byte fallback.

Because token boundaries may revise with additional observed bytes, the model
retokenizes the entire observed prefix at each byte readout. It finds the common
prefix with the previously encoded token sequence, reuses hidden states after
that prefix, and recomputes the changed suffix. This is pointwise equivalent
to the independent naive full-prefix reference with fixed model parameters.

State explicitly includes raw observed history, cached token IDs, and a packed
per-token hidden-state tensor. This is O(history) memory, not a fixed state
model or Transformer attention/KV cache. A guard of model parameter identity,
ordinary PyTorch version, dtype/device and tokenizer identity invalidates the
cache when weights, model instance or tokenizer change, then rebuilds from raw
history. Parameters created in inference_mode lack version counters: they
disable cross-call cache reuse. Adversarial .data/raw-storage tampering remains
OPEN. Owner/clock/cache shape and finiteness are checked.

## Verification and remaining obligations

42 constructed no-fit checks pass: size, future poisoning, owner isolation,
whole/chunk/serial correspondence, no raw-byte readout bypass, sparse versus
surrogate forward equality, BPE FIT-only rejection/roundtrip/revision/cache
invalidation, and independent manual-GRU output/state/all-parameter gradients.
The independent functional reference never calls the candidate or module
forward. FP64 correspondence tolerances are1e-12 outputs/state and1e-10
absolute/1e-9 relative parameter gradients. Ordinary FP32 cache inputs are
cloned from packed slices to preserve source-aligned single-state kernel layout.

Whole-model IEEE/AD proof, PNDS full graph formalization, real FIT natural-route
gradient activity, complete training FLOPs/data/tuning parity, downstream
quality, useful token reuse, energy, speed and independent protected
qualification remain separate OPEN obligations. No new model was trained.

Constructed B4×272 random-source forward+CE-backward (two CPU threads) ran in
roughly0.16–1.2s per model with cumulative process peakRSS364MB. The cached BPE
case used567072 bytes of tensor state versus3616 for BYTE in that fixture.
These figures are engineering smoke measurements, not authentic-source
capability, a controlled latency comparison or a guarantee on real FIT inputs.

The initial cache QA exposed two defects: packed-offset FP32 kernel layout
differences and missing version counters on inference-created parameters.
Both were fixed; no optimizer, capability threshold or source data changed.
