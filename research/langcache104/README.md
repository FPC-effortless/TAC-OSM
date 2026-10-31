# LANGCACHE104 — deep, multi-perspective carried-state cache

Status: **architecture implementation and no-fit engineering study**. No weights
have been fitted, no DEV/TEST capability score is reported, and no GPU speedup,
biological equivalence or architecture-superiority claim is made. This is a new
candidate, not a change to an existing trained checkpoint. Its source lineage is
LANGMATH098 → LANGCAP103 at
`99a1cb45ef65581f9c97dbb761d6910474969209` (PR #681).

## One integrated question

Can a genuinely deep language model combine selective associative compression,
signed state dynamics and addressable recent observations to improve useful
prediction and memory-dependent completion per unit of measured resource?

The candidate treats carried state as a **hybrid cache**: learned summaries in
fixed associative banks, a separate dynamic state, and actual recent K/V records.
An explicit-key byte store is supplied as a separate runtime component. It is
not a learned language-to-key mechanism and does not secretly feed answers into
the neural logits. Every component must be counted in any later end-to-end
comparison. No existing compression result establishes this candidate's quality.

## Computational graph

Observed bytes → embedding → L residual blocks → RMSNorm → byte logits.
Each block uses its own parameters and its own stream-owned state:

1. Pre-normalize the residual representation.
2. Project independent learned queries, rank-R keys/values, retention and write
   eligibility for each head in separate fast/slow associative banks.
3. Read both banks, preserving their separate outputs and evidence mass.
4. Update a separate signed, rotating field; attend to a causal recent K/V window.
5. Project the concatenated fast read, slow read, difference, elementwise product,
   dynamic field, recent attention and mass features into the residual stream.
6. Apply a second pre-normalized SwiGLU residual transformation.

For layer ℓ, residual multiplier γ = 1/√(2L):

    u = RMSNorm(x)
    h = x + γ W_o concat(r_fast, r_slow, r_fast−r_slow,
                         r_fast⊙r_slow, f, local_attention, evidence)
    x_next = h + γ W_2(SiLU(W_g RMSNorm(h)) ⊙ W_1 RMSNorm(h))

These are opportunities to learn comparisons, nonlinear representations and
composition; they do not guarantee heads will specialize or layers will discover
abstractions. Depth and width must be validated separately. The residual scale
is a conditioning heuristic, not a proof of stable optimization at arbitrary L.

Unlike LANGMATH098's previous-key → current-value update, this candidate writes
same-step projected keys and values. First-layer bank writes see the current byte
embedding; deeper-layer writes can use context supplied by lower layers. This
is a changed prediction architecture, not source-equivalent reuse of a trained
098 bank. It needs direct lexical-association and order-sensitive diagnostics.

Capacity controls are distinct: residual width d, depth L, head count H, key width
K, value width V, write rank R, FFN width F, local window W and chunk length C.
Increasing only d or the output head does not increase associative address space.
Rank-R writes increase instantaneous representational options but may be
redundant; they do not provide R times as many independent stored facts.

## Positive associative cache and its scoped bound

For each head and bank, M ∈ R^(V×K), z ∈ R^K:

    q_t = softmax(W_q u_t)
    k_tr = softmax(W_kr u_t), v_tr = tanh(W_vr u_t)
    a_t = sigmoid(W_a u_t + b_a)
    w_t = (1−a_t) sigmoid(W_e u_t + b_e)
    M_t = a_t M_(t−1) + (w_t/R) Σ_r v_tr k_tr^T
    z_t = a_t z_(t−1) + (w_t/R) Σ_r k_tr
    r_t = M_t q_t / (z_t^T q_t + 10^−4)

For a legal initial state, z_j ≥ 0 and |M_ij| ≤ z_j are preserved by the
triangle inequality. Since 0 ≤ w ≤ 1−a and each key coordinate is ≤1,
z_j ≤1 is also preserved from z_0≤1. A nonnegative query then gives |r_i|≤1.
The exact-real argument does not imply an unconditional floating-point theorem.

Retention and eligibility have different jobs: a suppresses old state, while
eligibility suppresses the new offer. Zero eligibility alone can still let old
state decay. A false explicit write mask sets a=1,w=0, preserving that bank
exactly; it does not hide the byte from the rest of the model. Learned sigmoid
gates generally approach rather than attain exact freezing. Fast/slow retention
biases initialize different horizons; training is free to change them.

This preserves TAC's positive normalized memory rather than converting it into
a signed state-space bank. Learned headwise forgetting is implemented;
address-column erasure and delta-rule correction remain alternative hypotheses.
Adding them without new equations would invalidate the current chunk algorithm
or its invariant. Multiple writes can still collide, and contractions can erase
useful old differences: holding future coefficients fixed,
ΔM_t=(Π_i a_i)ΔM_0. Numerical boundedness is not information survival.

## Separate signed dynamics

For each head, the even-dimensional field uses independent real 2D pairs:

    f_t = a_t R(θ_t) f_(t−1) + (1−a_t) tanh(W_f u_t)

R is block-diagonal with ordinary sine/cosine rotations. Rotation can preserve
and move signed directions instead of permitting only positive decay. It does
not share the bank's coordinatewise cone. A norm envelope satisfies
m_t=a_t m_(t−1)+(1−a_t)||offer_t||₂ and ||f_t||₂≤m_t from valid initialization.
Because ||offer||₂≤√V, ||f||₂≤√V from zero. The nonlinear residual network and
its Jacobian are not covered by this branch bound. A false bank-write mask also
sets field a=1,θ=0. No claim of parity/cycle learning follows without training.

## What is and is not exact lookup

Recent attention retains separate K/V records for the current byte and up to
W−1 prior bytes per layer. Its query–key comparisons use causal position-aware
attention. Separate records avoid compulsory long-history compression inside
this window, but softmax mixing and learned representations still do not
guarantee exact payload recovery. Outside the window, neural information must
survive in the compressed state.

`exact_store.py` instead preserves explicit byte keys and byte values, with
full-key equality, owner/entity checks, version timestamps, bounded record
capacity, deterministic eviction and explicit misses. Retrieval is exact only
conditional on the correct retained key and legal observation time. The
reference store scans retained records; expected O(1) indexed lookup, GPU
parallel retrieval and learned extraction/routing are not implemented.
Evicted records cannot be recovered. This component is an API building block,
not evidence that the language model has learned exact recall.

For finite precision, B bits of carried state have at most 2^B distinguishable
configurations. They cannot losslessly represent every possible longer random
history. Unlimited exact lookback requires growing storage, a constrained input
domain, external records or an explicit error allowance. Cache bytes, not only
trainable parameters, are part of the resource budget.

## Parallel mathematics and GPU obligations

Affine transforms compose as (A₂,B₂)∘(A₁,B₁)=(A₂A₁,A₂B₁+B₂).
Our positive-bank A is scalar per head; scaled 2D rotations also form a cheap
closed composition family. This avoids pretending a general dense transition
scan has inexpensive matrix products.

Inside a bank chunk, D_ti=Π_(j=i+1..t) a_j for i≤t. Each rank contributes
causally weighted query/key scores multiplied by the value matrix. Chunk
summaries propagate incoming M,z; only chunk-boundary bank states are needed,
not a T×V×K bank trajectory. Decay factors must be formed without dividing by
tiny prefix products. The source supplies a recurrent reference path and the
parallel/chunk implementation; the engineering report records actual parity.

With fixed C, bank work is linear in T, approximately
O(T H R [K V + C(K+V)]); state is O(H(VK+K)) per bank. Local attention is
tiled, approximately O(T H (W+C)(K+V)), rather than full-history quadratic
attention. Field work is O(THV); projections/FFNs add O(Td²+TdF) at typical
geometry. These are operation-order descriptions, not complete FLOP receipts.

Per-stream recurrent tensor capacity after the local window fills is

    L H [2(VK+K) + V + (W−1)(K+V)] floating values.

Count owners, clock, Python/runtime overhead and exact-store records separately.
Inference must use no_grad/inference_mode or explicit detach boundaries;
retaining autograd history is not constant-memory execution. Training activations
and backward storage are additional and grow with sequence length/depth.

Projections, heads, known token computations and within-layer scans can execute
in parallel. Layers depend on lower layers. Local attention uses query tiles.
Future autoregressive bytes cannot be generated concurrently by this recurrence
without another justified algorithm. Batching streams is supported.

The current implementation is device-generic PyTorch with FP32/FP64 state.
GPU execution, fused kernels, BF16/FP16 correspondence, tensor-core efficiency,
CUDA graph capture, batch-one decode latency and backward memory traffic remain
OPEN until actually measured. A mathematically parallel scan is not itself a
GPU performance result. Subsequent optimization should profile a fixed correct
graph before adding Triton/fused kernels, while preserving all parity tests.
Strict state validation also checks finite tensors using host-visible booleans,
which synchronizes a GPU. Production kernel fusion and asynchronous diagnostics
must be designed and verified separately rather than timed as though this
reference interface were already optimized for deployment.

Serialized state schema v2 binds the complete configuration, including RoPE
semantics, by a checked signature. It does not bind weight/checkpoint identity;
the runner must preserve that identity and handle stale state across optimizer
updates. Local RoPE phases use integer clocks and FP64 trigonometry before
casting to the state dtype, avoiding the FP32 adjacent-position collision beyond
2^24. Streams beyond the explicit 2^53 observation-clock limit are rejected.
This is a reference numerical policy, not a long-horizon error proof, and its
FP64 GPU overhead must be measured. A qualified relative-phase kernel remains
future optimization work.

## Actual untrained construction

`construct_presets.py` instantiates declared, prospectively selected geometries;
it performs no optimization or capability selection. `CONSTRUCTION.json` records
the actual parameter and owned tensor-storage counts:

| Preset | Layers / residual width | Heads / key / value / rank | Parameters | FP32 carried tensor bytes per stream |
| --- | --- | --- | ---: | ---: |
| Approximately 1M | 4 / 128 | 4 / 16 / 16 / 2 | 1,015,376 | 295,936 |
| Approximately 10M | 8 / 256 | 8 / 16 / 24 / 2 | 9,952,064 | 1,511,424 |

Both use W=128, C=32 and actual residual computation. Counts include all active
learned components, with tied embedding counted once. These are construction
presets inside the declared ±5% bands, not matched competitor registrations.
Tensor bytes are measured after the recent window fills; physical backing
storage is checked, not merely view size. Metadata/runtime overhead and explicit
record storage are excluded and separately disclosed. A bounded neural cache can
still be larger than a Transformer's KV cache at some shorter contexts; no
universal memory advantage is claimed.

## Why these mechanisms, and what the other experiments actually support

The current `Architecture_Portfolio_Phase_Two_Results.pdf` is exposed DEV evidence
and supplies design constraints, not winners to transplant:

| Evidence | Design consequence |
| --- | --- |
| Bounded field: 0/512 complete streams with unmarked distractors despite learnable no-delay/blank-delay recipes (p.7) | Separate write eligibility, forgetting and readout; require distractor tests |
| PSN: requested-key recall remains unqualified (pp.8–9) | Query must interact directly with preserved addresses; test query swaps |
| PSN2: local-readout collision witness; diversity helps native and controls (pp.10–12) | Supply required information before increasing width; test conditional composition |
| ATLAS: candidate comparisons learned by native and much smaller controls (pp.14–16) | Use learned relational projections; no demonstrated reason to add swarm geometry |
| DPSL: top-k5 blending fails under load, top-k1 diagnostic succeeds (p.16) | Keep exact records separate from blended summaries; test collisions/load |
| USEF: fixed pool also learns; NFD: responsiveness without alternating competence (pp.12–19) | Defer ecology/refractory dynamics until their necessity is demonstrated |

This creates computational analogues of distributed representations, selective
plasticity, multiple timescales, recurrence and hierarchy. It is not a simulation
of cortical tissue, a biological learning law, or proof that these conditions
suffice for intelligence. Biological inspiration must earn its place through
information availability, learnability, causal usefulness and measured cost.

## Prior art reused, not claimed as invention

- [LANGMATH098 / PR668](https://github.com/FPC-effortless/TAC-OSM/pull/668):
  positive normalized carried state and scoped invariants.
- [LANGCAP103 / PR681](https://github.com/FPC-effortless/TAC-OSM/pull/681):
  instantiated deep/wide candidates and source/probe infrastructure.
- [SSD / Mamba-2](https://proceedings.mlr.press/v235/dao24a.html):
  recurrent/parallel duality and block matrix computation.
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464): complementary forgetting
  and targeted updates, specialized chunk algorithms. No delta kernel is copied
  or claimed implemented here.
- [Mamba-3](https://arxiv.org/abs/2603.15569): expressive rotations and
  inference-oriented multiple-input/output engineering. Rank-R offers and a
  rotating auxiliary vector here are not a reproduction of Mamba-3.

The combination is a falsifiable engineering proposal, not a novelty proof.

## Validation and next trained study

Run the reviewed no-fit plan, then the focused unit checks and graph report.
No fixture prediction is a language capability score. The source adapter exposes
actual named state components; unsupported real-data, calibration, energy,
mixed-precision and trained-checkpoint dimensions remain INCONCLUSIVE.

The PNDS companion supplies source-pinned Python equations and Lean definitions.
Read its source map for exact coverage, build status and OPEN whole-model proofs.
No formal release is claimed while compilation/correspondence obligations remain.
The initial Lean interface covers the even-value-width subfamily; the optional
rotation-disabled source permits other widths. This coverage gap remains OPEN.
The independent Python formulation covers the full tiny tested neural graph,
including masked continuation and positions. Its numerical derivative audit
samples one coordinate in each of 60 named parameter tensors and 14 incoming
state tensors; it is not an all-element derivative proof.

Before any fit, freeze a new `declared-target-v2` authentic-data registration,
independent source/goal/reward/ACL evidence, measured counts and compute, and
operator-approved T4×2 execution. Reuse authentic FIT only after its licence,
near-duplicate and source reviews; do not reuse exposed DEV as final TEST.
Current-byte and window rules remain diagnostics. Primary comparisons must use
properly trained contemporary Transformer and recurrent/SSM implementations,
including real KV-cached inference, matched within 1% parameters and the declared
data/exposure/compute/tuning contracts. Do not relabel a GRU as Mamba or an
untrained cache probe as a trained language benchmark.

One integrated ladder should report held-out byte NLL, exact lexical completion,
query-swapped and distractor-delayed recall, unseen-repository reuse and
compositional transfer, then quality–cost curves. Keep semantic-memory-only and
explicit-store-assisted scores separate. Predeclare absolute utility targets and
relative gates before reading results. At each chosen scale require successful
learning controls and an adequate learning-curve budget, rather than assuming a
short failed run proves inability or a larger model guarantees success.

Necessary causal ablations: reset/shuffle each carried lane, remove local KV,
remove rotations, rank1 versus rankR, one versus multiple heads, matched depth
versus width, and exact store disabled. Match budgets or label the comparison
ineligible. Record all attempts. Measure prefill, batch-one/batched decode,
training backward, peak allocated memory, actual persistent bytes, complete
compute accounting and independently measured energy only when available.
