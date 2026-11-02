# COLLECT013 — global retrieval, routing and implementation proposals

Recorded 2 November 2026. User-supplied research brief, mathematical review and
executed admission diagnostics. These additions do not modify COLLECT012's
frozen training, measurements or launch request. No new neural optimizer updates.

## Research objective and all proposed directions

The objective is useful global retrieval with explicit accounting for memory,
computation, causality and the information lost by compression. Global historical
influence is not the same as exact arbitrary historical-token access.

| User proposal | Mechanism to retain | Qualification / experiment required |
| --- | --- | --- |
| Hierarchical memory: librarian and archivist | Exact recent-token window plus slower compressed phase state | Already retaining native TAC banks/SSA is not evidence this specific hierarchy works. Charge both paths; vary distance and distractor load separately; knock out each path. |
| External key-value tape or slots | Phase-addressed discrete entries, closest-slot retrieval | Discrete storage prevents additive superposition inside an entry, not address collisions, wrong retrieval, overwrites or eviction. Need stable identity/version ownership, absent rejection and collision policy. Exact arbitrary history needs retained information that generally grows with history. |
| Fact-retrieval/routing pretraining | NIAH, fact association and direct memory-gate supervision | Separate auxiliary supervision cost from ordinary LM exposure; retain natural-language held-out families and distant scopes. Articles/filler can matter to semantics; do not hardcode a universal stop-word write policy. |
| Dynamic expansion and sparse memory mixtures | Allocate capacity to complexity; specialize banks | Explicit maximum capacity, admission/eviction and charged routing/search cost. Growth gives up fixed-state memory claims. Sparse selection alone does not imply sparse executed kernels. |
| Switch-style balance loss | alpha * B * sum(f_b * p_b), per layer | B can be absorbed into alpha if B is fixed. f is measured hard assignment, p is mean soft probability. Differentiable surrogate encourages utilization; does not force optimal/even or semantically correct storage. |
| Entropy regularization | Add negative entropy to minimized loss | Use + alpha * sum(p log p), not the sign that favors collapse. High per-token entropy can conflict with confident useful specialization. |
| Noisy top-k and temperature | Explore routing with noise or sampling; anneal temperature | Positive temperature alone cannot change deterministic hard argmax. Train-only noise can change write bank identity; query/read consistency must be measured. Evaluate deterministically and retain RNG for replay. |
| Hard bank capacity | Bound assignments, reroute overflow | Shared batch quotas can violate owner isolation; full-sequence rerouting can depend on future tokens. Use owner-local causal prefix accounting or reject this option for autoregressive deployment. Do not silently drop writes. |
| Parallel associative scan | Compose affine state transitions | Plain delta with known gates has an affine transition; the normalized preconditioned direction depends on evolving P. Generic scan composition may cost cubic matrix work and large intermediates. Establish equivalence, work and memory before promising speed. |
| Triton/CUDA fusion | Fuse phase/read/gain/update where hardware permits | Existing timings are CPU observations, not evidence of GPU HBM thrashing. Profile first; preserve all backward paths, masking, precision/SPD and owner boundaries. Full state may not fit cheap on-chip residency at useful sizes. |
| Remove per-token SVD | Random projections, incremental QR, EMA, infrequent/asynchronous sketch updates | QR alone does not reproduce truncated-SVD rank selection. Fixed projections, stale geometry and EMA change the operator. Charge all workspaces and test strict maps and chronological consistency. No cheaper equivalent has been demonstrated. |
| Dimension padding | Align shapes and dtype with device kernels | Rank 32 is already aligned; it is not intrinsically an irregular dimension. Tensor Core use depends on GPU, dtype, operation and library, not a universal multiple-of-128 requirement. Current FP32 CPU fits say nothing about Tensor Core utilization. |

Primary grounding: [Switch Transformers](https://arxiv.org/abs/2101.03961),
[NVIDIA's balance-loss implementation documentation](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.moe.moe_utils.html),
[NVIDIA matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html),
and [MesaNet, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2c3d5a4eff20d24390a445762d61bfbb-Abstract-Conference.html).
MesaNet already describes stable chunkwise regression-based sequence computation
and an extra-FLOPs trade-off. RLS/Mesa, phase features, MoE balancing and fused
recurrences are prior art; no new general algorithm is claimed here.

## The proposed address and visibility formulations

Let D=2m. Proposed separate read/write projections construct unit phase keys
k=[cos(theta),sin(theta)]/sqrt(m), read queries q and separate values v.
theta=W_theta*x+omega*position is a proposed option. Current COLLECT012 instead
uses content/role-named hidden-feature phase projections, shared current read/write
address, and no explicit absolute-position phase term in this memory lane.

### A: causal recurrence

S_i = S_(i-1) + (v_i - S_(i-1) k_i) g_i^T.
Reading S_i*q_i is write-before-read; reading S_(i-1)*q_i is the currently
implemented read-before-write. Both can be causal when all ingredients use only
observed inputs and x_i predicts x_(i+1). They have different credit paths.
g=beta*k is plain delta; our normalized preconditioned gain is not exact RLS.

### B: bidirectional sweeps

y_i = S_forward_(i-1)*q_i + S_backward_(i+1)*q_i.
This supplies left/right compressed context, not exact pairwise attention.
It is valid for an encoder or an entirely observed prefix; it must not expose
unknown continuation tokens to a next-token predictor. Two sweeps are linear
in sequence length only for fixed state dimensions; include state-operation cost.

### C: one batch ridge memory

A = sum_i v_i*k_i^T; G = sum_i k_i*k_i^T;
S_star = A * inverse(G + lambda*I); y_i=S_star*q_i.
Use a solve/Cholesky rather than explicitly forming the inverse in code.
The accumulated statistics are permutation invariant to reordering the same
key/value/query-position records. Reassigning positions and regenerating keys
need not preserve the result. No item is guaranteed a nonzero or useful influence.

For key dimension D and value dimension V, work is approximately
O(n*(D^2+V*D)+D^3+V*D^2), including the solve with value RHS; retained statistics
cost O(D^2+V*D), excluding inputs, activations and temporary storage. The user's
O(n*m^2) simplification assumes fixed/comparable value dimension and includes
an additional one-time solve term. This is an encoder/bounded-prefix option,
not a whole-sequence next-token objective.

For a retrieval benchmark, write only context associations, never query answers
or label-bearing target values. Include a self-exclusion/control so reconstruction
from writing the queried token itself cannot masquerade as long-range retrieval.

## Exact RLS and what it does / does not prove

With S_0=0, P_0=I/lambda, lambda>0, no forgetting and unit observation weights:

g_i=P_(i-1) k_i / (1+k_i^T P_(i-1) k_i)

P_i=P_(i-1)-g_i k_i^T P_(i-1)

S_i=S_(i-1)+(v_i-S_(i-1)k_i)g_i^T.

Woodbury gives P_i=(lambda*I+sum_(j<=i) k_j*k_j^T)^(-1).
Multiplying S_i by P_i^(-1) gives sum_(j<=i) v_j*k_j^T by induction.
Thus each prefix is the corresponding batch ridge solution in exact arithmetic.
Finite precision, arbitrary initial S, forgetting and alternative gains require
their own qualification.

The existing operator instead uses direction Pk/(k^T Pk), multiplied by a learned
gain, while updating P with a different denominator. Do not transfer exact RLS
equivalence to it. Exact ridge also shrinks predictions when lambda>0: even
orthogonal keys do not generally give exact original-value recall.

The phase identity k(theta)^T k(phi)=mean_j cos(theta_j-phi_j) is exact.
Its relevance to interference depends on the whole operator: preconditioning,
subsequent updates, routing, value structure and decoder also matter.

For unregularized arbitrary value assignments, full-column-rank keys permit
interpolation and require N<=D. This is a worst-case representational condition,
not a universal maximum number of retrievable items. Compatible/structured values
can be reconstructed with N>D; repeated keys with conflicting values cannot all
be exact even when N<=D. A nonlinear decoder can alter practical behavior but
cannot recover arbitrary distinctions that have been erased from the state.

## Is it still attention?

For additive S=sum_j v_j*k_j^T, S*q=sum_j v_j*(k_j^T*q): a linear-feature
attention form. For batch ridge the coefficient is k_j^T*(G+lambda*I)^(-1)*q,
not simply k_j^T*q. For delta writes the affine recurrence has
T_i=I-k_i*g_i^T and S_i=S_(i-1)*T_i+v_i*g_i^T; later T matrices change earlier
coefficients. Therefore the user's simple additive expansion is not the exact
formula for every proposed update.

Removing materialization of an n-by-n matrix does not itself distinguish the
mechanism from attention; efficient exact softmax attention need not materialize
that matrix either. Linear reads do not generally equal softmax-normalized
query-dependent selection. Learned phase features may nonetheless yield selective
responses. An output nonlinearity is a candidate, not a cure for lost information.

## Executed COLLECT013 admission checks

Protocol: COLLECT013_PROTOCOL.json, fixed before execution. Source:
collect012/global_admission.py. Raw output: results/COLLECT013_ADMISSION.json.
Six generated mathematical fixtures: seeds 13001/2/3 in FP32 and FP64.

- Exact RLS matched every batch-ridge prefix: worst absolute error 2.0862e-6
  across both precisions (declared FP32 tolerance 2e-5, FP64 1e-11).
- RLS final states agreed under permutation within tolerance. The existing
  normalized write differed from RLS by at least 0.5515 in these fixtures.
- Changing only future values left causal prefix reads exactly unchanged,
  but altered global ridge reads by at least 0.0863. This demonstrates why the
  global variant cannot be slipped into whole-sequence autoregressive training.
- Positive ridge shrank even orthogonal-key recall. Compatible N=12,D=4 values
  were reconstructed by unregularized least squares within tolerance, qualifying
  the asserted universal N<=D ceiling.
- Positive temperatures 1,2,5,10 left every tested hard argmax unchanged.
- On the actual fixed COLLECT012 hybrid seed12003 checkpoint, a Switch-style
  balance loss had finite nonzero router-weight gradients in all four layers.
  Gradient L1 values: 3.4453, 3.9458, 2.4603, 1.6478. No optimizer step was taken;
  weights remained unchanged. This establishes that the auxiliary can reach the
  router, not that balancing improves retrieval or language loss.

The router fixture uses one saved 512-target minibatch; its usage percentages
are not the previous 8,192-target DEV utilization study. Keep the cohorts distinct.
All checks pass as admission fixtures, not a trained repair or global-retrieval result.

### Additional saved-model CPU profile

After the mathematical admission checks, collect012/profile_saved.py ran one
warm-up and one instrumented FP32 forward/backward on each seed12003 saved
512-target minibatch, one CPU thread, with no optimizer step. All checkpoint
weights remained unchanged. Raw events: results/COLLECT013_CPU_PROFILE.json.

| Actual graph | batched matrix multiply calls | where calls | multiply calls |
| --- | ---: | ---: | ---: |
| Local attention | 48 | 2,352 | 7,886 |
| Wide phase | 4,628 | 7,916 | 20,069 |
| Hard hybrid | 4,628 | 7,916 | 20,585 |

The phase paths execute many more small matrix and elementwise operations and
corresponding autograd nodes. In the wide-phase profile, self CPU time included
11.85% multiply, 8.35% where and 7.38% bmm; no single operation accounts for all
overhead. This supports investigating operation granularity and recurrent
forward/backward dispatch, without establishing the share due to Python itself.
Profiling overhead is substantial; these are single instrumented passes, not
latency rankings or complete allocation/peak-memory measurements. No GPU was
profiled and no HBM diagnosis or Triton speedup follows from these CPU events.

## Research sequence and criteria

First establish a faithful causal reference and routing-loss gradient path; these
admission checks now exist. Next preregister actual training comparisons before
any optimizer update: unchanged hard routing, balance loss, and a separately
charged exploration/warm-up control, with original ordinary-text exposure and
fixed final checkpoints. Measure delayed fact retrieval, overwrite/absent rejection,
language loss, per-bank usage, read/write consistency and all-seed whole-map recall.
Balanced usage by itself is not the success criterion.

Separately compare exact causal RLS, current normalized writes and plain delta,
then independent read/write addressing versus shared addressing. Compare phase
and normalized unconstrained keys at equal dimension and explicit parameter/
compute accounting. Treat positional-phase choice as a separate factor.

For the noncausal encoder track, compare batch ridge, bidirectional recurrence
and true global softmax on identical available context, sweeping N around D/2,
D and 2D plus distant overwrite/absent cases. Do not compare an encoder's full
context to an autoregressive model's past-only context as an architecture win.

Profile the exact CPU implementation before assigning Python, SPD checks,
linear algebra, threading or allocation as the dominant cause. GPU fusion needs
separate measured forward/backward/reference, precision and memory checks.
Only mathematically equivalent chunk/scan changes qualify as implementation
speedups; QR/EMA/sketch changes remain algorithmic ablations. No Triton speedup,
new trained repair, additional GPU allocation or full global-attention replacement
has been established in this round.

## Lay explanation and game concept retained

Retain the title “The Shorthand Detective”: a stream of mystery clues, a bounded
notepad, address codes, residual erasure, delayed questions, overwritten facts,
collisions and bank-utilization failures. Hierarchy adds recent verbatim notes
and an archive; external storage adds individually addressable cards.

Corrections to the analogy: full attention is a finite-context scrapbook, not
infinite perfect recall. Dense full-context training has quadratic pair work;
cached per-token inference reads a linearly growing history and stores a linearly
growing KV cache, not an inherently quadratic retained corkboard. Sliding-window
stacks can propagate older information through representations, so shredding
does not mean no old information can survive. Residual correction is a numerical
association update, not a semantic truth/contradiction detector. Uneven bank
usage is measured; its causal effect on task failure is not yet established.

No game was built: this section preserves an explanatory concept for future use.
