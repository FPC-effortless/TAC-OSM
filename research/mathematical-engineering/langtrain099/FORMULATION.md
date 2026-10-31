# Exact 10M graph and proof scope

Let d=256, c=376, k=24, v=32, w=4335. With byte vocabulary 256, f=2d+c=888, projection p=4k+2v+2=162, comparison h=d+c+3v+2=730 and decoder z=2d+c+3v+2=986, the actual parameter sum is

P = 256d + 3d + c(d+1) + c + (f+1)p + (h+1)v + 2z + 2(z+1)w + 256(w+1)
  = 332950 + 2230w = 10,000,000.

Every term is active. Persistent per-stream state contains c+3d+2k+2vk+2k = 2776 floats = 11104 bytes in FP32, excluding weights, allocator/Python overhead and temporary activations.

The graph is the dimension-scaled LANGMATH098 graph: bounded byte embeddings and three-lag local mixing; positive-timescale diagonal context EMA; contextual softmax key/query and tanh value projections; two normalized lag-key/next-value associative banks with learned retention in [0.97,0.995] and [0.995,0.99995]; per-coordinate fast/slow comparison; directly exposed bank reads plus local/context features; LayerNorm and gated feed-forward byte logits. Previous-key/current-value writes bind next information to its observed past context. No future targets enter the forward graph. All floating state is reset or intervened together.

For each bank, M'=aM+(1-a)vuᵀ and m'=am+(1-a)u with u≥0, sum(u)=1, |v_i|≤1, a∈[0,1]. The invariant m≥0 and |M_ij|≤m_j is preserved, yielding |(Mq)_i/(m·q+epsilon)|≤1 for q≥0 and epsilon>0. Read normalization controls write interference amplitude; it does not eliminate addressing collisions or prove useful recall. First observations cannot form a lag binding and do not write. Nonwrite leaves bank memory/mass unchanged but still updates observed context, lexical history and previous key.

The two affine recurrences have associative tensor composition (a,b)∘(a',b')=(a'a,a'b+b'). Pair reduction and reconstruction give O(T) state operations and O(log T) ideal scan span at fixed geometry, no division by prefix products. Dense learned projections/readout dominate scaled FLOPs. Constant history-independent stored state does not mean zero acquisition cost: all observed bytes are processed, and the forward pass remains O(T). No optimality claim follows from scan work efficiency or exact parameter count. Streaming launch overhead, dense projections, training activations and optimizer memory are measured separately.

PNDS/LANGTRAIN099.lean specializes the existing generic real-valued 22-theorem suite to native context376/embedding256/keys24/values32 and proves the full persistent-state invariant, actual bank bound and tensor affine action/associativity. Independent NumPy reference covers the complete input-to-output graph, including normalization and decoder. Actual-source constructed FP64 reference, serial/scan, chunk, causality and gradient checks accompany it. These give explicit scoped proof and numerical evidence. Complete IEEE floating-point/automatic-differentiation refinement of the neural encoder and readout remains OPEN; no finite test suite completes it. Learned capability and global mathematical optimality are not theorems here.
