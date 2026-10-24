# TAC-OSM mathematical engineering audit v1.1

**2026-10-24. Mathematical research and pinned-code verification only. No training, new checkpoint, real-world gain or architectural conversion to SSM.**

Formal companion: https://github.com/FPC-effortless/pnds-lean/pull/7
Original source: TAC-OSM commit 081e534a074001c6a3b2251b3ff31ad20ec4fca8, src/tac_osm/aligned_associative_tac.py.

## Reference standard

Transfer the *engineering discipline* of Gu and Dao, Mamba (https://arxiv.org/abs/2312.00752), and Dao and Gu, Structured State Space Duality (https://proceedings.mlr.press/v235/dao24a.html): derive state equations; establish exact assumptions and composition laws; decide legal serial versus parallel operations; prove stability/accuracy only for characterized state classes; verify actual kernels and backward numerics; measure capability separately. TAC's nonlinear, action-conditioned, evidence-aware memory architecture is retained as its own model family. It is not made an SSM merely by applying scan theory to an optional affine filter.

## Full source-factual graph order

For DIAGKEY004 the typed state is (fast[88], slow[88], associative[8,8]), plus the current encoded observation and address. Its original forward graph is:

1. Encode known-at-time event, key and value; normalize the learned address and learned content.
2. Read the **old** associative matrix at the observed key; update nonlinear fast state using that read.
3. Update nonlinear slow state using **new** fast state and **old** memory read.
4. Compute retention and write gates from encoded event, new fast and new slow.
5. On writable events: decay old memory, correct read prediction using value-by-key outer product, project to radius 4. On read-only events: preserve original matrix *exactly*, skipping decay.
6. After all history updates: read last key, run learned decoder and value-codebook similarity correction.

This sequence is inherently state-dependent and **not** commutative. Independent embeddings and batch/entity/horizon reads are parallelizable; serial fast-to-slow-to-write dependencies are not established as a scan. ME095's diagonally affine adjunct admits associative composition only if its per-step affine coefficients are independent of the state scanned.

## New exact conditional native memory mathematics

Let k be the 8D learned address, v the 8D value, alpha the retention, beta the correction and M the 8x8 store. On a writable observation:

D = alpha M, e = v - Dk, U = D + beta e k^T,
M' = Projection_FrobeniusRadius4(U).

On a **read-only** event M' = M. For any query q, *before projection*,

Uq = alpha Mq + beta e <k,q>.

Consequently a stored q which is orthogonal to k has no correction contamination, but its read still decays by alpha. For a unit k,

v - Uk = (1-beta)(v-alpha Mk).

These are exact real-algebra statements with Lean two-key-coordinate row proofs. The finite-size 8x8 Python reference tests the generalized identity. The Lean results do not yet establish 8x8 torch/FP semantics, because actual implementation normalizes with an epsilon floor and applies clipping; a zero embedding is not a unit address. The source also clamps retention to a maximum of 0.9999 for writable events.

For orthogonal later writes with no projection and no refreshing, an old query obeys the conditional relation r_T = (product_t alpha_t) r_0. The maximum retention contribution after 8192 such writable steps is 0.9999^8192, approximately 0.4406, even with no direct corrective contamination. This is a mathematical *conditional attenuation* observation, not an observed recall score.

For nonorthogonal writes, with e_j = v_j - alpha_j M_(j-1) k_j, the exact unprojected recurrence at q unrolls to:

r_T(q) = (product_i alpha_i) r_0(q)
       + sum_j [(product_{i>j} alpha_i) beta_j e_j <k_j,q>].

This formula exposes correction, interference and forgetting as separate contributions. It does **not** justify parallel scanning of the whole model: e_j, beta_j and alpha_j depend on previously computed states. Under norm projection, the formula is not exact and must be replaced by a bounded-error or measured analysis.

Exact post-write recall requires stronger premises. An ideal beta=1 unit-address write makes the *unprojected* read equal v, but a radius-four projection can reduce a value-five write to four. A shared key cannot simultaneously read two different target vectors. Neither bounded Frobenius norm nor 99.84% *synthetic immediate* recall demonstrates 256-distractor real long-memory quality. The existing 2/819200 strict raw-logit mismatch remains a historical verification failure, not quietly corrected by new equations.

## Whole-state stability is still OPEN

Scalar fixed-target gated interpolation has derivative 1-g if gate and target are held fixed. In a learned state-dependent system, however,

h' = (1-g(h,u))h + g(h,u)c(h,u),

d h'/d h = 1-g + g * d c/d h + (c-h) * d g/d h.

The last two terms invalidate a universal contraction conclusion from 0<=g<=1. A full TAC certificate must use the *joint* block Jacobian J = d(f',s',M')/d(f,s,M), including old-read dependencies, fast-to-slow coupling, memory writes, role masks, normalization, and projection boundaries. Require either a weighted norm disturbance bound with explicit domain premises, or a reproducible counterexample.

Do not force every component to contract indiscriminately: a strongly contracting fast reflex and a protected, identity-addressed long memory are potentially compatible but need separate explicit invariants and independently evaluated useful state. That is a prospective mathematical design objective, not a demonstrated trained improvement.

## Other researched variants retain independent semantics

- Physical ME094: horizon-conditioned, multimodal median translation is coherent only under the base-median premise and finite inversion; implicit gradients divide by base density and need valid density floors and numerical checks. Its quiet-state empirical gate failed. Per-horizon marginals are not a jointly consistent physical trajectory by construction.
- ESWAM014: real actor IDs and 256 observed frames are finite-window representation, not unlimited recurrent memory. Learned slow relevance and quadratic speed gating cannot be claimed to solve its failed low-speed/quiet gate.
- Enterprise ME073: source end-of-record and conditional activity/duration distributions are not verified real workflow completion, causal action effects, or established decision costs.
- Counterfactual: p(y | observed a,s) is not an identified p(y | do(a),s) without a structural causal model and a valid identification design.
- Temporal and Edge: recorded valid-/known-time correctness and numerical/hardware constraints are different evidence obligations from a new learned model.

## Required typed computational-graph certification

For each next model version: (1) immutable source SHA and original checkpoint/evidence; (2) typed observation/action/time/mask/unit spaces and state dimensions; (3) exact ordered input-to-output node graph, role-conditioned memory writes and decoder; (4) Lean exact operator lemmas, counterexamples and explicitly OPEN whole-state goals; (5) Python reference and actual original-source FP32/FP64 forward/backward, JVP/VJP tests; (6) resource graph with measured FLOPs, temporal critical path, parallel-safe and sequential sections; (7) reproducible genuine real predictive tests under the standing parameter, training-compute and data gates. Equal CPU wall time is not required.

Level S0 means source/shape recorded; S1 means exact conditional theorem kernel checked; S2 means source-operator numerical correspondence; S3 requires full source-to-proof/gradient/stability contracts; S4 requires independent authentic demonstrated skill and decision utility. **No escalation between levels without its own evidence.**

Original source/negative artifacts, including ME069 loss to GRU, ME076 loss to persistence, ME094 quiet regression, ESWAM014 low-speed regression, and DIAGKEY004 strict-logit mismatch, are untouched. The prior 17-variant reciprocal register remains unchanged. This addition is a source-grounded proof-and-verification improvement, not a trained model update.
