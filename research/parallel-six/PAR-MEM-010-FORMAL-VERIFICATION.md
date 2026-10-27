# PAR-MEM-010: formal specification and verification record

This work specifies the complete default **point-prediction** graph and verifies
its associative-memory mathematics. It does **not** certify the complete Torch
program, its automatic-differentiation implementation, or an IEEE deployment.
Those distinctions are recorded as OPEN obligations in
`PAR-MEM-010-FORMAL-CONTRACT.json`; the record cannot be read as full-model
mathematical completion.

Source: TAC-OSM `63ea0aa67904fa946b9baa69d2e4b0cc23dc1c9a`
(`PAR-MEM-010-NUM-001`), descended from
`0c97e852cbdee1655a139e9ac203114ec6c692d6` / draft PR #520.
The numerical revision fixes masked-write byte preservation, without changing
the exact-real equations, learned shapes, checkpoints or six-family registry.
The native batch refinement is enabled only by the PAR-MEM-010 candidate; older
time-aware model instances retain their original batch arithmetic by default.
The independent executable reference pins this source SHA and the reciprocal
contract pins SHA-256 hashes of all five contributing source files.

## 1. Typed graph and information boundary

For one source stream, observation and recorded action are in R^14; delta time
is a nonnegative real in caller-defined physical units; source owner and event
ordinal are exact nonnegative integers. The admission mask is a Boolean supplied
by the source. A requested forecast-time historical ordinal must be strictly
less than the current ordinal. A standalone historical query can address any
nonnegative ordinal no greater than the last processed ordinal, with matching
owner. These guards do not attest measurement existence or access permission.
The source runtime uses int64; Lean uses naturals and does not prove the int64
refinement or overflow behavior.

Persistent real state is fast basis sum [88], native memory [8,8], elapsed and
last-write clocks [2], and aligned memory [2,16,8]: **410 values**, or 1,640 bytes
when stored as FP32. Owner, last ordinal, seen flag and count are additional
metadata. Learned parameters total **98,326**; the 14-component delta scale is a
buffer. Lean's `exact_parameter_count` counts every active learned tensor.

`PNDS.SourceAlignedModel` explicitly specifies every dense projection, Fourier
feature, epsilon normalization, sigmoid, tanh, softmax, SiLU, fast/slow proposal,
pre-write read, memory update, historical decoder and forecast residual.
`source_aligned_model_math.py` independently executes those equations in NumPy,
using an event loop rather than the Torch parallel scan.

Clocks advance before the current time features. Last-write age uses only the
**previous** admitted-write timestamp. Source-only embeddings produce a basis
prefix sum and causal mean context; proposals and all memory schedules are then
known without either associative memory. `prepare_memory_independent` checks
this exclusion in the complete typed specification. This is not a claim that
the gate schedule is independent of earlier inputs: time features and context
have their own prefix computations. Nonlinear read refinement and both output
residuals remain after the memory scan.

Current measurements and recorded actions may affect the current prediction;
the current memory **write** cannot be read by that same prediction. Hidden old
measurements and future target labels are not query inputs. Historical targets
may enter a separately declared training loss after prediction; this work does
not invent or qualify such a training protocol. Masking memory writes does not
erase the current observation from the encoder or basis sum, and does not freeze
clocks, chronology or count. Recorded actions are conditioning variables, with
no `do(action)` interpretation. Outputs define no predictive probability law,
uncertainty calibration or abstention guarantee.

## 2. Corrective write, read interference and stability

For head h, M is 16x8, k is 8-dimensional, v is 16-dimensional:

    M' = alpha M + beta (v - alpha M k) k^T
       = M P + Q,
    P = alpha (I - beta k k^T),   Q = beta v k^T.

The source uses alpha=sigmoid(retention(e)),
beta=sigmoid(strength(e))*softmax(write_router(e))[h], and v=tanh(content(e)).
For non-admission the operator is exactly (I,0). No clipping/projection is
silently inserted. `row_affine_correspondence` proves the matrix entries equal
the corrective update. `four_block_write_correspondence` and
`rowIndex_bijective` cover every row of two 16x8 matrices with four 8x8 blocks.
The actual Torch permute/reshape realization also receives numerical checks;
its compiler/tensor-stride refinement remains OPEN.

For any old query q, with r=M[row,:], the exact read equation is

    <r',q> = alpha <r,q> + beta (v_row-alpha<r,k>) <k,q>.

This is `read_after_write`. Orthogonal keys avoid corrective interference but
still decay by alpha. For a unit key, the residual at the written address is

    v_row - <r',k> = (1-beta)(v_row-alpha<r,k>).

For an arbitrary key, replace beta by beta||k||^2. This includes epsilon-scaled
and zero keys; unit normalization must not be assumed at zero. A full write at
a repeated key can erase the earlier value. Soft routing does not remove this
limitation.

For two states with the **same** input schedule, write value and admission,
their row difference d evolves as alpha(d-beta<d,k>k). Lean proves the identity

    ||d'||^2 = alpha^2 [||d||^2-beta(2-beta||k||^2)<d,k>^2].

Thus if 0<=beta<=1 and ||k||<=1, ||d'||^2<=alpha^2||d||^2.
`normalized_key_energy` supplies the key hypothesis for the exact epsilon
normalization; `sigmoid_range` and `softmax2_range` give gate ranges. This is
state stability, not information survival. Different input schedules or changed
weights require additional perturbation terms.

Analytic corollary (not claimed as a separate Lean theorem): for fixed finite
weights and tanh-bounded embedding, alpha_h <= rho_h < 1, where
rho_h=sigmoid(retention_bias_h+sum_i|retention_weight_hi|). With
beta_h<=b_h<=1, ||v||<=4, and zero initialization,

    ||M_h,t||_F <= 4 b_h (1-rho_h^t)/(1-rho_h).

This bound can be large and is not the native model's radius-four budget.
There is no weight-independent uniform contraction margin during arbitrary
optimization, and bounded memory does not imply recoverable history.

## 3. Legal parallelism and chronology

Chronological right-affine composition is

    (P_a,Q_a) THEN (P_b,Q_b) = (P_a P_b, Q_a P_b + Q_b).

It is associative, generally **not commutative**. All four blocks obey this
law. `block_fold_matches_stream` proves arbitrary finite sequential/prefix
equivalence from nonzero initial state. `ordered_tree_matches_stream` proves
every ordered binary grouping, including empty identity padding. Applying any
pure downstream decoder preserves equal states. The full specified graph has
`causal_prefix` and `stream_resume` theorems. The concrete Blelloch control flow
and its IEEE differentiation are distinct refinement obligations.

For fixed head/key widths, the scan uses O(T) work and O(log T) composition
depth; streaming holds O(1) history state. Training materializes O(T) activations,
so constant deployment state is not constant training memory. Dense KxK
composition costs O(K^3) per node, whereas a direct rectangular streaming delta
write costs O(VK). No latency or energy improvement follows from depth alone.

## 4. Two unavoidable retrieval limits

The source encoding depends only on ordinal residues modulo 7,13,29,59. Their
product is **155,701**. Lean proves periodicity of the residue encoding and the
full real Fourier/key/router/historical decoder graph. The actual source also
gives bit-identical phase/key/router inputs at n and n+155701 because modulo
precedes floating conversion. For the same fixed memory and both legal queries,
the output is identical; it cannot exactly equal two distinct true contents.
This is a forced collision period, not a guarantee of injectivity before it.
Learned projections, zero rotations and other degeneracies can collide sooner.

Separately, source processing ordinals {10,20} permits a historical query for
15. That passes the chronology guard although 15 was never observed.
`chronology_does_not_prove_observation` formalizes this counterexample.
Evidence-backed retrieval needs independent observation-membership and source
permission evidence; arbitrary accepted ordinals must not become real targets.

## 5. Numerical revision and executable evidence

On the original source, a 129-event mixed-admission fixture changed **7,593**
aligned FP32 coordinates at nonwrite positions (maximum 1.4901161193847656e-8).
The equivalent FP64 fixture also changed bytes. The scan had evaluated the same
logical prefix through different parenthesizations. The zero-state/one-step
tests could not detect this. This is a finite-precision failure of byte identity,
not a counterexample to exact-real affine associativity.

NUM-001 gathers the most recent admitted prefix in **both** batch memory paths.
Before the first admission it selects zeros. This preserves the last admitted
bytes and gradients, retaining the same exact-real model. Streaming already used
an exact masked selection. A new 129-event regression requires bit equality for
both memories in FP32 and FP64, including leading nonwrites and backward flow.

The source correspondence runner currently records **467 comparisons**:

* FP32/FP64, T=1,2,3,7,16,33,129; all/none/mixed admission; two source owners;
  large exact int64 ordinals; every implemented trace and persistent float field;
  standalone history and unchanged query state;
* actual parallel versus stream gradients for all 37 learned tensor groups and
  observation, action and delta-time inputs, with nonzero group gradients;
* independent centered finite differences for all 40 groups;
* analytic rectangular-head VJPs versus actual Torch affine-kernel gradients,
  including masked writes and a zero key;
* exact masked-prefix equality for both memory paths.

Temporal projection, retention/strength weights and predictive residual are
made nonzero to exercise paths hidden by initialization. The NumPy implementation
does not import TAC or Torch. Separate Fraction tests verify rational algebra,
rectangular row splitting, all prefixes from nonzero states, interference,
energy identities, chronology and periodic collisions.

Recorded maximum source/reference output errors: FP32 5.960464477539063e-8;
FP64 5.551115123125783e-17. Masked memory byte differences after NUM-001 are zero.
Per-comparison errors and fixed tolerances live in TAC's
`research/parallel-six/PAR-MEM-010-MATH-CORRESPONDENCE.json`.
These constructed checks are engineering evidence, not trained retention scores.

## 6. Finite precision and differentiation: remaining proof boundary

Under normal-range round-to-nearest arithmetic, a length-eight dot product
admits the standard componentwise bound gamma_8 sum|m_i k_i|, with
gamma_n=nu/(1-nu). Applicability requires an actual backend operation model:
FMA/reduction order, overflow, underflow, subnormal handling and transcendental
approximations cannot be inferred from the dtype name. Such backend bounds are
not yet proved here, and the single precision numerical tolerances do not apply
uniformly to arbitrary sequence lengths, weights, clocks or devices.

Lean's conditional `perturbation_step` and `accumulated_error` show how explicit
local norm error budgets propagate under a nonexpansive schedule. They do not
establish the IEEE hypotheses. The scan and streaming path have distinct local
operation/error budgets, including P/Q construction, regrouping and decoding.
The prefix-gather fix establishes byte reuse structurally, not bitwise equality
of all parallel and streaming predictions.

For one admitted head and upstream G, writing r=v-alpha M k gives the independent
analytic VJP:

    dM = alpha [G-beta(Gk)k^T]
    dv = beta Gk
    dk = beta [G^T r-alpha M^T(Gk)]
    d_alpha = <G, M-beta(Mk)k^T>_F
    d_beta = r^T Gk.

For a rejected write dM=G and other write-input derivatives are zero. These are
numerically checked against the actual kernel. Whole-graph formal AD remains
OPEN, especially epsilon-normalization/clamp boundaries and discrete masks,
owner IDs and ordinals, through which continuous derivatives are not defined.

## 7. Reproduction and release boundary

From PNDS:

    lake build PNDS.SourceAlignedModel
    python3 scripts/audit_source_aligned.py
    python3 -m unittest discover -s tests -p 'test_source_aligned*math.py' -v
    python3 scripts/validate_parmem010_math.py --tac-root /path/to/TAC-OSM

From TAC (CPU Torch and NumPy installed):

    python3 scripts/verify_parmem010_math.py --pnds-root /path/to/pnds-lean --output /tmp/parmem010.json

The original 17 historical variant records and negative findings are preserved.
The prospective record is added separately in both reciprocal manifests. No
optimizer update, protected-source evaluation, checkpoint promotion, causal or
calibration claim, domain-head qualification or merge is part of this work.
Complete formal verification requires closing the explicit implementation,
IEEE and AD obligations; retrieval advantage additionally requires the real-data
matched training/evaluation study described in the research program.
