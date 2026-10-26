# ME107 — role-specialized native TAC computational graph

Status: **untrained prototype**, not a capability result. No Transformer, attention,
SSM recurrence, extra recurrent state or decoder feedback into the same scan.
This records mathematical engineering standards, not Mamba performance parity.

For each event, the inherited masked encoder generates input-only source and
8x8 associative update schedules. Chronological affine composition is unchanged.
The final source/evidence/current-key read gives `phi in R^102`.

Three independent learned branches have fixed computational roles:

`F_r(phi)=W_r2 tanh(W_r1 phi+b_r1)+b_r2`, r in {current,width,point}.

Each hidden width is16. Outputs are12/12/6 respectively. These are ordinary
feed-forward networks acting on native TAC state, not temporal token mixers.
Fixed roles hardcode **permitted computation/output type**, not the target,
correct answer, data regime or useful-memory identity. No argmin-variance expert
selection or oracle task labels are supplied. All branch weights are learned.

Current outputs produce inherited `anchor+2*tanh(raw_mean)` and
`.0025+softplus(raw_var)^2+latent_variance`. Width outputs produce bounded
left/right multipliers `exp(log(4)*tanh(raw_width))`; source-age and missingness
floors are retained. Point outputs produce `.5*tanh(raw_point)`. The inherited
equal-half asymmetric Gaussian law in standardized log1p coordinates, censoring
at raw zero, CDF/NLL/Jacobian and quantiles remain the same probability law.
It is not a multivariate posterior or a new joint multi-horizon distribution.

The FFNs can be evaluated as parallel DAG branches after their common features
are available. Framework scheduling or CPU speedup is **not** guaranteed merely
by this dependency structure. The core scan retains its original input-only
affine schedule; no new sequential memory access is introduced by these heads.

Learned weights: `100349 - (1236+1236+618) + (1852+1852+1750)=102713`.
All39 parameter tensors are trainable. No dummy/frozen padding. Primary recurrent
floating state remains152floats=608B/sample plus existing integer/provenance
metadata; it is not whole-model/runtime RAM. New FFNs use fresh nonzero standard
initializations: **no bit-identical warm start** is claimed. A new trained
comparison against100349parameter ME106 would violate1% parity unless separately
matched. This no-fit prototype is not such a comparison.

## Gradient and optimizer boundary

ME106's uncertainty view still cancels the median-correction tangent for NLL and
interval objectives. The median is structurally independent of the width head;
reconstruction has its own parameters. **Shared ancestors can still conflict.**
Hard-assigned output roles do not prove specialization quality or loss harmony.

The separate pure guard consumes a complete proposed displacement `d_b` after
all optimizer effects, and point gradient `g_b` for each exhaustive parameter
block. It returns

`d'_b=d_b-max(0,<g_b,d_b>)/||g_b||^2*g_b` when the norm is positive,
and leaves zero-norm blocks unchanged. Already-feasible blocks are unchanged.
In exact reals `<g_b,d'_b> <=0` for positive norm. This is a first-order dot
constraint, not an AdamW/finite-step loss descent, MAE kink, other-loss or
generalization theorem. In FP32/64 the implementation computes in FP64, rounds,
rechecks and rejects any still-opposing protected-coordinate move. Zero-gradient
coordinates remain free. Invalid blocks/nonfinite intermediates fail closed.
The finite-precision recheck is tested, not formally refined to exact reals.

**The guard is not yet integrated into a trained optimizer.** There is no hidden
post-hoc claim that ME106 used it. Future training must specify moment updates,
clipping, decay, rejected-move semantics and costs prospectively, with source/math
parity and original checkpoint/replay evidence.

## Source/reference/formal scope

`RoleSpecializedSensorWorld` inherits the complete checked causal Python graph;
only its three raw head modules change. `scripts/me107_graph_reference.py`
rebuilds the three FFNs with direct tensor equations and uses the chronological
ME102 reference; it never calls the new model or FFN forward implementations.
FP32/64 source/reference forward, recurrent-state, parameter-gradient and chunked
streaming tests use inherited2e-5/2e-6 and1e-9/1e-10 contracts.

`PNDS/RoleSpecializedFFN.lean` defines the three FFNs, backbone without replaced
head tensors, inherited scan, current belief, source-protected forecast and
probability-operation wrappers. It proves exact finite-vector displacement facts
and parameter arithmetic. Constant unused old-head carrier fields bridge the
legacy mathematical API only; they are not learned/source parameters.

Whole-neural source equivalence, automatic differentiation semantics, IEEE
arithmetic, normal-CDF implementation correspondence, predictive improvement,
coverage, useful long memory and commercial utility remain **OPEN**. Existing
negative results and FP32 cross-host diagnostic exceptions remain recorded.
