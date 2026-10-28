# LANGLAYER034 — layered causal-residual language TAC (UNTRAINED)

**Parent:** CONTEXT026–028, TAC draft PR #556 at `b33a7c2e354d1d99c8a90d87a1683354d36675c1`.
**Scope:** separate no-fit architecture candidate. No existing source, trained checkpoint, baseline,
experiment output, source receipt, or seal is overwritten. LANGTRAIN025 (native NLL
2.099349 versus exposed-DEV P3 1.877306, 0/5 wins) remains negative evidence.
The recent LANGCOMPARE031 TAC mean NLL 2.099715 and TAC/P3 hybrid result are
**historical exposed-DEV measurements**, not scores of this variant.

## Construction

Input bytes [B,T] -> unchanged CONTEXT026 full native/aligned/wave/concept
causal layer -> original 280-wide LayerNorm -> 1 or 2 *independent*
causal residual memory/concept layers -> shared narrowed SwiGLU head +
original direct concept skip -> next-byte logits [B,T,256].

DeepSeek-V4.1-Flash introduces an asymmetric Causal Encoder–Decoder (CED) with
cross-layer cache reuse, not TAC's pre-existing Concept Equilibrium Dynamics
(CED). This work borrows the *hypothesis of asymmetric layered reuse*, not
DeepSeek's Transformer, sparse attention, 20+20 stack, MoE, KV compression, or
claimed performance. Parent first-stage source processing is reused across
the deeper residual layers, which have their own weights and recurrent states.
This is **not two identical full-native associative TAC blocks**.

Every deeper layer reads the previous layer's same-timestep features z_t,
but no later token. It computes:

    u_t = tanh(W_in LN(z_t))
    a_t = 0.995 sigmoid(W_a u_t)
    v_t = tanh(W_v u_t)
    h_t = a_t ⊙ h_(t-1) + (1-a_t) ⊙ v_t , h_(-1) = 0
    g_t = sigmoid(W_g [u_t, h_(t-1)])
    c_t = (1-η_t) ⊙ c_(t-1) + η_t ⊙ o_t, η_t=.05 sigmoid(W_c u_t)
    G_t = reshape(o_t, 6, 16) + reshape(c_(t-1), 6, 16)
    A_t = offdiag(0.5*[softmax(QG_t (KG_t)^T/4) + transpose])
    L_t = diag(A_t 1)-A_t
    H_t = 1.25 I_6 + 0.25 L_t
    C_t = solve(H_t, G_t+0.25 reshape(c_(t-1),6,16))
    p_t = Σ_i softmax_i(mean_j G_(t,i,j)) C_(t,i,:)
    z'_t = z_t + sigmoid(γ) W_out tanh(u_t + g_t⊙h_(t-1) + W_p p_t)

Here o_t=tanh(W_o u_t), Q and K have independent learned 16x16 projections,
and the recurrent c_t is a 96-coordinate six-concept seed. Two independent
input-only affine scans (h,c) are used per added layer. The two-layer
configuration has ONE such block; the three-layer configuration has TWO.

Exact-real property: symmetric A has nonnegative off-diagonal weights,
therefore v^T L v = Σ_{i<j} A_ij (v_i-v_j)^2 >= 0. Hence
v^T H v >= 1.25 ||v||² and H is nonsingular. This mathematical derivation
has NOT yet been imported into and checked as a complete Lean/source proof.
It does not prove IEEE finite behavior, conditioning at all input scales,
robust gradient magnitudes, long-term memory benefit or capability.

Causality depends on stage-one CONTEXT026's already-scoped prefix semantics:
every deeper recurrence reads only source-prefix-conditioned z_t, and all
scan schedules depend on current u_t. No labels, future source bytes, target
cache, external n-gram/P3, attention KV cache, or hidden oracle enter forward.
The existing write_mask applies ONLY to the original associative memory banks,
not to stage-one wave/concept updates or to added layers. This is deliberate,
and must not be misrepresented as universal observation masking. Window
start must equal zero; **no carried-streaming state API** is implemented.

## Live model budget (256 bytes)

| Model | Active and deployed learned parameters | Final FFN width | Residual recurrence |
| --- | ---: | ---: | --- |
| Parent CONTEXT026 | 500,122 | 501 | none |
| LANGLAYER034 depth 2 (primary) | **499,725** | 442 | one 48-coordinate tracker + six 16D concept seeds |
| LANGLAYER034 depth 3 (ablation) | **499,934** | 446 | two separate 24-coordinate trackers + separate six 16D concepts |

The budget is redistributed from the shared head, never padded with
unused/frozen weights. Parent vs depth-2 gap 0.079%; depth-2 vs depth-3
gap 0.042%. Base original summary is 520 floats; added reset-window
persistent state is 144 (depth 2) or 240 (depth 3) floats. This excludes
scan workspaces, activations, parameters and unimplemented streaming costs.
Time O(T log T) work per Hillis–Steele scan here, not demonstrated O(T)
hardware work or asymptotic throughput advantage.

The repository's old general **100k** mandatory training rule and the
operator's newer language-specific **~500k** research direction conflict.
These are **NO-FIT prototypes only** until the applicable model-scale policy
is explicitly reconciled in a preregistered prospective training gate.
Do not infer that the parity percentage itself authorizes any fit.

## Local no-fit validation and future scientific gate

    PYTHONPATH=src python -m pytest -q tests/test_langlayer034.py

Local constructed-graph validation: 8 parameterized pytest cases passed
for forward shape, prefix noninterference, source admission, gradient
connectivity, independent serial-versus-parallel recurrence, SPD numerical
eigenvalues, and invalid inputs. These are synthetic/unit tests ONLY, not
language prediction, memory generalization, independent Lean proof or source
qualified empirical evaluation. The standalone residual recurrence test uses
FP64 tolerance 1e-13.

**Before fitting:** separately finish PNDS full layer graph + executable
source correspondence, freeze original publisher FIT/DEV splits/sampling,
parameter counts, training steps/tokens/FLOP budgets and checkpoint replay.
Use unchanged CONTEXT026 and the two depths, P3 as an *unmatched non-neural
reference*, and matched selective SSM/GRU/Transformer where possible.
Ablate recurrent state and concept exchange separately without changing
source exposure. Use P3 and hybrid as separate practical controls; report
training + inference cost/state bytes and every negative seed. DEV was
previously exposed; do NOT tune or claim superiority on protected TEST.

**Open:** full PNDS Lean->Python->actual-source equivalence, independent
backward/AD analysis, whole-graph finite precision, trained retrieval and
prediction gains, throughput/streaming evidence, external validation,
safety/reward qualifications. No new training, GPU, checkpoint promotion,
protected TEST, merge or deployment.
