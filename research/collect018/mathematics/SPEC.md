# COLLECT018 source-specific representation and open obligations

The three forward graphs have two registered objective profiles each. Native
phase/context use the frozen four-block width16/FFN57 C16 graph (99,971 active
weights). Standard GRU uses embedding[256,32], one GRU[32,125], affine LayerNorm125
and output[125,256] with bias (100,323 active weights). All output heads score
all256bytes. No frozen pretrained model, dummy parameter or legal-candidate mask
is added. Counts differ by0.3521%, below the independent1% ceiling.

For native block l, compute the original fast/slow bank paths and rotating
field from z=RMSNorm(x) and the prior owner-specific state. Let B contain the
original concatenation (fast,slow,fast-slow,fast*slow,field,0,fast-evidence,
slow-evidence), shape[B,T,392]. The context candidate computes

    c = RMSNorm(x + scale_l * W_perspective B)
    phase,route,admission = original projections applied to c
    value = tanh(W_value z)

The normalized preconditioned delta write and read-after order are unchanged.
The original final perspective/FFN residual equations, SSA placement and complete
state are retained. The baseline uses z for both paths. No phase result enters
its own c, no target/role metadata enters either path, and no new weights or
carried tensors are created. The extra perspective projection has compute cost.

Native state per owner/block has fast/slow matrices[4,16,16], masses[4,16],
rotating field[4,16], empty local KV fields and phase memory/precision[4,16,16].
The midpoint SSA has32keys/values of width16 plus survival scalars. GRU carries
hidden[1,B,125]. Owner order, clock, shape, dtype and model signature are checked;
continuing state cannot silently move to another owner. Float32/64 are the
numerical reference domain. All clocks count observed bytes; no unknown future
answer is supplied before its decision. Explicit reset and wrong-history/query
interventions are evaluation controls, not training oracles.

The two losses are joint=naturalCE+.25taskCE+answerCE+.01balance, and
focus=joint+3firstCE. Four answer bytes remain supervised in both. The latter
changes the initial retrieval pressure without changing target alphabet or
teacher-forcing the evaluated answers. Natural row0 and three copied rows share
the same128byte budget. Full BPTT uses a complete128-byte training call. Arbitrary32-byte carry
boundaries are verified separately and used at evaluation. All six arms see identical bytes/targets/order per seed; objective profiles
are an intentional treatment, not a hidden fairness mismatch.

The executable independent reference enumerates the actual native/phase and GRU
equations and compares forward, carried state and gradients in FP32/64. Native
hard routing uses a declared straight-through surrogate, never a claim of the
ordinary derivative of argmax. Masked native writes and GRU observation-freeze
support differ; the latter is explicitly unsupported, not silently certified.
Generic candidate permutations have no legal axis and are INCONCLUSIVE. Zeroing
a precision matrix violates its SPD contract and is not claimed a legal probe.

The typed Lean definitions specify the context composition, GRU update, losses
and causal composition interface. Full operators/weights/SSA/normalization/IEEE,
surrogate-VJP and information-sufficiency theorems are OPEN. Lean compilation,
local algebra and finite numerical agreement do not prove a complete neural
model, independent source authenticity or retrieval capability. No proof holes
or custom axioms may be counted as proved.

Source inspection found a1.0371e-5 FP32 batched/individual precision-state
difference at the generic probe's tighter default1e-5 threshold. That failed
receipt is retained. The same fixture gives3.55e-14 in FP64 and fixed-batch
owner-poison tests are exact. The campaign uses the4e-5 bound already used by its
initial source/reference and chunk tests; no trained score or checkpoint was
used to choose it. Finite-precision/global information-flow proof remains OPEN.

Memory is bounded; information survival is not guaranteed. Contextual addresses
can also interfere, become unstable or favor current prediction. This is not an
exact-RLS implementation, an attention-equivalence theorem, an external archive,
a capacity result or a novelty claim. Prior art includes Gated DeltaNet's local
context before projections and the standard PyTorch GRU equations (reset applied
after the recurrent candidate affine). Primary references:
https://arxiv.org/html/2412.06464v3 and
https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html.
