# LANGLAYER038 — recurrence-only upper-layer readiness

UNTRAINED candidate, following the completed original LANGLAYER035/036/037
studies, not another replay of their completed lesions or fitted results.
Preregistered c0e5936e387fec77a7be889f81bcf44acdfd6ea6 before source changes
and measurements; original source/splits, results/checkpoints, H64 negatives
and prior mathematical records preserved. CPU20min plan, zero GPU, no-fit
metadata gate PASS. No new source labels scored or protected TEST opened.

## Candidate and falsifiable motivation

Original recurrent depth2 has NLL2.098918 vs FFN2 2.108155 on exposed DEV;
paired descriptive confidence intervals include zero. Neither establishes a
useful win over original TAC or P3, and training FLOPs were not matched. The
frozen036 lesions already found recurrent reads influential but upper concept
graph specificity/history nearly inert at this protocol. That motivates a
candidate, not an inference that concepts never matter.

Reuse the inherited complete CONTEXT026 first stage and its native/aligned/
wave/concept paths. Replace ONLY the extra recurrent+concept block with:

```
u_t = tanh(W_in LN(z_t))
a_t = .995 sigmoid(W_forget u_t)
v_t = tanh(W_offer u_t)
h_t = a_t*h_(t-1) + (1-a_t)*v_t, h_(-1)=0
g_t = sigmoid(W_read [u_t,h_(t-1)])
z'_t = z_t + sigmoid(gain) W_out tanh(u_t+g_t*h_(t-1))
```

No extra concept seed/graph/projection/equilibrium solve. Original base
concepts remain. Upper recurrence and readout widen48→59; final SwiGLU stays
442, exactly the old recurrent2 decoder width. This is **reallocation**, not
an otherwise identical same-state-width concept lesion. The total upper
parameter formula4h²+564h+841 gives48,041 at h59, closest integer state size
to old47,865. Selection used size arithmetic, not DEV scores. All weights
are active graph components; no frozen/dummy padding.

| Architecture | Active/deployed parameters | Upper recurrence | Final FFN |
|---|---:|---:|---:|
| Original CONTEXT026 | 500122 | none | 501 |
| Frozen recurrent2 | 499725 | 48 +96 concept seed | 442 |
| Frozen FFN2 | 499661 | none | 450 |
| New recurrence-only038 | **499901** | **59**, no extra concepts | **442** |

Reset source windows only; no whole-model carried-state/streaming interface.
The actual upper observe/read entrypoints support its source-reviewed
ResearchProbeAdapter. They match its scan forward in tested cases. Native
write_mask still applies only to original associative bank admissions;
upper/wave/concept updates are not globally masked. The adapter covers ONLY
the new upper block; inherited full-core state intervention/AD coverage
remains INCONCLUSIVE, never borrowed from another model. Candidate permutation
is inapplicable/not declared. Nominal state coordinates exclude weights,
scan workspaces, context buffers and activation/recomputation storage.

## Actual engineering measurements and limits

9 local constructed actual-graph tests PASS: live count, all13 upper-parameter
nonzero finite gradient paths, prewrite chronology, source prefix/owner/reset,
native admission, inherited base trace identity, invalid inputs and pure
upper adapter/scan agreement. These are not learned skills.

34 independent serial cases PASS: six constructed cases (B2,length1/7/17,
FP32/64), plus28 genuine-input checks (all14 authenticated public PSF CPython
FIT owners × both precisions, first33 bytes). Captured actual first-stage
features feed the independent upper reference. Compare every upper output,
state/gate/offer trace, all13 parameter VJPs and observed-feature VJP with
fixed sine-weighted output; no labels, training loss, optimizer or DEV NLL.
Zero VJPs at first-step boundaries are recorded. Whole prediction prefix
poison checks pass on the authentic FIT bytes. Full receipt emitted by CI.

| Precision | Max forward/state error | Max VJP error |
|---|---:|---:|
| FP64 | 6.35e-16 | 4.34e-19 |
| FP32 | 2.84e-7 | 3.21e-10 |

Frozen tolerances are FP64 atol1e-11/rtol1e-9 and FP32 atol1e-5/rtol1e-3.
Independent operator composition shares PyTorch primitives/autograd; one
VJP objective/finite cases are not complete Jacobian or IEEE proofs.
[PNDS draft64](https://github.com/FPC-effortless/pnds-lean/pull/64) defines the
complete extra upper graph with independent Python and six claimed scoped
Lean properties (CI required): sigmoid bounds, conditional convex coordinate
invariant, same-schedule state-difference identity, zero prior at first read,
prefix-state and prefix-output causality. Whole composed parent+upper math,
formal IEEE/AD and trained information survival remain OPEN.

## Next gate

Register an actual future fit separately with applicable owner-scale policy,
original source/order/exposure and checkpoints, fair data/tuning/training
compute controls, and complete applicable mathematical obligations. Older
035/037 study-specific500K exceptions are not silently extended by this
no-fit registration. No external signer blocker applies to ordinary internal
checks. Current readiness is engineering only, internally measured and not
independently qualified; it establishes no trained NLL/accuracy advantage,
long-context benefit, safety/reward certificate or model promotion.
