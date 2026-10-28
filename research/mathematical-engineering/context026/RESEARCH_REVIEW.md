# Evidence-corrected architecture direction

LANGTRAIN025's source-paired4096-case P3 loss is preserved:2.099349221 neural
vs1.877306112 practical count reference,0/5 neural seed wins. Whole-FIT exposure
and compute were not matched; no500K SSM architectural comparison was completed.
The cause of this gap is not established merely by the score. All sampled DEV
targets are ASCII, so256 input/output support is not measured Unicode learning.

Published mechanisms and scope:
- [GatedDeltaNet2 official equation](https://github.com/NVlabs/GatedDeltaNet-2)
  and [preprint2605.22791](https://arxiv.org/abs/2605.22791): key-axis erase,
  value-axis write and channel decay are distinct. This is a reported mechanism,
  not a universal Euclidean nonexpansiveness theorem. Its oblique erase can
  transiently amplify norms even with interior gates. Our2D counterexample
  uses unitk=(1,1)/sqrt2,b=(.9,.1),D=.99I,x=(1,-1): output squared norm2.273832
  exceeds input2. Do not call unrelated channels unconditionally protected.
  No NVIDIA implementation code is copied here.
- [Mamba3 ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html)
  reports complex updates/discretization/MIMO and empirical scaling results.
  Current TAC already has affine scans and8 damped2D wave modes. Adding complex
  arithmetic alone does not implement Mamba3 or promise GPU speed. Current
  reference Hillis–Steele scan has logarithmic span but O(TlogT) work; no new
  GPU benchmark or exact full-model hardware acceleration established.
- [BDH2509.26507](https://arxiv.org/abs/2509.26507) reports sparse positive
  activations and measured monosemantic behavior in its architecture. Weighted
  TAC pooling now allows graph influence, but neither sparsity nor nonuniform
  weights guarantees interpretability/nonzero graph sensitivity for every
  input. For identical/zero slots, graph sensitivity can still vanish.
- [Titans NeurIPS2025](https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)
  trains neural memory updates involving gradient-based surprise/momentum.
  A high token-surprise gate is not automatically source truth, relevance,
  safety or the same learning rule; predictable facts can still matter later.

CED is a latent quadratic energy solve. With H=1.25I+.25L positive definite,
C=H^-1(G+.25P) minimizes a corresponding latent quadratic; this does NOT turn
that energy into a verified physical constraint, action outcome, safety score
or independent goal evaluator. Other architectures can also support energy
heads/formal analysis, so the proposed comparison-table entries“None” are not
established. Formal operator proofs are not an unassailable full IEEE/AD moat.
Nominal state bytes exclude weights, working activations, context history,
reconstruction snapshots and recomputation time. No valuation/customer claims.

Bounded increment: learn native keys from observed lexical context, shift the
write to encode observed transitions, and cap only the native retention while
preserving current TAC fast/slow/aligned/concept/wave structure. Exact equations
and scoped Lean/Python obligations accompany source. The new500122p candidate
is UNTRAINED. Source/gradient fixtures and norm envelopes do not establish
better NLL, reasoning, long-term information survival or superiority to SSMs.
Future trained comparisons need source-identical schedules/objectives, active
counts within1%, full compute receipts and a proper same-case P3 reference.
World-model capability remains a research objective requiring authentic
state/action/outcome evidence, not a positioning claim derived from text loss.
