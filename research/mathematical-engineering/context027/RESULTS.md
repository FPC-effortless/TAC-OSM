# CONTEXT027: full numerical graph correspondence, no training

This extends existing draft CONTEXT026, not a new architecture, fitted model or
predictive experiment. Original model source SHA c88f3f7c1b628cb733a4a58dcd168a75411d8bb3d80e382068593713e887510a
and all original LANGTRAIN025 checkpoints/results remain unchanged.

The preregistered question is whether the implemented scan predictor has the
same complete bounded-case forward and reverse-mode behavior as a separately
written chronological predictor. Protocol/router a0e1013cd424db1b3a77dfb651a7d9e90bf1fa84
was published before implementation/execution. Initial invalid router kind and
missing suite metadata were blocked without executing the experiment; both
blocked receipts are preserved. Final route is READY_NO_FIT on GitHub CPU.

PNDS `foundation/v1/context026_graph.py` imports no TAC code. It takes explicit
parameter tensors and observed bytes, computes all operators chronologically,
and does not consume source traces, source helper functions, or scan schedules.
Its native and aligned banks use direct residual-form outer-product updates;
wave and concept recurrences use explicit steps. It independently constructs
position/ordinal codes, causal lexical features, context means, fast/slow
features, symmetric concept Laplacian, equilibrium solve, nonuniform pooling,
LayerNorm, SwiGLU and complete 256-byte decoder. It shares PyTorch primitives,
linear algebra and autograd with the implementation: it is **not** an independent
AD engine, an IEEE proof, or a whole-model Lean proof.

Local pinned-reference run passed all 52 cases: two precisions, lengths 1/3/9/17,
batch2, all/none/alternating admissions (24 constructed software fixtures), plus
all14 original PSF-licensed CPython FIT documents, first17 source bytes in both
precisions (28 genuine-input engineering checks). Every logit and every returned
state trace is compared. All42 parameter tensor VJPs are compared with a fixed
sine-weighted logit scalar; no labels, predictive NLL or optimizer are involved.
Zero or disconnected VJPs are recorded, not advertised as useful information.
Authentic source licenses/hashes and immutable candidate source are verified.
No model weights changed.

| Precision | Maximum forward/state absolute error | Maximum parameter VJP absolute error |
|---|---:|---:|
| FP64 | 1.055e-15 | 2.603e-18 |
| FP32 | 2.683e-7 | 1.281e-9 |

All differences satisfy the preregistered elementwise tolerances (FP64
atol1e-10/rtol1e-8; FP32 atol1e-4/rtol1e-3). These are finite-case checks,
not worst-case bounds. Full per-case/per-parameter details are emitted by the
runner as FULL_GRAPH.json; LOCAL_RESULT.json records the local receipt hash and
summaries. Future-suffix poison preserves prediction prefixes and isolated batch
owners agree, for both complete implementations. All-zero write masks suppress
only associative-bank writes; wave/concept states still evolve by design. This
mask is not a full missing-observation/frozen-state interface.

Checker falsification tests reject incorrect forward outputs, identical forward
outputs with reversed gradients, nonfinite values, and Boolean state mismatches.
Those are constructed engineering controls, not authentic model measurements.
The dedicated read-only CI workflow repeats the pinned cross-repository check
and uploads its full receipt; CI results must be checked before claiming PASS.

## What remains open

The Lean module retains its original six scoped exact-real claims; the complete
parent graph is not yet fully formalized in Lean. Formal AD and global IEEE
bounds, long-history trained information survival, matched SSM/data/exposure/
training compute, and any new fit remain OPEN. This untrained500122p candidate
has no DEV NLL or capability claim. Any prospective fit still needs its own
applicable scale and research gates; this no-fit audit grants none.

Original trained499978p LANGTRAIN025 remains a genuine negative: five-seed
exposed-DEV NLL2.099349221 versus practical P3 1.877306112 (0/5 wins), previously
cold-Git replayed at8e3e004. Internal research only, not independent qualification.
No GPU, TEST, promotion, merge, deployment or customer hosted inference.
