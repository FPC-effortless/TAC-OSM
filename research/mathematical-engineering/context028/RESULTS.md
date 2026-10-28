# CONTEXT028 — complete exact-real chronological graph and prefix causality

This is an additive formulation of unchanged CONTEXT026, not a new model fit or
predictive experiment. Protocol/router 0f1c56dd9c625fb839027034497771fbd90752cf
was published before implementation; CPU20min, zero GPU, READY_NO_FIT and
metadata preflight PASS. Source is pinned to TAC7e40b72 and SHA
c88f3f7c1b628cb733a4a58dcd168a75411d8bb3d80e382068593713e887510a.
Original trained025 checkpoints, numerical failures and prior evidence remain.

PNDS/ContextGraph028.lean defines all42 actual learned tensor fields and their
shapes, owner-local native/aligned/wave/concept/basis states, three lexical lag
embeddings and previous context key. Its complete step constructs positional
and lexical features, shifted native write/current read, aligned prewrite read,
fast/slow representations, concept graph, equilibrium inverse, weighted pool,
LayerNorm,501-wide SwiGLU,256-byte logits and all chronological state updates.
No opaque decoder, externally supplied score or source labels are arguments.
The reset-state clock increments once per consumed byte. Admission suppresses
associative bank writes only; the other observed-state updates continue.

The four named claims concern THIS explicit exact-real graph:

- Equal token/mask prefixes produce equal states after that prefix.
- Logits at t depend only on token/mask observations through t, not future bytes.
- Native no-write admission preserves native memory in the actual full step.
- The first step has no predecessor-key native write.

Compilation/axiom inspection must actually PASS before these are reported as
verified. BUILD_HISTORY.json preserves the first failure (reserved identifier
`local`), and its correction; no assumptions or result thresholds were relaxed.
Source and parameter mapping receipt generation verifies exact pinned Lean,
Python and reciprocal record hashes and compares EVERY Lean learned-tensor
field with the actual untrained PyTorch model, rejecting omissions, duplicate
mappings and dimension mismatches. It does not call model.forward. These checks
are source/formulation engineering, not learned predictive capability.

## Formalization limits that remain explicit

Lean uses real arithmetic and mathematically unshifted softmax; PyTorch's stable
finite-precision kernels are covered only by the separate bounded CONTEXT027
serial/scan numerical checks. The exact-real matrix inverse is totalized by
Mathlib. Proof that this specific CED matrix is nonsingular/has a unique solve,
Lean-to-Python executable correspondence, formal automatic differentiation,
global IEEE bounds and long-history learned information survival remain OPEN.
The new prefix theorem applies to the Lean graph; applying it unconditionally
to arbitrary PyTorch execution would exceed the correspondence evidence.

CONTEX027's original52 cases/all14 authenticated PSF CPython FIT sources and
full42-tensor VJP evidence are preserved, not upgraded to formal IEEE proof.
There is no new DEV NLL, additional source split or protected TEST. Candidate
500122p remains UNTRAINED, with applicable prospective scale/data/training-
compute/matched SSM gates unresolved. The original499978p model's genuine
negative NLL2.099349221 vs practical P3 1.877306112,0/5 wins, remains unchanged.
Internally measured/formulated, not independently qualified. No safety/reward
certification, GPU quota, promotion, merge/deployment or hosted customer inference.
