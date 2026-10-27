# PAR-MEM-010: full-model proof extension

Source remains TAC numerical revision
`63ea0aa67904fa946b9baa69d2e4b0cc23dc1c9a`. No learned graph, weights,
checkpoint, original research outcome or six-family registry is changed here.

## Exact execution

`SourceAlignedScan.blelloch_all_prefixes` proves **every inclusive leaf** of
a balanced chronological upsweep/downsweep, not just its final summary. The
proof permits noncommuting affine operators. `padded_blelloch` proves cropping
after power-of-two padding. The coordinate-offset and value-row theorems
cover the permuted four-block source-memory layout.

`SourceAlignedExecution.full_model_state_after_scan` proves equality of the
entire reconstructed mathematical state and streaming state, for arbitrary
finite histories and parameters. An independent prepass erases both associative
stores, prepares the actual learned schedules from the remaining source context,
and scans native and aligned memories together. The reconstruction includes
clocks, write history, basis sum, ownership and chronology metadata.

`full_prediction_after_scan` includes the **complete nonlinear decoder**, the
temporal backbone, post-scan FFN, associative read routing and predictive memory
residual. This closes the former gap between a memory-only affine theorem and
a whole default-model mathematical output theorem. It is a theorem about the
explicit real graph; it is not a Python/Torch compiler correctness theorem.

## Full-network differentiation

`SourceAlignedDifferentiation.forward_ad_correct` proves the scalar AD evaluator
for addition, multiplication, negation, inverse, exponential, logarithm, square
root, sine, cosine and maximum at explicitly regular points. Dense maps,
sigmoid, tanh, SiLU, softmax and epsilon normalization use this vocabulary.

`SourceAlignedPrograms.complete_graph_compiler_correct` proves interpretation of
the **whole typed neural graph** into that language. Every parameter tensor,
floating input and carried real state has an expression slot; source owners,
ordinals, counts, masks and query choices are fixed discrete metadata.
`all_history_compiler_correct` extends interpretation to arbitrary finite
histories, retaining shared parameter uses.

`all_history_ad_correct` proves the full graph JVP for simultaneous parameter,
input and state curves. `vjp_pairing` establishes the transpose/JVP identity.
`all_history_vjp_correct` proves the resulting weighted-loss pullback for the
entire unrolled model, including all earlier events and shared parameter uses.
No assumed memory derivative or opaque learned decoder is substituted here.

The derivative hypotheses are deliberate. A clamp has no classical two-sided
derivative at its tie: `clamp_not_differentiable_at_zero` proves this. Its two
one-sided derivatives are proved separately. Normalization is compiled with
the equivalent denominator `sqrt(max(sum(x*x),epsilon^2))`. This avoids a false
sqrt-at-zero singularity: `normalized_zero_regular` and
`normalized_zero_tangent` prove the zero-key Jacobian is I/epsilon.
`normalized_epsilon_boundary_not_regular` identifies the genuine floor kink.

The tested CPU Torch 2.14.1 runtime selects the **constant branch** at a
`clamp_min` tie (gradient zero). Its zero-key normalization Jacobian also matches
I/epsilon. These are versioned runtime observations; a chosen subgradient is
never promoted to a nonexistent classical derivative. At the floor boundary,
the symbolic selected branch matches this recorded convention, while the
classical AD theorem correctly excludes the kink.

## Numerical error mathematics

`SourceAlignedRoundoff` proves explicit forward error propagation for add,
negate, multiply, maximum, inverse, sqrt, exp, log, sin and cos. The inverse/log
theorems require separation margins. The exp theorem states its perturbation
range. The sqrt bound is a Holder bound and does not assume differentiability
at zero. Clamp/max forward stability holds even at ties.

`rounded_sum_error` includes initial error, exact partial-sum magnitudes,
relative unit-roundoff allowance and additive underflow allowance. Overflow
exclusion is a domain premise. The result neither assumes rounded arithmetic
is associative nor asserts a history-independent fixed absolute tolerance.

**Backend accuracy remains a premise.** These are kernel-checked mathematical
implications from local primitive error certificates. They do not establish
the accuracy of a particular libm, BLAS reduction, FMA arrangement or Torch
autograd implementation. There is no hidden backend-correctness axiom.

## Source checks and falsifiers

The independent NumPy extension prepares schedules with both associative stores
erased, runs a separate balanced five-block native+aligned scan, reconstructs
each prefix and invokes the complete independent decoder. It adds 61 source
comparisons in FP32/FP64 over T=1,3,7,16,33,129 and all/none/mixed writes.
The earlier 467 source/gradient comparisons are retained. Exact Fraction tests
cover every downsweep leaf, padding, nonzero initial state and every tensor
coordinate. Scalar-program tests check JVP/VJP pairing, shared uses, zero keys,
clamp ties, nonlinear error bounds and subnormal/cancellation accumulation.

The new source runner also preserves two concrete counterexamples:

* Finite FP32 source-key rotations at the largest finite value can overflow the
  address computation; valid finite observations and parameters alone do not
  guarantee a finite prediction. The source rejects the nonfinite schedule.
* In the **actual streaming model**, a physical clock at 2^24 followed by a
  delta of one stays at 2^24 in FP32. The exact-real clock advances by one.

These refute an unrestricted all-finite numerical domain or a universal small
absolute state-error bound. They do not invalidate the qualified real and
roundoff theorems.

## Remaining implementation trust boundary

The exact mathematical graph, its full-history scan execution, and its
regular-point JVP/VJP are proved. The numerical error implications are proved
under their visible local accuracy and domain hypotheses. The source refinement
evidence is numerical and source-hashed.

**Complete formal verification of the deployed Torch program is still not
established.** It requires a formal source/runtime correspondence plus verified
local CPU kernel/transcendental accuracy contracts on a bounded admissible
deployment domain, including exceptional behavior. The existing release flag
therefore stays false. Passing fixtures, adding proof counts, or treating
Torch as an unstated axiom cannot close this boundary.

Source observation membership and permissions remain external provenance
requirements. Trained retrieval/forecasting advantage remains an experimental
gate. This deterministic point regressor has no claimed calibrated probability
law or interventional causal semantics to certify.
