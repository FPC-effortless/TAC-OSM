# DEV-035: frozen recurrence dynamics

Registered diagnostic, not retraining or a repaired performance result. Preserve
DEV-034's failed original pilot and separate numerical continuation. Inspect
the initial independent-retention model, pre-update140 failure snapshot, final
FP64 continuation, and completed local-tied/global-multi/global-tied checkpoints.
Checkpoint hashes come from archived DEV-034 inspection reports.

Use only three predetermined **training** batches: steps0,139,215. Each has two
H512 histories and the original final inspection receipt. No validation/test
selection or new scientific test generation. Trace513 transitions per history
with original batched FP32 operations. Reconstructed transitions must be bitwise
equal to the original proposal and complete encoder-plus-receipt update. Preserve
any non-finite forward failure and report incomplete coverage rather than dropping it.

For each expert's value-coordinate row, fixing input and responsibilities gives

`m' = alpha*m + beta*r*(v - dot(m,k))*k`.

The exact conditional difference operator is `A = alpha*I - beta*r*k*k^T`.
Its Euclidean operator norm is
`max(abs(alpha), abs(alpha - beta*r*||k||^2))`.
For nonnegative beta/r and0<alpha<1, feedback above1+alpha amplifies a direction.
Measure these coefficients in FP64 from the actual FP32 transition coefficients.
Count gains above1+1e-6 as amplifying to avoid interpreting rounding near one.
This algebra holds routing fixed, despite actual responsibilities depending on
memory. It is not the full nonlinear Jacobian or a global stability theorem.

Also compute the actual96-dimensional augmented-bank state Jacobian (72 memory
coordinates plus24 responsibility coordinates) in FP32, with norms/SVD in FP64.
Preserve original batch shape while perturbing only the selected bank. Compute
both ambient gain and gain restricted to the responsibility simplex tangent:
responsibility perturbations sum to zero across experts for each coordinate.
The tangent basis is orthonormal. This avoids confusing an off-simplex direction
with a valid responsibility perturbation. Euclidean gains still depend on the
chosen relative scaling of memory and responsibility coordinates.

Predetermined sampling: positions0,255,511,512 plus the maximum conditional-row
gain position in each history; deduplicate overlaps. In global-writer controls,
select the bank with largest conditional-row gain at that position. These
selection rules bias the Jacobian sample toward large conditional gains. They
are not an unbiased Jacobian distribution, architecture ranking or independent
seed sample. Per-event conditional gains cover every written bank and row.

Report memory-to-memory and responsibility/memory cross-block norms, checkpoint
and trace hashes, actual Jacobian tensors, missing coverage and direct-replay
identity. Independently re-evaluate SVD summaries and counts from saved matrices
and per-event trace evidence. All six snapshots remain in the report.

A one-step gain above one does not prove long-product expansion: directions may
rotate or contract later. No parameter-gradient factorization, full backward
causal attribution, information-loss claim, trained intervention advantage or
formal Lean/PyTorch correspondence is established. Five algebra/Jacobian contract
tests check the implementation, not learned capability. Use the result to choose
the next prospectively registered intervention; do not rewrite DEV-034 outcomes.
