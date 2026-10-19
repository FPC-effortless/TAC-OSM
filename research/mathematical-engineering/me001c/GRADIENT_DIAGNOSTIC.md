# ME-001C — untrained long-horizon gradient diagnostic

**Date:** 2026-10-19. **Evidence:** CPU PyTorch engineering-only, independent of real SCANIA TRAIN or reserved TEST. **Architecture:** the original ME-000 native bounded TAC recurrence, not the new ME-001B sensor-feature router, and NOT an SSM.

Three fixed seeds; compute the gradient L2 of a random terminal linear state projection with respect to an untrained initial recurrent state. All inputs observed; lengths 8, 32, 128. No data fitting or outcome labels.

| Seed | L8 | L32 | L128 |
|---|---:|---:|---:|
| 20261019 | 0.071177326 | 4.49654e-6 | 0 (float32 underflow) |
| 20261020 | 0.061138149 | 1.01185e-4 | 2.68900e-21 |
| 20261021 | 0.184473127 | 6.08255e-6 | 2.11151e-17 |

ME-000 boundedness is a **forward-invariant amplitude property**, not a guarantee of backward gradient transport or useful long-history state memory. This numeric demonstration motivates retention-path hypotheses but is not real-world predictive evidence or an architecture failure verdict. No performance rank is inferred.

**Next candidate comparisons (not executed):** identity-first residual path; orthogonal/near-identity persistent subspace; learned slow updates; separate explicitly persistent information store; long-horizon credit losses. Baseline recurrent GRU and exact state machine remain mandatory. Compare gradients first as engineering diagnostics, but only real temporal tasks establish scientific advantage.

Separate original diagnostic source and raw JSON are in conversation-local research package. This GitHub note reproduces recorded summary values only; it is not an independent replay. Do not misattribute to successful ME-001B training.
