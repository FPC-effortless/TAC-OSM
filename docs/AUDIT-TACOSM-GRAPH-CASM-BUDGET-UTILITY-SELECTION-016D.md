# G-CASM-016D — Budget-capped probe objective

Status: PREREGISTERED / IMPLEMENTATION.

016D directly tests the conceptual shift from Shannon information to the
terminal verifier's actual budget. Both arms use the same six-bit activation
trace and the same target-blind one-step action domain. Only the action
objective changes.

For a candidate partition with M hypotheses and terminal budget B:

U_B = sum_e min(B, |H_e|) / M.

The objective is the expected utility gain over the current no-probe terminal
baseline, divided by acquisition work.

The primary question is whether this changes downstream hypothesis resolution.
Because G-CASM-015 already shows the trace partition is near its finite-domain
collision floor, the preregistered hypothesis explicitly allows a near-zero
effect.

No submodularity claim is made here. Adaptive submodularity is a separate
property requiring proof under the exact observation model.
