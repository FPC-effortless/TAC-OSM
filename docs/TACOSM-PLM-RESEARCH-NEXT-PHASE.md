# Unified PLM Next Research Phase

The unified substrate passed P1. The next experiments are orthogonal rather
than cumulative feature stacking.

## P2 — State stability

Question: how does verifier error alter persistent-state contamination?

The experiment directly submits correct and incorrect experience proposals to
the same persistent store under strict verification, injected false
acceptance, and no-verifier control. It measures wrong verified writes rather
than inferring contamination from task accuracy.

## P3 — Computation addressing

Question: can the computation/operator population E grow without making every
query execute a full scan?

The operator population is randomly keyed, queried with controlled noise, and
resolved by a multi-level index. Raw bucket work is charged separately from
the admitted candidate cap.

The empirical break-even point is the measured E where indexed address work
drops below the full-scan baseline.

## P4 — Continual specialization

Question: does separating slow shared updates from fast specialist residuals
reduce interference during sequential domain shifts?

Three arms are measured:
- global fast updates;
- specialist-only fast updates;
- slow shared + fast specialist updates.

Training covers domains 0,1,2. Domain 3 remains held out until evaluation.

## Promotion gates

P2 requires a measurable relationship between verifier error and wrong
persistent writes.

P3 requires target-admission measurement and explicit indexed-vs-scan cost.

P4 requires retention and adaptation to be reported together and held-out
performance to be preserved before specialization is promoted.

No phase reopens settled C5 representation selection.
