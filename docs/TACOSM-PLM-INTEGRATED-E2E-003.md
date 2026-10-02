# TACOSM-PLM-INTEGRATED-E2E-003

## Purpose

E2E-001 and E2E-002 cannot be used as scientific evidence because their shared
episode generator ignored the supplied query entity and returned the first
episode entity for every query.

E2E-003 is the corrected temporal/addressing experiment. It preserves the
architecture and intervention question while independently constructing the
benchmark and enforcing distinct q1/q2 target entities.

## Corrected temporal structure

Each episode has three entities.

q1 targets entity 0.

q2 targets entity 1.

Therefore the post-action q1 update and the q2 query are temporally separated
and target different persistent slots.

The benchmark generator asserts this invariant during construction and the test
suite checks it independently.

## Arms

**explicit_write:** writes use the structural observation entity; reads remain
learned.

**explicit_read:** reads use the structural query entity; writes remain learned.

**explicit_both:** writes and reads use the structural entity identity.

The arms are diagnostic interventions, not a ranking exercise.

## What remains fixed

The multimodal encoders, 12-bit complementary payload, held-out compositions,
optimizer, 300 training steps, five seeds, 400 evaluation episodes, CASM,
verifier, and learning objective are unchanged from the corrected intended
design.

The invalid E2E-001 and E2E-002 numbers are not used to choose hyperparameters,
seeds, or intervention decisions.

## Scientific boundary

A recovery under explicit addressing shows that the corresponding learned
address path is a bottleneck. It does not prove learned semantic addressing.
Failure of explicit addressing would shift investigation upstream toward
representation/task learning or downstream toward CASM.
