# Audit — TACOSM-PLM-INTEGRATED-E2E-001

## Status: INVALIDATED / PROVENANCE ONLY

The predecessor experiment is void for benchmark validity.

The q2 ground-truth answer was computed from the second observed entity
(`entities[1]`) but the q2 query tuple returned `entities[0]` as its target
entity. The model therefore received a query naming one entity while being
graded against a target derived from another entity.

This is a benchmark-generation defect, not a negative capability result.

The historical artifact and its headline 0.5200 mean must not be used for
model selection, tuning, baseline selection, capability claims, or rejection
of the PLM hypothesis.

The corrected successor is `TACOSM-PLM-INTEGRATED-E2E-002`.

The old runner must never be used for confirmatory measurement.
