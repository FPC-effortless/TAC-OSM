# Audit — TACOSM-PLM-INTEGRATED-E2E-001

## Status: INVALIDATED / PROVENANCE ONLY

The predecessor experiment is void for benchmark validity.

The q2 ground-truth answer was computed from entities[1] while the q2 query
tuple advertised entities[0] as its target entity. The model therefore
received a query naming one entity while being graded against an answer
derived from another entity.

This is a benchmark-generation defect, not a negative capability result.

The historical artifact must not be used for model selection, tuning, baseline
selection, capability claims, or rejection of the PLM hypothesis.

The corrected successor is TACOSM-PLM-INTEGRATED-E2E-002.

The old runner remains in the repository only for provenance. Confirmatory CI
must never execute it.
