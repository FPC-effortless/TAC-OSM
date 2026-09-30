# TACOSM-C5-COVERAGE-FRONTIER-002 RESULT

Status: VOID.

## Provenance

- GitHub Actions run: 36789999312
- Measurement commit: 8e47ce52a67dbda419cf9316456f6fe8d86bbe93
- Full artifact: TACOSM-C5-COVERAGE-FRONTIER-002-full
- Artifact id: 11131012112
- Artifact digest: sha256:ab406dd63773d3ee3e34f53eaa8c8df1eeb1bfe8ad69dda4b1d87ea58bec1d0b
- Preconditions: passed
- Full runner: completed successfully

## Disposition

The experiment is voided for its preregistered frontier decision.

The corrected state-distinct executor passed its explicit validity audit, but the runner used inconsistent quantities for frontier eligibility: the reported `capability_retention` was recomputed from pooled selective/exhaustive success rates, while `meets_capability_floor` was evaluated from the pre-update mean of per-cell ratios.

As a result, some rows are internally inconsistent (for example, a row can print capability retention above 0.80 while being marked ineligible, or vice versa). The frontier selection therefore does not correspond to the preregistered decision rule.

## Retained observations

The artifact contains valid diagnostic measurements of candidate-set computation, proposal coverage, selective target recall, state-distinct exhaustive/selective success, and arithmetic cost. These are not used to promote a capability-constrained frontier.

Notably, the state-distinct executor audit itself passed, so the predecessor's non-discriminating-executor problem was actually corrected. The remaining failure is a statistical aggregation/eligibility implementation error.

## Non-promotion

No capability-constrained C5 frontier is promoted from this run.

No universal retrieval, sublinear-compute, semantic-addressing, or hardware-speedup claim is changed.

## Corrective experiment

A new preregistration will compute pooled success totals first, derive exactly one pooled capability-retention value from those totals, and derive the eligibility flag from that same scalar. Frontier selection will consume only that stored scalar.

The voided experiment remains unchanged and auditable.
