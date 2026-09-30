# TACOSM-C5-COVERAGE-FRONTIER-001 RESULT

Status: VOID.

## Provenance

- GitHub Actions run: 36789410610
- Measurement commit: 4faf16fb31e7ff165fd9fd96d02c8f6ac61a0e4e
- Full artifact: TACOSM-C5-COVERAGE-FRONTIER-001-full
- Artifact id: 11131500534
- Artifact digest: sha256:c009b92334055c04fdee1c0f74ad7e5b0a59819ccc678705558982f991bf0a65
- Preconditions: passed
- Full runner: completed successfully

## Disposition

The experiment is voided for its preregistered capability-constrained decision.

The runner measured all registered factor-size/beam configurations, but the
capability endpoint is not discriminative with respect to state identity.

## Validity failure

The registered `EndToEndExecutor` only contributes when a candidate descriptor
matches the reference, then computes the contribution from the candidate's
fixed executable structure. In this workload, the four candidates associated
with every state code have the same aggregate output. Therefore different
state values can produce the same expected end-to-end result.

This creates a real failure mode in which selective end-to-end success exceeds
the exhaustive selector even when the selected persistent state is not the
target. The measured `capability_retention` therefore cannot be interpreted
as retrieval-capability retention.

This is an endpoint-validity failure, not a model failure.

## Retained measurements

The following measurements were produced before the validity failure was
recognized and remain useful as diagnostic observations, but they do not
license the preregistered frontier decision:

- `states_scored_over_M`;
- `proposal_target_retention`;
- `selective_target_recall`;
- arithmetic estimates.

The candidate-fraction surface reproduces the expected granularity/beam
tradeoff, but it is not used to select a capability-constrained configuration.

## Non-promotion rule

No capability-constrained C5 frontier is promoted from this run.

No universal retrieval, sublinear-compute, semantic-addressing, or hardware
speedup claim is changed by this result.

## Corrective experiment

The next experiment will preserve the same factor-size/beam surface but replace
the non-discriminating executor with a fixed state-distinct downstream task.
A successful output must depend on the selected persistent-state value, so a
wrong state cannot pass solely because another state happens to produce the
same aggregate output.

That corrected experiment will be registered separately rather than rewriting
this voided record.
