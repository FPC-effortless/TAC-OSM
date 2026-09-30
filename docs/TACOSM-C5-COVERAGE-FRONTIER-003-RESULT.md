# TACOSM-C5-COVERAGE-FRONTIER-003 RESULT

Status: VOID.

## Provenance

- GitHub Actions run: 36790451982
- Measurement commit: e92dd4ea224d1486ad7f49faffb6891a94b3de48
- Full artifact: TACOSM-C5-COVERAGE-FRONTIER-003-full
- Artifact id: 11131032708
- Artifact digest: sha256:a5021b5d55830ab1679a8490f6b4bd91fa80cec1930635aa8d2a00cec323fa0b
- Preconditions: passed
- Full runner: completed successfully

## Disposition

The experiment is voided for its preregistered capability-constrained frontier decision.

The state-distinct downstream executor passed its explicit validity audit. However, pooled success counts were averaged across seed cells and then converted to integers before the retention ratio was computed. For example, 5.4 exhaustive successes and 4.2 selective successes became 5 and 4, yielding 0.80 instead of 4.2/5.4.

Thus the measured capability-retention values and frontier eligibility are numerically distorted by truncation.

## Retained observations

The artifact contains valid diagnostic observations for state-scored computation, proposal coverage, selective target recall, state-distinct exhaustive/selective success, and arithmetic cost.

These observations do not promote a capability-constrained C5 frontier.

## Non-promotion

No capability-constrained frontier is promoted from this run.

No universal retrieval, sublinear-compute, semantic-addressing, or hardware-speedup claim is changed.

## Corrective experiment

The next registration will aggregate integer success counts by summation across seed cells, aggregate evaluation counts by summation, and compute the capability ratio directly from those totals without any rounding or truncation. A runner-level invariant will assert that every pooled count equals the sum of its constituent cell counts.
