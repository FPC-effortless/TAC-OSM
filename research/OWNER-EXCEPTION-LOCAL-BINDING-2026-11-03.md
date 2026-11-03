# One approved fixed-case diagnostic

The owner explicitly approved one synthetic-training exception and its bounded
trial on 3 November 2026, in response to the question identifying three roughly
100k models, 128 updates each, and aggregate 360 CPU-second / 600 wall-second caps.
Approval was conditional on source and runtime checks. The direct response was
"Yes". The scope and conversation provenance are recorded in
`local_binding_owner_fixed_20261103/OWNER_AUTHORIZATION.json`.

This is a prospective exception to the synthetic-only fitting prohibition and
the no-fit diagnostic category for **this fixed/default matrix only**. It creates
no general synthetic-fit permission. The approved experiment uses seed18001,
unchanged GRU/phase/context models, 128 updates per arm (384 total), default AdamW
epsilon1e-8, all256 output bytes, unchanged cases/exposures/loss/checkpoints and
no tuning. Active/deployed 95k–105k bounds and within1% arm matching still apply.

The old validator and planner rejections remain in the receipts. They are not
silently turned into real-data or signed-qualification passes. The new scalar
gate authenticates the exact approved scope and preserves those old results.
The unchanged planner selects GitHub Actions CPU; its decision is not admission.

All generated-data results are finite-case optimization diagnostics, without
architecture-superiority, reliable-binding, real-world, independent-qualification,
safety, commercial-readiness or promotion claims. The original real-data and
independent-qualification gates remain unchanged. Protected TEST, GPU, paid
service, merge, retry, a later stage or another optimizer regime are not approved.

The new branch adds an explicit admission path; historical C12/C18 source, policy,
registrations, checkpoints and all20diagnostic/15historical numerical failures
remain untouched. Model graphs are unchanged, so no new mathematical variant or
source-equivalence proof is claimed. Existing local/scoped proofs do not close
full graph/IEEE/AD obligations.

The released workflow must run source/fixture/scorer/approval tests, runtime
version checks, live parameter checks and initial same-weight numerical/causal
checks before optimizer work. Final checkpoints, optimizer/RNG state, attempted
and completed updates and owned-process accounting are retained. A partial or
failed matrix is not a winner and does not authorize a retry.
