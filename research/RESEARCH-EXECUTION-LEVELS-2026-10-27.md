# Research execution levels

Operator amendment, 2026-10-27. This supersedes blanket external-signature
requirements for ordinary internal research; it does not relax scientific
fairness, source integrity, checkpoint replay or protected TEST policy.

| Level | Execution | Evidence and claims |
| --- | --- | --- |
| `internal_research` | Genuine, publicly licensed, authorized FIT/DEV diagnostics, exploratory evaluations and correctly preregistered research | Internally measured, **not independently qualified**. No external signers required. |
| `independent_qualification` | Protected evaluations, reward-integrity certification, safety claims and checkpoint promotion where required | Existing external verification, evaluator isolation and qualification gates remain compulsory. |

Internal studies explicitly declare `execution_level: internal_research` in
their manifest and phone proposal. They record publisher, license and evidence,
original file hashes/source receipt, frozen split/cohort identities, causal
protocol and immutable checkpoint/replay policy. Runners must verify the
actual data, features and checkpoint bytes; metadata is not source proof.
`research_scope` permits only authorized FIT/DEV and explicitly denies protected
evaluation, reward certification, safety claims and promotion. Missing metadata
or inconsistent actual source/split/checkpoint evidence blocks execution.

All 95k–105k active/deployed model bounds, <=1% trained-arm parameter parity,
original data/exposure, training/tuning budgets and evaluation conditions remain
mandatory. Preflight and final actual-resource checks still apply to fits.
Internal mode cannot turn a failed fair comparison into a valid one. CPU speed
is an efficiency observation, not a fairness equality requirement.

The router defaults legacy proposals to independent requirements, supports an
explicit internal level, and never auto-submits evaluation/training or grants
TEST access. `validate_research_parity.py` always checks scientific resources
before applying the execution-level evidence policy. Explicit internal studies
receive `NOT_INDEPENDENTLY_CERTIFIED_INTERNAL_RESEARCH`, never a reward audit
certificate. `run_qualified_research.py` remains a strict qualification launcher
and rejects internal manifests. Its signing, scorer/goal, ACL and checkpoint
promotion checks are retained.

PR #483's historical audit defaults to its registered internal level. Its
`--execution-level independent_qualification` mode still requires external
receipts. The internal path verifies the publisher/license record and frozen
archive, code, cohort and checkpoint identities without signer keys. The
existing original H64 metric, cases, seeds, controls and thresholds are unchanged.

Keep failures, negative findings and INCONCLUSIVE checks. Label real internal
measurements separately from constructed software fixtures. Internal results
may motivate another preregistered experiment; they do not certify safety,
causal intervention utility, reward integrity or independent generalization.
Original historical records and prior blocked preflight receipts remain intact.
