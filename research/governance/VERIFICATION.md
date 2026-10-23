# Enforcement verification — 2026-10-23

Scope: software governance only. No authentic-data model was trained, resumed,
retuned or promoted. Optimizer integration tests use labeled software fixtures.
No original trained checkpoint, scientific score or historical registration changed.

Executed locally on the default-branch implementation:

| Check | Observed result |
|---|---|
| `python -m pytest -q` | 978 passed, 14.46 seconds |
| `python scripts/audit_training_budget.py` | PASS, 88 Python files audited |
| `python scripts/verify_frozen_replay.py --all` | PASS, six pinned registrations, 15 distinct original checkpoints (each checked for prefit/final) |
| `python scripts/sync_test_count.py --check` | PASS |
| `ruff check --select F821` on new enforcement modules, validator and integration tests | PASS |

The new focused tests reject 40k/94,999/105,001/150k declarations through all
four fitting routes at both phases, verify actual whole-model counts override
manifest claims, include frozen weights, reject recycled historical IDs, verify
the durable pre-update receipt and checkpoint tensors, and reject unregistered
optimizer updates. Existing data/compute arithmetic tests remain separate from
the unmocked runtime-integration tests. A reused v2 label cannot restore CPU parity.

`FROZEN_REPLAY_AUDIT.json` preserves the actual immutable-Git-byte audit output.
This is not a replay of scientific forecasts. Original scores remain historical.
CI status and branch-protection enforcement must be checked separately; local
passes do not establish default-branch integration or mandatory merge protection.


## ME073 compatibility overlay

The default-branch proposal is PR #354. This overlay applies the same guard to
ME073's additional robot/event fitting entrypoints, including its capability
lane. No original registrations, source receipts, forecast arrays or checkpoint
bytes changed. Current guarded fitting rejects incomplete historical manifests;
no new training is authorized by their old IDs or their replay records.

Executed locally on this overlay: **1,103 tests passed in 19.09 seconds** and
**164 Python source/entrypoint files passed the guard audit**. Counts differ
because the default baseline has 924 tests and ME073 has 1,043 before this work;
these are different branch suites, not a deletion of scientific tests. Original
ME057 tests now preserve its recorded receipt while rejecting prospective reuse;
the capability-lane software fixture uses a compliant declared size.

A literal-newline syntax defect inherited in ME066's archived runner was
repaired in the guarded working copy so source coverage can parse it; the
pinned original is preserved. No original dynamics or training recipe changed.
The adapter migration audit compares normalized source ASTs against the pinned
snapshot and rejects model/recipe changes or scientific-evidence edits. Its CI
lane authorizes no fitting; substantive future changes still go through normal
prospective registration validation.

ME071–ME073 replay workflows verify pinned provenance, restore the original code
in a temporary worktree and install a global optimizer-update rejection hook.
The working branch's old numerical registrations are never relabeled as newly
qualified. Older experimental qualification/fitting lanes require explicit
manual opt-in and cannot bypass the new live guard. Automatic root and policy
checks remain active.
