# Enforcement verification — 2026-10-23

Scope: software governance only. No authentic-data model was trained, resumed,
retuned or promoted. Optimizer integration tests use labeled software fixtures.
No original trained checkpoint, scientific score or historical registration changed.

Executed locally on the default-branch implementation:

| Check | Observed result |
|---|---|
| `python -m pytest -q` | 979 passed, 14.66 seconds |
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

The initial smaller GitHub CI environments exposed an accidental NumPy dependency
in tensor hashing (four integration failures). Hashing now uses Torch uint8 bytes
and the standard library; an explicit no-NumPy regression test passes. The full
local suite was rerun after this fix. Earlier failed workflow runs remain visible.
