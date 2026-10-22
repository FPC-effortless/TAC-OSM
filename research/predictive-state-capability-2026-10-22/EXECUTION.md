# Execution record — 2026-10-22

Source-only implementation on governance PR #318 head
`fbd293da5bcfa243c71966f4b92a04f43574fdea` (based on master). The 887-test count
is for this branch, not a claim to have rerun every test in all historical
research worktrees. No original experimental evidence is changed.

- Publisher OBD example: actual download at the commit/URL in
  `OBD_SOURCE_AUDIT.json`; full SHA-256 verified. CSV header parsed, no reward
  values parsed/scored, no entity identity inferred. Audit returns exit 2 for
  entity-memory eligibility and records the actual missing identifier.
- Preflight on `PARITY_DRAFT.json`: expected exit 2, original error in
  `PREFLIGHT.txt`. No optimizer is created by the audit or validator.
- 17 new numerical/source tests; all pass. Synthetic tensors exercise numerical
  contracts only. No generated benchmark or manufactured decision outcomes.
- 11 inherited compulsory scientific-parity tests; all pass.
- Full canonical suite: `887 passed in 13.67s` with Python 3.12.14/Torch 2.14.1+cpu
  in the local workspace. Command: `PYTHONPATH=src python -m pytest tests/ -q`.
  Startup environment: `ATEN_CPU_CAPABILITY=avx2`,
  `MKL_ENABLE_INSTRUCTIONS=AVX2`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`. No CPU latency or FLOP-parity claim is derived from
  test runtime.
- The first full run found only the stale baseline test-count provenance row:
  886 passed, one failed. The prescribed sync script changed the current count
  from 870 to 887, preserving frozen count 327; targeted checks and the full
  suite then passed. No model behavior or tolerance changed to fix a test.
- `python scripts/sync_test_count.py --check`: pass.
- `git diff --check`: pass.
- Untrained state-dict save/load numerical replay: exact on the same runtime;
  no trained checkpoint archive or cross-ISA replay claim.

Model fits: **0**. Trained optimizer updates: **0**. Newly scored heldout rows:
**0**. Measured predictive/decision lift: **not available**. Dataset and actual
FLOP/CPU parity qualification remain open; the training gate has not passed.
