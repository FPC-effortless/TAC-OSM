# ME073 engineering check ledger

- Initial nine new native trajectory math/causality tests passed.
- First full regression:1,040 passed, one stale current-test-count row failed;
  synced the allowed current count, preserving frozen327. Full1,041 then passed.
- Added updated master100k-scale guard and truthful derived scale sidecar during
  FIT. Full1,042 tests passed at an intermediate revision.
- Extending absolute scale checks to comparison fixtures exposed12 older10k
  synthetic-fixture failures at7f08bfe2. They were corrected to100k fixtures,
  preserving the1% mismatch and compute/CPU tests; no real model was resized
  or retrained. Current targeted25 architecture/governance checks passed.
- Added readout eligibility/EOR integration check and preserved its result.
  Final full local regression: **1,043 passed**,18.22s; original output
  preserved in REGRESSION.log. Independent GitHub run39948180370 passed original-model full real-score/
  generated-path replay with no tolerance change or refit. No unit fixture is real-data evidence.

Actual original training continues under fixed six epochs/five seeds. These
software corrections do not change source, model, objective, FIT order or
original frozen forecast protocol. No REPORT label was used for them.
