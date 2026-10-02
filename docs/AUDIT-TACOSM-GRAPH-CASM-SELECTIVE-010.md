# Scientific Audit — TACOSM-GRAPH-CASM-SELECTIVE-010

## Audit status

**Pre-confirmatory disposition: CONDITIONAL — blockers identified and corrected in the runner; confirmatory evidence remains pending.**

This record is part of the experiment provenance. It does not contain outcomes and cannot be changed in response to confirmatory results without a new experiment identifier.

## Benchmark validity

- Unit of analysis: one target program/task at a registered population M.
- Primary populations: M = 32, 64, 128, 256, 512.
- Seeds: 0–4.
- Target programs are excluded from router training by exact structure and complete truth-table signature.
- Candidate pools reject duplicate target structure and target truth signature.
- Four support rows are router-visible; twelve complementary rows are verifier-only.
- The exhaustive exact graph executor is independently checked on complete 16-row truth tables.
- The primary endpoint is a capability-constrained ratio, not a raw routing score.
- No asymptotic claim is licensed.
- A materially failing exhaustive ceiling invalidates a selective verdict at that M.

## Leakage audit

| Channel | Boundary | Check |
|---|---|---|
| target index | evaluator only | target index is stored in Task and never passed to router query/candidate features |
| target ID | evaluator only | task address derives from public support examples; ID is not a router feature |
| verifier-only rows | evaluator only | remaining 12 truth-table rows are passed only to execution/verifier |
| complete truth table | evaluator only for evaluation target | target truth signature is excluded from router training and decoy construction |
| true wiring | public intervention | graph arm intentionally exposes executable wiring; summary arm removes the wiring field |
| candidate generation | before routing | decoy inclusion is not conditioned on router-visible support |
| candidate ordering | deterministic tie-break only | routing score depends on candidate content; order is not an input feature |
| post-result adaptation | prohibited | primary benchmark parameters are fixed before confirmatory measurement |

## Protocol audit

- Contract is preregistered.
- Population, seeds, budgets, steps, capability threshold, executor, and endpoint are fixed.
- The registered NOT/XOR secondary-role holdout is now explicitly constructed from targets requiring the NOT→XOR role pair.
- Primary train/evaluation structure and truth-table disjointness are checked before measurement.
- Persistent-experience replay is implemented as a secondary before/after measurement with three unrelated writes.
- The primary minimum-over-budgets endpoint receives a seed-level bootstrap that repeats the budget-selection rule.
- Operational workflow changes do not alter the scientific grid.

## Implementation audit

Primary execution chain:

contract -> task generation -> representation -> router -> ranked candidates -> exact graph executor -> verifier -> work accounting -> aggregate.

Key safeguards:

- exact graph executor checks complete truth tables;
- actual execution work is counted as edge operations plus active non-input node operations;
- exhaustive and selective arms execute the same candidate/task instances;
- target candidate is not used as a routing feature;
- evaluation uses the trained router returned by fit_router;
- failed benchmark split or executor checks terminate the run.

## Instrument/adversarial controls

The test suite checks:

- wiring present versus absent in the registered representation arms;
- exact executor complete-truth-table correctness;
- positive execution-work accounting;
- NOT→XOR holdout generation;
- train/evaluation structure and truth-table disjointness;
- target-index independence of semantic execution;
- primary bootstrap selection rule.

The confirmatory interpretation must separately inspect routing recall, semantic capability, actual execution work, adaptive execution, and persistence replay.

## Statistical audit

Primary estimand: minimum actual execution-work fraction over registered budgets satisfying the preregistered capability retention floor at each M.

Uncertainty: seed bootstrap with the complete primary selection rule repeated inside each bootstrap resample.

Budget curves and primary-selected values must be reported separately. No winner across representation arms is selected by the contract.

## Provenance audit

Result artifacts must retain:

- TAC-OSM commit SHA;
- workflow run ID;
- Python and Torch versions;
- pinned CASM generator commit;
- executor checks;
- split-integrity checks;
- full seed/M/budget grid;
- raw trial-level data.

## Disposition rule

A successful CI run establishes software/preflight integrity only. Scientific promotion requires a completed confirmatory artifact whose benchmark, leakage, protocol, instrument, statistics, and provenance checks all pass.

A run with a material protocol or instrument failure is VOID/BLOCKED rather than a negative scientific result.
