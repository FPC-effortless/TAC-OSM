# TACOSM-C5-END-TO-END-001 RESULT

## Run provenance

- workflow: `36667352698`
- clean registered run: #8
- clean measurement head: `03bc42761a914051bc62d12588b842a1357a5618`
- precondition gate: passed
- full test suite: **876 passed**
- measurement step: passed
- artifact: `TACOSM-C5-END-TO-END-001`
- artifact id: `11077145266`
- artifact digest: `sha256:b9000565904f4d465496765dab0a43d6218601898f12776f246e0d8d0320fbaa`

## End-to-end path

The selective arm executed the complete controlled path:

`one-bit-noisy query -> persistent state at M=64 -> Hamming state index -> exact candidate index -> execute R`.

The exhaustive arm used the same task truth and executor but scanned, matched, and executed the full H candidate population.

## Capability

| H | M | R | Exhaustive success | Selective success |
|---:|---:|---:|---:|---:|
| 64 | 64 | 4 | 1.0000 | 1.0000 |
| 128 | 64 | 4 | 1.0000 | 1.0000 |
| 256 | 64 | 4 | 1.0000 | 1.0000 |

All 15 seed/H cells in both arms matched the independent aggregate evaluator.

## Work

Each candidate execution carries 12 fixed work units.

| H | Exhaustive executor calls/query | Selective executor calls/query | Exhaustive work/query | Selective work/query | Reduction | R/H |
|---:|---:|---:|---:|---:|---:|
| 64 | 64 | 4 | 768 | 48 | 93.75% | 0.0625 |
| 128 | 128 | 4 | 1,536 | 48 | 96.875% | 0.03125 |
| 256 | 256 | 4 | 3,072 | 48 | 98.4375% | 0.015625 |

At H=256, the selective path executes 4 candidate programs instead of 256 while producing exactly the same task output.

## Addressing integrity

- persistent state target retention: **1.000**;
- all four relevant candidate programs retained: **1.000**;
- state index query probes: **56**;
- candidate index query positions: **10**;
- M=64 state items are written once and reused within each query block;
- H candidate population is indexed once and reused within each query block.

The state and candidate indexes are deterministic exact controls. Their build work is reported separately and amortized over the registered 100-query cells.

## Interpretation

This is the strongest bounded C5 mechanism result in the current ladder.

It demonstrates the combined controlled relation:

`persistent semantic state -> relevant subset R -> execute(R)`

while the exhaustive baseline executes all H candidate programs.

The result is stronger than SELECTIVE-001 because execution itself is now a population-level operation whose correct output requires all relevant programs.

## Limits

The experiment does **not** establish the broad L4 C5 program claim.

Specifically, it still uses:

- a 10-bit synthetic state representation;
- a hand-designed Hamming index;
- an exact candidate equality index;
- a synthetic fixed-work executor;
- a fixed relevant set R=4;
- one-step persistence with M=64.

It therefore does not establish learned semantic indexing, natural-language retrieval, hardware speedup, long-horizon memory, or general capability.

## C5 status

C5 remains ungraded as a broad program claim.

The repository now has bounded positive evidence for its component relation:

`C_address -> C_route(R) -> C_execute(R)`

with exact capability parity in the registered synthetic workload.

The remaining test is whether the same behavior survives when the retrieval boundary is learned rather than hand-designed and when full index-maintenance and query costs are included in the end-to-end accounting.

## Next experiment

Replace the exact state/candidate indexes with a learned semantic coarse index while preserving the same scorer and executor. The capability rule should be one-sided:

`indexed_success >= exhaustive_success - 0.05`.

Any positive capability delta should be reported separately rather than retrofitted into the decision rule.

## Frozen evidence

REP-001 through REP-009, SELECTIVE-001, C5-EXEC-001, and earlier C5 artifacts remain separate historical measurements.