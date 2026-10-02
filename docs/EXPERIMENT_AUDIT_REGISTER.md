# Experimental Audit Register

This register is the portfolio-level scientific-process ledger. It is distinct from the claims/evidence ledger: a claim records what the evidence supports; this register records whether the process that generated the evidence was valid.

| Experiment | Benchmark | Leakage | Protocol | Instrument | Implementation | Statistics | Provenance | Current disposition |
|---|---|---|---|---|---|---|---|---|
| TACOSM-TEMPORAL-001 | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | Existing bounded result; audit before reuse |
| TACOSM-BASELINE-001 | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | Existing mechanism result; audit before reuse |
| TACOSM-HS-001 | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | **HISTORICAL FAILURE FOUND** | **historical integrity defect** | REVIEW NEEDED | REVIEW NEEDED | Historical result constrained by C4; audit provenance |
| TACOSM-MATCHED-001 | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | Negative result; audit before reuse |
| TACOSM-LEARN-001 | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | Negative result; audit before reuse |
| TACOSM-SURROGATE-001 | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | Mechanistic ranking result; audit before reuse |
| TACOSM-C5-001 | **FAILED** | REVIEWED/RECORD | REVIEWED | **VOID** | REVIEWED | REVIEW NEEDED | REVIEWED | VOID; do not use as C5 evidence |
| TACOSM-C5-002 | **FAILED GATE** | REVIEW NEEDED | REVIEWED | **BLOCKED BY REPRESENTABILITY GATE** | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | Instrument-invalid; successor required |
| TACOSM-C5-003 | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | PREREGISTERED | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | Do not run until audit passes |
| TACOSM-SELECTIVE-001 | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | REVIEW NEEDED | Bounded runtime mechanism; audit before reuse |
| TACOSM-GRAPH-CASM-SELECTIVE-010 / PR #68 | REQUIRED NOW | REQUIRED NOW | PREREGISTERED | REQUIRED NOW | REQUIRED NOW | REQUIRED NOW | REQUIRED NOW | **NO SCIENTIFIC RESULT DISPOSITION YET** |
| TACOSM-VRS-DYNAMIC-REPRESENTATION-001 / PR #69 | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | PREREGISTERED | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | **NO MEASUREMENT RESULT** |
| TACOSM-REPRESENTATION-REFINEMENT-002 / PR #71 | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | PREREGISTERED | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | REQUIRED BEFORE RUN | **NO MEASUREMENT RESULT** |

## Mandatory ordering

For any experiment not yet measured:

**audit benchmark -> audit leakage -> audit protocol -> audit implementation -> adversarial instrument tests -> approve confirmatory run -> run -> freeze artifact -> statistical review -> claim review.**

For already-measured experiments:

**freeze historical artifact -> audit -> classify -> retain/retract/qualify -> only then use as evidence.**

No experiment advances merely because its implementation passes CI.
