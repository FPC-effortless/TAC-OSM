# G-CASM-016C-R1 — INVALID / PROVENANCE ONLY

Status: INVALIDATED. R2 is the authoritative corrected repeat.

The R1 artifact was registered as a corrected repeat, but the required scalar
evidence extraction correction was not actually applied. The scalar arm compared
bare integer candidate evidence against a one-tuple target representation, so
the compatible scalar bucket was structurally empty rather than empirically
small.

The reported R1 capability delta is therefore not a measured arm difference.
No R1 confirmatory result may be used for scientific inference, tuning, model
selection, benchmark selection, or baseline selection.

The authoritative successor is:
TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R2.

R1 remains in the repository only to preserve the failure provenance and to make
it impossible to silently erase the methodological history.

Required CI behavior: audit the invalidation only; never execute the R1
measurement runner.
