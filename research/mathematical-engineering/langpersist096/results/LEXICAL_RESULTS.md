# LANGPERSIST096 frozen-checkpoint persistence

Outcome: **NEGATIVE_RESULT**.

Internal publisher-DEV measurements; historical training FLOPs are unmatched. This is a within-checkpoint state study and establishes no architecture superiority.

| Positive-cohort paired effect | Mean | Publisher-cluster 95% interval |
| --- | ---: | ---: |
| reset_minus_carry_nll | -0.007613 | [-0.067019, 0.051793] |
| donor_minus_carry_nll | 0.023795 | [-0.033839, 0.080476] |
| carry_minus_reset_exact4 | 0.000000 | [0.000000, 0.000000] |
| carry_minus_reset_first_byte_accuracy | -0.016667 | [-0.040000, 0.006667] |

NLL gate: **FAIL**. Exact-four utility gate: **FAIL**.

Intervals resample only five publisher clusters after averaging the three frozen TAC seeds. They are exploratory and do not provide independent qualification.

Descriptive positive-minus-negative NLL interaction: -0.074055 [-0.223862, 0.070888]. This is excluded from gates and selection, and does not isolate semantic identifier storage.

Resource accounting: **PASS**.

Lexical reuse does not validate autonomous structure discovery, reusable procedures, action utility or semantic verification. Verified-write false acceptance remains OPEN.
