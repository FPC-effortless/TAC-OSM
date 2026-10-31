# Parent097 ARC output diagnosis

Posthoc read-only analysis of preserved LANGACQUIRE097 predictions, prepared separately from LANGCONTROL098 outcomes. No model calls or optimizer steps; frozen098 endpoints are unchanged.

The original archive SHA256 is `0e8c425a8cc99483cc70f5b9cde238614d348c340762f848f5ca4c59e07a5b63` (135,590,835 bytes; artifact13019296161). The script validates frozen actor/judge hashes, forty prediction/replay pairs, checkpoint identities and all320 carried ARC score labels.

| Measurement | Outputs /320 |
| --- | ---: |
| Legal dimension header | 291 |
| Correct dimensions | 124 |
| Contains semicolon | 36 |
| Valid complete grid | 1 |
| Exact correct grid | 0 |
| Correct dimension-and-cell prefix even ignoring delimiter | 0 |

Exclusive first-failure categories, in the script's declared order:

| Category | Outputs |
| --- | ---: |
| illegal_header | 29 |
| missing_delimiter | 261 |
| non_digit_payload | 0 |
| payload_length | 29 |
| valid_wrong_shape | 0 |
| valid_wrong_cells | 1 |
| exact | 0 |

The one valid carried grid was Transformer/query/seed1705/task9565186b: `33300000000;` versus gold `33535335535;`. Both declare3×3; the generated cells are wrong. None of the320 strings already contains the complete correct dimension-and-cell prefix, even when the final delimiter is ignored. The observed failures therefore cannot be described as only a delimiter problem.

First-byte NLL evaluates the row-count character. Header, stopping, length and cell errors can be decomposed from saved bytes; outcomes under constrained decoding and probabilities of later correct tokens require new model calls and are not estimated here.

The320 outputs repeat eight original tasks across four families, two objectives and five seeds. They are not320 independent tasks. This is error diagnosis of a prior negative, not a new capability result or evidence for an architecture winner.

Reproduce using the exact original archive:

```bash
python research/mathematical-engineering/langcontrol098/results/parent_diagnostics/diagnose_parent_arc_serialization.py --archive /path/to/LANGACQUIRE097_original_results.zip --output /new/path/PARENT_ARC_DIAGNOSIS.json --markdown /new/path/PARENT_ARC_DIAGNOSIS.md
```
