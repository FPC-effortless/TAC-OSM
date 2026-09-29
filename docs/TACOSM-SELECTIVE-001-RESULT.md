# TACOSM-SELECTIVE-001 — Result

Run: GitHub Actions 36509506303
Measurement commit: `ebb1dad73edd79c94bac97af6ac42c7d8729af1f`
Contract fingerprint: `7c7db37bef062e02`
Artifact: `tacosm-selective-001` (artifact 11009135323)
Configuration: H={8,64,256}, K={2,4}, seeds 0–4, 100 queries per cell.
Indexed-control capability gate: **PASS for all 6 indexed cells**.

## Primary endpoint

| H | K | exhaustive success | indexed success | indexed router candidates/query | indexed amortized candidate work |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 1.0000 | 1.0000 | 2 | 4.08 |
| 8 | 4 | 1.0000 | 1.0000 | 2 | 4.08 |
| 64 | 2 | 1.0000 | 1.0000 | 2 | 4.64 |
| 64 | 4 | 1.0000 | 1.0000 | 4 | 6.64 |
| 256 | 2 | 1.0000 | 1.0000 | 2 | 6.56 |
| 256 | 4 | 1.0000 | 1.0000 | 4 | 8.56 |

The indexed arm builds the exact equality index once per static population and
then routes only the retained bucket. At H=256 this changes actual router
input from 256 candidates to 2 or 4.

The one-time build is included in `amortized_candidate_work` over the declared
100 queries. Query-time address inspection is 2 marked positions per query.

## Interpretation

The result establishes that the repaired **runtime retrieval boundary is real**
for this exact synthetic equality control: indexed execution preserves the
same 1.0 capability while the router receives a bounded retained set.

It does **not** establish the full C5 economic claim. Executor invocations remain
one per successful task in both arms, and wall-clock time is nearly identical
because this control's executor is deliberately tiny. The result therefore
supports the runtime boundary and candidate-work accounting, not an end-to-end
compute or latency reduction claim.

The indexed control is exact content addressing, not semantic or learned
retrieval. A learned/semantic addresser must be evaluated separately.
