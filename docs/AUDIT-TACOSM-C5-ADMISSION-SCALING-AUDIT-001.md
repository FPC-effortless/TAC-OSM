# Scientific Audit — TACOSM-C5-ADMISSION-SCALING-AUDIT-001

## Disposition

**CONDITIONAL / reusable as a finite-range routing/admission audit**.

## Benchmark

PASS within the registered synthetic binary-code workload. The experiment measures dense target rank and explicit LSH admission/work from M=64 through M=8192, with five seeds, calibration/held-out splits, a fixed LSH cap, and a registered workload. The task remains a one-bit relational code benchmark with an exact Hamming shortcut, so it is not a semantic representation benchmark.

## Leakage

PASS for the main dense/admission measurement as documented: the main arm has no persistent-state or hidden-target input, and target correctness is evaluator-side. The exact Hamming shortcut is a benchmark limitation, not an information leak introduced after registration; it limits generalization of the result.

## Protocol and implementation

PASS within the audited source/result record. The run is pinned to commit 842edc6d56728df4c3b5ec251adbec8b652a8a1f, uses seeds 0–4, fixed M levels, fixed LSH table sweep, and reports dense rank and admission/work separately.

## Statistical interpretation

QUALIFIED. The reported full-range gamma=0.8732 and local exponents 0.585, 0.737, 0.848, 0.918, 0.957, 0.978, 0.989 are descriptive over the finite range. The increasingly high local exponents are useful evidence that rank growth becomes close to proportional to M in the observed range, but they do not establish an asymptotic exponent.

The explicit LSH sweep is stronger evidence than the iid table-count diagnostic: it directly measures admission/work tradeoffs. The fact that 24 tables are needed to exceed 90% admission while the iid diagnostic predicts approximately 9.2 is a workload-specific empirical discrepancy, not a general refutation of the iid approximation.

## Material result

The experiment demonstrates a clear routing/admission bottleneck: dense rank grows close to linearly by M=8192, while higher admission recall requires materially higher reranking work. This separates two effects that had previously been conflated: representation/rank scaling and LSH admission efficiency.

## Nonclaims

- no semantic-retrieval claim;
- no persistence claim for this main arm;
- no optimal LSH operating point;
- no asymptotic complexity theorem;
- no general PLM or language-intelligence claim.

## Reuse rule

Future work may inherit the finite-range observation and the explicit admission/work frontier as a diagnostic baseline. It must remove the exact Hamming shortcut before using this line to support claims about learned semantic or relational representations.