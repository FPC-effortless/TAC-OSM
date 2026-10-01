# TACOSM-C5-EXECUTION-FEEDBACK-PRODUCTKEY-001 RESULT

## Execution

Dedicated GitHub Actions run: 36819603954
Artifact: TACOSM-C5-EXECUTION-FEEDBACK-PRODUCTKEY-001
Artifact ID: 11142589095
Artifact digest: sha256:f36c481ee5c84acd959027d77ecc0fa6154ddc085412bdac8e26e1562dc51420
Repository suite: 814 passed
Registered protocol: M={128,256,512}, seeds={10,11,12}, train=900, eval=300, factor_count=3, factor_size=16, beam=7, max_shortlist=32, refresh=32.

## Pooled product-key results

| M | Top-1 recall | Admission recall | Conditional selection | Mean target rank | Candidates scored | Scored/M |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.4611 | 0.6067 | 0.7520 | 1.304 | 12.35 | 0.09646 |
| 256 | 0.4144 | 0.6656 | 0.6216 | 1.515 | 25.03 | 0.09779 |
| 512 | 0.2267 | 0.4856 | 0.4907 | 3.757 | 53.60 | 0.10468 |

Mean scored/M across the three M levels is 0.09964.

Factor-score MACs remain 256 per query and pair generation remains 343 operations per query. Exact state-rerank MACs average 197.55, 400.53, and 857.55 at M=128/256/512.

## Interpretation

Verifier-driven representation learning is effective: the same overall harness previously improved dense M=128 top-1 recall from 0.0367 to 0.2444. In the factorized product-key bridge, top-1 reaches 0.4611 at M=128 and remains measurable at M=512, but the capability does not meet the prior C5 frontier.

The main bottleneck is admission. Pooled admission is 0.6067, 0.6656, and 0.4856. Conditional selection also falls to 0.4907 at M=512. The factorized router therefore does not preserve the target reliably enough before exact reranking.

The rerank candidate count grows from 12.35 to 53.60 as M increases. Thus the measured sparse fraction is approximately ten percent, but the absolute rerank workload still grows with M. This is not evidence of sublinear routing.

## Decision

This experiment is MEASURED and NEGATIVE for the specific verifier-trained representation + 3-factor factor16 beam7 product-key configuration as a capability-preserving C5 solution.

It is POSITIVE evidence that:
1. post-execution verification can train the representation/router without teacher labels;
2. factorized addressing can keep the scored fraction near ten percent on this workload.

It does NOT establish:
- broad C5;
- sublinear addressing;
- hardware speedup;
- universal reasoning;
- superiority to the earlier teacher-trained factor16 beam7 frontier.

## Next bottleneck

The next registered intervention should attack the middle boundary between product-key admission and exact reranking: target-rank diagnostics, fixed absolute rerank budgets, and a structural/centroid middle filter. The objective is to make the absolute rerank count bounded while preserving admission and execution capability.
