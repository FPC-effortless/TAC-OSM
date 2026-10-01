# TACOSM-FUSED-FRONTIER-003 — Result

Status: measured and reproducible.

## Provenance

- branch: research/fused-tacosm-frontier-003
- clean measurement workflow: 36823989132
- measurement commit: ef1d9a9ed23b78d08483e032e41d6bfd05d6dbbc
- tests: 821/821
- artifact: TACOSM-FUSED-FRONTIER-003
- artifact SHA256: 0597f949f78709663f3f5551500df2fde11685f824e0fb5513321c45c84e8882
- seeds: 10, 11, 12
- M: 128, 256, 512
- final budgets: 4, 8, 16
- task protocol: 80 unique-target held-out toggle tasks per seed/M cell

## Main result

Frontier 003 held the deterministic operator descriptor, StructMeans family gate, PST transition reranker, and unique-target evaluation fixed while changing only product-key factorization.

At B=16, execution success was:

| M | 3f16/b7 | 4f16/b4 | 4f16/b5 | 4f16/b7 | 5f8/b4 |
|---:|---:|---:|---:|---:|---:|
| 128 | 88.75% | 88.75% | 88.75% | 88.75% | 88.75% |
| 256 | 71.67% | 70.42% | 70.42% | 70.42% | 70.42% |
| 512 | 84.58% | 84.17% | 83.75% | 83.33% | 82.08% |

The 4f16/b4 configuration is the lowest-cost tested geometry while remaining close to the 3-factor control in execution success.

## Cost scaling at B=16

| M | geometry | exact addresses/query | exact-address fraction | pair-generation ops | total route ops |
|---:|---|---:|---:|---:|---:|
| 128 | 3f16/b7 | 77.89 | 60.85% | 686 | 2252.8 |
| 128 | 4f16/b4 | 59.16 | 46.22% | 512 | 1872.5 |
| 256 | 3f16/b7 | 163.89 | 64.02% | 686 | 3199.8 |
| 256 | 4f16/b4 | 122.42 | 47.82% | 512 | 2569.7 |
| 512 | 3f16/b7 | 379.20 | 74.06% | 686 | 5568.2 |
| 512 | 4f16/b4 | 228.54 | 44.64% | 512 | 3736.9 |

4f16/b4 reduces exact-address work versus the 3-factor control while retaining similar execution success on the registered workload. Exact-address work still grows materially with M, so this is not C5.

## Research interpretation

1. Frontier 001 isolated learned semantic admission as the bottleneck.
2. Frontier 002 showed that a deterministic compositional operator descriptor restores held-out admission and that PST can recover high execution success after admission.
3. Frontier 003 showed that factor geometry can reduce collision density without adding another learned rescue mechanism.
4. The remaining question is whether the 4f16/b4 geometry scales to M=1024 and M=2048 under fixed absolute budgets.

## Next phase

TACOSM-FUSED-FRONTIER-004 keeps 4f16/b4 fixed and tests M=512, 1024, and 2048 with absolute budgets B=16, 32, and 64.

Required routing accounting: factor scoring, cell expansion, exact-address scoring, PST prediction, and family-gate work.

Success condition: maintain usable execution capability while reducing or containing the exact-address fraction and total routing arithmetic. If exact-address density again becomes dominant at M>=1024, the next intervention will be a second-stage exact structural-signature index rather than more factorization or extra learning.

No claim of C5, generalized reasoning, or broad-domain continual learning follows from this experiment.