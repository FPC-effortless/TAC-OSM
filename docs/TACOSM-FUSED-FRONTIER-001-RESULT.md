# TACOSM-FUSED-FRONTIER-001 — Result

**Status: measured.** Registered run completed on 2026-10-01 at commit \`9701a25eba4d0f18e37? NO\`.

## Provenance

| field | value |
|---|---|
| experiment | TACOSM-FUSED-FRONTIER-001 |
| branch | \`research/fused-tacosm-frontier-001\` |
| measurement commit | \`9701a25eba4d0f18cd82317144b79350b4a25273\` |
| workflow run | 36822838654 |
| research artifact | TACOSM-FUSED-FRONTIER-001 |
| seeds | 10, 11, 12 |
| M | 128, 256, 512 |
| absolute final budgets | 4, 8, 16 |
| product-key | 3 factors × 16, beam 7, max shortlist 32 |
| structural index | StructMeans, 3 clusters, cluster budget 2 |
| training | 600 verifier-driven representation updates per seed/M cell |
| single-step evaluation | 80 held-out task instances per seed/M cell |
| composite evaluation | 60 non-single-step composite tasks per seed/M cell |
| test gate | 817/817 passed |
| artifact SHA256 | \`2964e09116919f87e37bbc726d46e985d761e5d44085ba2de3dcb4570094eb2f\` |

## Primary routing result

The result does **not** support capability-preserving fusion on this exact held-out-signature routing workload.

| M | mode | B=4 top-1 | B=8 top-1 | B=16 top-1 |
|---:|---|---:|---:|---:|
| 128 | PKM | 0.0208 | 0.0208 | 0.0208 |
| 128 | StructMeans | 0.0250 | 0.0250 | 0.0250 |
| 128 | StructMeans+PST | 0.0583 | 0.0875 | 0.0958 |
| 256 | PKM | 0.0208 | 0.0208 | 0.0208 |
| 256 | StructMeans | 0.0167 | 0.0167 | 0.0167 |
| 256 | StructMeans+PST | 0.0292 | 0.0417 | 0.0625 |
| 512 | PKM | 0.0000 | 0.0000 | 0.0000 |
| 512 | StructMeans | 0.0000 | 0.0000 | 0.0000 |
| 512 | StructMeans+PST | 0.0125 | 0.0208 | 0.0292 |

The improvement from PST is real within this registered run, but absolute capability remains low and degrades with memory population size.

## Admission and cost

PKM coarse admission recall was 0.1333 at M=128, 0.1417 at M=256, and 0.0958 at M=512.

At B=16, final admission recall was 0.1208, 0.1125, and 0.0542 respectively for M=128, 256, and 512.

The internal exact-address candidate counts were:

| M | internal exact addresses scored | scored / M |
|---:|---:|---:|
| 128 | 8.95 | 6.995% |
| 256 | 18.73 | 7.316% |
| 512 | 29.95 | 5.850% |

The factor-score arithmetic was constant at 256 MACs per query because the registered factor geometry is fixed. However, internal exact-address scoring increased with M and approached the 32-item shortlist ceiling. Therefore the experiment does **not** establish bounded or sublinear total addressing cost.

StructMeans added roughly 89.9, 96.9, and 99.5 measured structural operations per query across M=128,256,512. PST prediction work at B=16 was about 99.0, 159.5, and 153.9 measured operations.

## What the structural loop did establish

The persistent verified-experience/operator-learning side behaved very differently from routing.

| mechanism | pooled result |
|---|---:|
| verified REGM records | 192 |
| held-out-signature PST reconstruction accuracy | 1.0000 |
| StructMeans kind purity — training | 1.0000 |
| StructMeans kind purity — held-out | 1.0000 |
| StructMeans exact-signature purity — training | 0.0625 |
| StructMeans exact-signature purity — held-out | 0.0625 |
| AXON bound reuse accuracy | 1.0000 |
| AXON consolidated macros | 48 |
| SECA pre-composition success | 0.0000 |
| SECA post-verification success | 0.0481 |
| verified novel composites | 2.89 per seed/M run on average |
| final pair admission recall | 0.00185 |

The structural result is especially diagnostic. With 16 operators per kind, exact-signature purity of 0.0625 is consistent with a clustering representation that identifies the operator family but loses the within-family mask identity needed for exact retrieval.

SECA also created verified executable compositions, but pair admission was extremely low. The evidence therefore supports the existence of a functioning composition/verification mechanism without showing that sparse retrieval reliably supplies the operands it needs.

## Interpretation

This run splits the fused architecture into two separate findings.

First, **verified transition learning and execution consolidation are functioning** on this synthetic operator family. PST reconstructs the transition law perfectly on held-out signatures, AXON reuses bound operators exactly, and SECA can produce independently verified novel composites.

Second, **the current sparse routing stack is not capable enough for exact held-out operator retrieval**. Adding StructMeans did not materially raise admission, because its learned abstraction stops at operator kind. Adding PST improves the final selection after admission, but it cannot recover operators that the product-key stage never admits.

This is not a C5 result. The measured product-key internal work grows from 8.95 to 29.95 addresses as M grows, so the total routing computation remains population-dependent.

## Next research phase

The next intervention should therefore target **hierarchical structural addressing**, not another verifier or another execution module:

\[
\text{query}
\rightarrow
\text{coarse PKM}
\rightarrow
\text{operator-kind StructMeans}
\rightarrow
\text{fine structural signature index}
\rightarrow
\text{PST compatibility}
\rightarrow
\text{AXON execute}
\rightarrow
\text{independent verify}
\rightarrow
\text{REGM}
\rightarrow
\text{SECA on retrieval failure}.
\]

The fine index must preserve exact operator signature while remaining sparse. The cleanest controlled test is a deterministic descriptor/product-key index over the 11-dimensional operator descriptor, with held-out signatures included only as indexed items, not in the codebook-training set.

The next experiment should compare:

1. learned PKM admission used here;
2. deterministic structural PKM over operator descriptors;
3. StructMeans kind admission followed by deterministic structural PKM;
4. the same hierarchy plus PST reranking.

Keep the same M, seeds, and absolute budgets, and report all internal address-scoring work. The primary diagnostic is whether exact-signature admission rises substantially before adding any new execution machinery.

**No claim of generalized reasoning, continual learning, or C5 follows from this experiment.**
