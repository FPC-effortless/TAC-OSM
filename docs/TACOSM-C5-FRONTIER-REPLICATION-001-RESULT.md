# TACOSM-C5-FRONTIER-REPLICATION-001 RESULT

Status: MEASURED — REPLICATION FAILED.

## Provenance

- GitHub Actions run: 36791657545
- Measurement head: ca18cf86fab349c46701e133a00a22941362914e
- Full artifact: TACOSM-C5-FRONTIER-REPLICATION-001-full
- Artifact id: 11131704780
- Artifact digest: sha256:94d95340730c4b3ad7606351c8b59a092b3c04026ef14e74da06a775061cbade
- Preconditions: passed
- Smoke: passed
- Full measurement: passed

## Validity

The replication used the same state-distinct executor and exact count-conserving aggregation as C5-004. Independent verification found zero pooled count inconsistencies across 42 pooled cells. Each pooled cell contains 500 evaluations (five independent seeds × 100 evaluation steps).

## Primary replication result

The preregistered configuration `(factor_size=16, factor_beam=6)` did not reproduce the 0.80 capability-retention floor at every M:

| M | exhaustive successes | selective successes | retention | states scored / M |
|---:|---:|---:|---:|---:|
| 128 | 172/500 | 166/500 | 0.9651 | 0.1548 |
| 256 | 104/500 | 97/500 | 0.9327 | 0.1526 |
| 512 | 21/500 | 14/500 | 0.6667 | 0.1522 |

The replication therefore fails criterion 1 at M=512.

## Configuration frontier

The independent-seed global frontier is `(factor_size=32, factor_beam=8)`:

- mean `states_scored_over_M` = 0.0791;
- minimum capability retention across M = 0.8721;
- mean total query arithmetic = 1,046.44 MACs.

This configuration was not the preregistered replication target. Its appearance as the independent-seed frontier shows that the capability/computation surface is sensitive to factor configuration.

## Combined ten-seed observation

Combining the original C5-004 seeds 0–4 with replication seeds 5–9 gives exact pooled counts for `(16,6)`:

| M | exhaustive successes | selective successes | retention |
|---:|---:|---:|---:|
| 128 | 353/1000 | 325/1000 | 0.9207 |
| 256 | 197/1000 | 185/1000 | 0.9391 |
| 512 | 48/1000 | 37/1000 | 0.7708 |

The ten-seed aggregate remains below the preregistered 0.80 floor at M=512. Thus the failure is not solely caused by one unusual seed in the replication set.

## Scientific interpretation

C5-004 remains a valid bounded result for its registered seed set: it demonstrated a fixed `(16,6)` configuration preserving the reference capability floor on seeds 0–4.

The independent replication shows that the specific `(16,6)` configuration is not robust to the registered seed change at M=512. The evidence therefore weakens the claim that `(16,6)` is a stable fixed frontier configuration, while leaving intact the narrower observation that product-key granularity and beam expose a capability/computation tradeoff.

This is not evidence that sparse product-key retrieval fails in general. A different configuration, `(32,8)`, satisfies the floor across all tested M on seeds 5–9 while scoring about 7.9% of M on average.

However, that `(32,8)` result is exploratory relative to C5-004's accepted fixed configuration because it emerged from the same registered search surface but was not the preregistered replication target. It should not replace `(16,6)` in the claim ledger without its own robustness protocol.

## C5 status

No upgrade is made to the C5 claim from this replication. The accepted C5 result remains PARTIALLY SUPPORTED, bounded, with an explicit reproducibility qualification: the first fixed frontier configuration does not reproduce its capability floor at M=512 on independent seeds.

The next experiment should test robustness across a larger seed set and determine whether a configuration such as `(32,8)` is genuinely stable before introducing a new routing architecture. This separates optimizer/seed sensitivity from architectural inadequacy.
