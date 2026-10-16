# DEV-021 — Audited trained decay/erase suffix transport results

**Completed adaptive-development diagnostic; independent audit PASSED. Native TAC world-model success remains FALSE.** October 16, 2026.

## Source, complete evidence and validation

- [Frozen full GitHub Actions run 38834746064](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38834746064): initial lint/regression/smoke passed, **all six original-checkpoint seed/regime jobs passed**, separate stdlib recomputation/paired-history audit passed.
- **1,053 tests passed** in the full suite; suite provenance synchronization, ruff undefined/unused checks and diagnostic identity smoke passed.
- [Complete audited evidence ZIP — artifact 12019045747](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38834746064/artifacts/12019045747), 660,302 bytes, archive SHA-256 `be752f3d175c5b76a1569f80648803b6cbf7ef0ec5fad65d3ad6fd6de8a5d3a3`. Contains all six per-seed/regime raw `report.json` files plus aggregated `summary.json`.
- The original DEV-018 checkpoint artifacts come unchanged from [run 38802569971](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802569971) and are checked against the archived checkpoint and parameter hashes for all five arms. This is **not** independent retraining of the original model.
- [Committed auditable summary](DEV-021-AUDITED-SUMMARY.json) retains full per-seed values and the independent auditor's reported aggregate; the original ZIP is the authoritative raw evidence.
- The [protocol](DEV-021-SUFFIX-TRANSPORT-PROTOCOL.md) was committed before training/diagnostics and the [earlier proposal](DEV-021-SUFFIX-TRANSPORT-PROPOSAL.md) was authored during DEV-020.

Six analyses cover three training seeds (43/59/71) × two DEV-018 training recipes (short/long). Each frozen analysis measures 192 balanced opposite-future H512 pairs for each of two native TAC arms, across pending/reissued cancellations, at/after due, and delays 32/128/512, yielding **2,304 paired transport rows**. The same public suffix was checked exactly after each pair's last unequal event; all 2×2 coefficient combinations were evaluated on that same suffix.

## Main result — native TAC joint

Mean H512 Euclidean norms of opposite-history memory differences, averaged across **three seed means**, with 192 pairs per arm/seed/regime:

| Dynamics of the frozen state *difference* | Short training | Long training |
| --- | ---: | ---: |
| Original trained decay + erase | **0.001544** | **0.000832** |
| Force decay = 1 (erase retained) | **0.053018** | **0.043365** |
| Force erase = 0 (decay retained) | **0.003101** | **0.001348** |
| Both disabled | **0.095766** | **0.065120** |

On the ratio of regime-level mean norms, removing decay raised state separation by **about 34.3× short and 52.1× long**, while removing erase alone raised it **about 2.0× short and 1.6× long**. Ratios of the mean of per-seed ratios are different (35.41× / 54.45× for no-decay) and are also preserved in the JSON. These are **not accuracy gains**, memory information scores, or effects on native service decisions.

### Frozen-encoder native writer control

The `tac_encoder` arm freezes the memory writer at its original initialization, while adapting the encoder. Its short/long mean norms were:

| Frozen-encoder writer dynamics | Short | Long |
| --- | ---: | ---: |
| Original | 0.001585 | 0.001609 |
| No decay | 0.045904 | 0.046584 |
| No erase | 0.002669 | 0.002423 |
| Neither | 0.077200 | 0.070087 |

This arm's per-expert suffix decay product is **0.0345839** across regimes and seeds; its decay gates were *not learned* because the writer was frozen, so the product chiefly reflects the initialized retention over the checked suffix lengths. By contrast, `tac_joint` trained writer had per-expert mean decay products from about **0.0092–0.0343**, depending on training regime and expert. The products cannot be directly treated as measures of label information.

## Raw H512 control distances and original outputs

Mean across seeds, measured only as observational readouts:

| Control metric | Short training | Long training |
| --- | ---: | ---: |
| Schema41 state L2 | 0.699323 | 0.720371 |
| GRU768 state L2 | 0.0000000740 | 0.0000203522 |
| Memoryless state L2 | 0 | 0 |
| TAC joint original read-input L2 | 0.0002659 | 0.0001068 |
| TAC joint original paired positive-probability gap | 0.00004419 | 0.00000894 |

State/vector norms from different architectures use different coordinates and are **not a fair cross-architecture encoding performance scale**. The original native TAC target predictions assigned the same class to both opposite-outcome partners in **every evaluated pair** across both arms, three seeds and both regimes. No original native output success was recovered.

## Exact decomposition integrity

The tested frozen actual state update has a difference equation:

```text
D[t+1] = decay[t] * D[t] - assignment[t] *
         ((D[t] @ (erase[t] * key[t])) outer key[t])
```

The additive write term cancels only because inputs are identical after the actual last unequal public observation. All suffix values, learned address keys and assignments were fixed in each factorial condition. Direct two-state update and independently transported differences agreed, with maximum componentwise errors in the individual reports of roughly **1.6–2.4 × 10⁻¹⁵**, vastly below the preregistered `2e-9 + 2e-5 * |D|_max` threshold. Every report and raw row was verified by the separate standard-library auditor, which also regenerated the public paired histories and exact last-divergence positions.

The float64 arithmetic is explicitly a **post-hoc algebraic diagnostic** of the same frozen weights, not a replacement of DEV-018's original float32 prediction benchmark. Numerical correctness does not imply scientifically useful retained state.

## Scientific interpretation, qualified

**What DEV-021 now supports:** On this fixed synthetic H512 suffix and checkpoint set, the frozen TAC writer's multiplicative decay accounts for much more attenuation of the *measured state difference* than the isolated erase term. This is evident across all six seed/regime analyses. The original TAC joint model's learned decay also contributed, but the frozen encoder/writer control already shows large initialized contraction; a learned-vs-frozen causal training attribution is **not** established.

**What DEV-021 does not support:** It does not prove that any removed component would preserve task-relevant causal information, that changed full-state dynamics remain calibrated, or that prediction and intervention costs improve after editing retention. The factorial acts on differences, not full states; removing contraction can amplify distracting history and cause instability or overwrite. No TAC overall capability, memory-attribution or commercial readiness gate has changed from **FALSE**.

The next architecturally meaningful step is a **prospective trained intervention** that adds explicit retention controls while separately testing identity-aware memory and key/read alignment with balanced parameter/training budgets. That future experiment must report original future-prediction accuracy, generalization across history and entity counts, failure and unnecessary service rates, full physical cost, baseline comparisons, checkpoint provenance, and end-to-end replay; a lifted memory norm alone is not success.

**Status: keep draft and unmerged.** DEV-018, DEV-019 and DEV-020 negative findings remain intact.
