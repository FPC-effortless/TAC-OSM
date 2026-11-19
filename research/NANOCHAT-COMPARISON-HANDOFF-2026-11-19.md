# TAC-OSM language architecture and nanochat comparison

Status as of November 19, 2026. This handoff records existing results and proposes a comparison. Prospective settings below require registration before any fit or scored evaluation.

## Architecture lineage

- **Latest integrated language candidate.** BYTE100K-001 has 99,050 learned parameters and a tied byte readout, with native CSL role-state recurrence, FP64 carried memory and bounded grouped-query attention. The hosted work exercised original FIT bytes and reproduced the original untrained weights, predictions, states and gradients. It performed zero optimizer updates. [Implementation and engineering evidence](https://github.com/FPC-effortless/TAC-OSM/blob/a7ec101ee67fce09aaf42b5c8e880586f77be0e9/research/byte100k_001/README.md)
- **Larger candidate.** CSL-125 R4 at 123,873,263 parameters exercises full-model gradients, streaming and partial model exits on CPU. The precision repair cleared the 84 registered operator-gradient cases in FP64 memory and added 17.83% median latency at full geometry on the reported CPU fixture. Both measurements concern untrained software behavior. Its tool and risk heads belong to the registered R4 variant. A language-only adaptation would need a new source pin and parameter count. [CSL-125 R4 evidence](https://github.com/FPC-effortless/TAC-OSM/pull/721)
- **Already-trained scaling candidate.** LANGSCALE001 tested moving capacity into temporal state at matched 1M and 10M parameter budgets. The completed Actions run contains 30 fits and a successful cohort audit. It is the immediate source for a controlled follow-up on whether the memory reallocation actually helps. [Training source](https://github.com/FPC-effortless/TAC-OSM/tree/8d456c1bc1194114b376ad037f8662a38397199b/research/langscale001) · [Completed run](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41407899927)

## Recovered LANGSCALE001 evidence

The cohort audit records 30 complete fits and 150 verified checkpoint milestones. Each fit used 2,560 updates, 1,310,720 byte presentations and the registered CPython FIT corpus. The table averages the recorded mean loss over the final 128 training updates across the three seeds in each arm. The weights change over those updates. The data were used to fit the models.

| Arm | 1M training loss | 10M training loss |
| --- | ---: | ---: |
| Original TAC associative stack | 1.64266 | 1.51170 |
| Head-to-state TAC | 1.66683 | 1.54219 |
| Memory-scaled TAC | 1.65385 | 1.46332 |
| Causal Transformer | 2.19907 | 1.99151 |
| Stacked GRU | 1.64755 | 1.24711 |

At 10M, memory-scaled TAC improved on original TAC in each seed, by 0.04838 on average. At 1M, it lost to original TAC in each seed. The 10M GRU scored lower than both TAC variants in every seed. The campaign recorded zero DEV or TEST access. Full compute parity remains open. The frozen authorization covers the original FIT campaign.

The GitHub metadata digest matched the downloaded audit ZIP's SHA-256, d8516283cfe4611aa08903efcde0d05559aa7806b8f1f789cdbfb795a7667427. This handoff recomputed grouped means from ALL_30_FIT_AUDIT.json. It did not rerun hosted training or reload the 150 checkpoints. GitHub lists the artifact expiry as January 31, 2027. [Audit artifact](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41407899927/artifacts/13128843899)

The earlier, separate LANGCAP105 held-out CPython DEV evaluation measured 10M TAC at 1.61762, Transformer at 2.08296 and GRU at 1.34676. Those are frozen original models. The measured quality of the newer memory-scaled variant on held-out data remains open. [LANGCAP105 DEV evidence](https://github.com/FPC-effortless/TAC-OSM/pull/687)

## A direct nanochat target

Pin upstream nanochat at a825e63 for the speed reference. The published leaderboard reports 0.2626 CORE, 0.71800 validation bits per byte and 1.65 hours on 8 H100s. The frontier quality row at 6ed7d1d reports 0.2690 CORE. A hardware comparison requires rerunning the baseline on the same hardware and at the same precision and measurement boundaries. Upstream also supplies a separate chat evaluator. [Published leaderboard](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md#time-to-gpt-2-leaderboard) · [Chat evaluation](https://github.com/karpathy/nanochat/blob/a825e63/scripts/chat_eval.py)

Proposed program objectives for a future registration are at least +0.03 absolute CORE versus the pinned reference on the full suite, a lower held-out bits-per-byte result, and retention of task-level chat quality after comparable supervised training. Measure GPU-hours to the same CORE score, batch-1 prefill and decode latency, and peak memory on identical hardware. A proposed efficiency target is 25% fewer GPU-hours to the same quality. Report task scores and latency distributions alongside aggregates. Document benchmark exposure and use a new access-separated holdout with real long-document questions that depend on distant evidence.

## Handoff for the next study

1. Register a frozen read-only evaluation of all eligible LANGSCALE001 checkpoints. Include original TAC, both reallocation variants, the GRU and the Transformer. Start from the cohort audit and original source-order receipts. Use a newly pinned source-disjoint development set and identical scorer definitions. Include carry, reset and wrong-history controls at different evidence distances. Record the new protocol separately from the historical FIT authorization.
2. Advance the CSL-derived language candidate if the next study demonstrates both useful longer context and credible execution cost. Train a faithful from-scratch nanochat control and a strong recurrent control on the same data. Use a shared pinned tokenizer, matched context and parameter counts, equal sample exposure and tuning, and compare learning curves at matched measured compute. Retain the published nanochat model as an external absolute quality reference. Follow the repository's declared-target-v2, PNDS and compute rules before fitting. [TAC-OSM research rules](https://github.com/FPC-effortless/TAC-OSM/blob/5f288776f87be1423ec607f5077b96400654ffdc/research/REAL-WORLD-EVIDENCE-RULES.md)
3. Record hashes for source, corpus, tokenizer, sample order, checkpoints and scorer through the Research Lab and TrainLab workflow. Preserve original predictions for independent rescoring and exact replay. Use Veritas task contracts and outcome verification for downstream executable tasks. The language score still comes from the pinned external language evaluator. [Veritas ecosystem](https://github.com/FPC-effortless/veritas/blob/eec439492407b0aa26922c23b5ae40d95289f4eb/docs/ecosystem/README.md)

No fit, new evaluation, external compute or protected data access occurred for this handoff.
