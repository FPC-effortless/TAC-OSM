# TAC-LANG-MEM-500K-007 — owner-bound dual-timescale language memory

**Date:** 2026-10-28. **Status: exploratory real-text DEV results; full predictive + retrieval gate FAIL.** This branch is STACKED on #529 (graph/stream repair), itself stacked on original #528. All original checkpoints and negative results remain untouched. No sealed TEST accessed, no model promotion or merge.

## Source-level changes and why

Exact candidate modules: `tac_wave_ced_500k_v3.py` (Git blob `5c0fe85d7b63464a76fa93b408e8f8bc4d0bb653`) and `historical_500k_v3.py` (Git blob `0f3373c589f67a60566452612025fffb4792091d`).

- The original native and first aligned bank remain fast. The **second aligned head** becomes slow with `alpha_slow=.995+.004*sigmoid(raw_retention)`, `erase_slow=.002*sigmoid(raw_erase)` and `write_slow=(1-alpha_slow)*sigmoid(raw_write)`. The coupled write remains conditionally Frobenius-bounded in exact arithmetic, without claiming perfect source recovery. The old fast/slow interference and capacity limits are not solved.
- Fixed read gain 32 for the slow bank, both ordinary decoder read and lag64/192 history reader; no additional learned weights. No claim this gain is optimal or portable.
- Optional **caller-supplied strictly past** `query_ordinals` can drive the *actual primary next-character decoder* through its aligned-memory read. Queries are only legal with per-row `source_ids`; carried state contains the same nonempty source IDs, and owner mismatch fails closed. This is caller-provenance validation, NOT independently authenticated owner authorization. No old source character or future target is supplied to the model.
- Total original base trained weights **499,290** and historical-head weights **503,514**, unchanged; persistent learned floating state remains 520 scalars plus new owner metadata.

## Authentic source protocol

36/36 original human-authored Vim9.1 document SHA-256 checks passed: 30 FIT whole chapters, six already exposed DEV chapters. **Only 48 DEV windows from six previously exposed texts** for evaluation. Byte vocabulary 128; non-ASCII bytes mapped to 63. All tests here are exploratory/forensic; no independently sealed TEST or general-language capability claim.

Base training: seeds 1701–1705, 120 AdamW updates/seed, batch8, context128, LR .001, gradient clip1 and identical original GRAPH006 per-seed FIT-window sample hashes. Historical fine-tuning: the corresponding five v3 base states, 120 additional updates per seed, context256, batch4, LR .0006, exact paired FIT-window sampling; `history_aux` loss = next CE +0.3×mean(lag64,lag192 historical CE), whereas separately trained `next_only` uses next CE. Objectives differ; controls are not equal-objective architectural parity. No exact FLOP-parity measurement.

### Primary next-character performance (5 seeds)

| Arm | DEV mean PPL ↓ | Mean NLL ↓ | Full-256 vs reset-128 advantage (reset-full) ↑ |
|---|---:|---:|---:|
| Original GRAPH006 ancestor | **12.6518** | **2.537438** | -0.003715 |
| Corrected GRAPH006 v2 graph-on | 12.9776 | 2.562773 | -0.004547 |
| New GRAPH007 v3 dual-timescale | 13.3144 | 2.588341 | -0.008226 |

New v3 base per-seed DEV PPL: 1701 **13.9023**, 1702 **13.1808**, 1703 **12.9946**, 1704 **12.7865**, 1705 **13.7077**. The model **loses next-character perplexity to v2 in every seed**. Base-model state reset remains negative on average; model does not prove primary-language benefit.

### Authentic old-source reconstruction (5 paired additional training seeds)

| Arm | Next PPL ↓ | Next NLL ↓ | Next top1 ↑ |
|---|---:|---:|---:|
| v3 history-supervised | 10.32613 | 2.334495 | 36.99% |
| v3 next-only | **10.22538** | **2.324731** | **37.21%** |

| Target | v3 correct key NLL ↓ | Wrong shifted key | Zero read | FIT unigram prior ↓ |
|---|---:|---:|---:|---:|
| Lag64 | **4.22580** | 4.76454 | 4.77120 | **3.53225** |
| Lag192 | **4.70200** | 4.89441 | 4.77681 | **3.63324** |

Historical v2 comparison: lag64 **4.39764** and lag192 **4.91637**. v3 makes the correct key outperform wrong/zero reads at lag192, but **both v3 lags fail the FIT-only source prior**. History auxiliary training harms next-character NLL in all five paired seeds. Note different v2/v3 base priors: apparent historical-NLL improvement is exploratory, not matched reimplementation proof or successful retrieval.

### Causal source-specific evidence

At lag192, zeroing **only aligned memory at token128**, preserving all other carried states and absolute ordinals, increases v3 reconstruction NLL by mean **+0.021413**, positive **5/5** seeds (0.02963, 0.02626, 0.01983, 0.01554, 0.01581). GRAPH006 was exactly zero at recorded FP32 precision. This establishes meaningful causal dependence on old aligned state, **not enough information to identify correct old content**.

Frozen real-text source-t0 embedding-output gradients of the lag192 loss are **nonzero 5/5**, ranging **0.00352–0.00910** in this small 5-passage probe, compared with GRAPH006's targeted zero gradients. Actual historical logits change by 0.017–0.061 for a real-old-byte substitution in the probed passage. An explicit past-ordinal query changes the ordinary decoder logits; this proves a usable computational route, not improved prediction from user-provided pointers. Owner mismatch in streamed state is rejected 5/5 probes, but callers remain responsible for authenticating source identity.

## Replay, scope, qualification

All **5/5 v3 base and 10/10 v3 historical checkpoints** pass local frozen no-refit NLL replay against source-hash-verified DEV. Original 15 new checkpoint files, optimizer/RNG/sampler states, per-seed sample hashes, exact local training/evaluation scripts and finite diagnostic receipts are provided as a separate downloadable archive. **They are NOT checked into Git or LFS; compulsory independent repository-backed cold replay is BLOCKED.** The new versioned source and constructed numerical tests are present in this PR; CI verdict must be checked independently. This comparison is 500K research scale, not legacy 100K promotion.

Companion PNDS source-math branch `research/500k-dual-timescale-memory-007-2026-10-28` adds Lean and Python exact-real specifications of gate ranges, coupled write and owner equality. Five new Lean theorem checks are proposed but **NOT** an end-to-end model, IEEE, AD, 16×8 Frobenius, or actual recall proof. Do not report Lean success without clean CI.

**Research decision: FAIL / meaningful partial mechanism repair.** Avoid automatically scaling to 1M or claiming a product. Next prospectively register an independent source-ID task with genuine >512-token dependencies, a proper indexed lookup and RAG control, explicit wrong-owner protection, direct downstream next-token utility, and matched 500K training compute/parameters/data. Source-specific content capacity/address collision remains an open bottleneck.
