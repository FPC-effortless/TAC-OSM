# TAC-LANG-GRAPH-006 — corrected 500K language train/benchmark report

**2026-10-28 — authentic Vim9.1 text, previously exposed DEV only — scientific gate FAIL.**

This is a new five-seed study of the exact corrected source in draft PR #529, preserving original PR #528 and its negative historical retrieval results. No independent sealed TEST was opened and no model is promoted.

## Frozen protocol and source

- 30 genuine Vim9.1 FIT chapters, 6 previously exposed DEV chapters; source files 36/36 SHA-256 match original protocol. Real source bytes excluded from replay archive.
- Three base arms: original WaveCED500K; corrected v2 with graph_scale=1; corrected v2 with graph_scale=0 **during training**.
- Five seeds 1701–1705, exactly matched per-seed FIT document/offset SHA schedules, 120 AdamW updates per arm, batch 8, 128-char contexts, next-character cross entropy, LR .001, gradient clip 1.
- Original and corrected architectures each have 499,290 named learned tensor parameters. The disabled graph makes 512 query/key coordinates functionally inactive; actual first-step backward-active count is 498,778. This 0.103% active-path mismatch is disclosed.
- Source version Git blob (v2 base): `d14c8a0cf562db15db5dcb555460126aba8a4666`. Historical adapter blob: `8815cb8f11a1ab898dc6ac3e2aa8f3ebc0e09cf5`. All training used the actual PR #529 implementation.
- Evaluation: 48 previously exposed DEV windows, 6,144 next-character targets per seed, with document overlap. Training FLOP parity and independent TEST are NOT established; CPU time is reported as an efficiency metric, not a parity requirement.

## Five-seed next-character development results

| Arm | Mean perplexity ↓ | NLL ↓ | Top-1 ↑ | Top-5 ↑ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| Original TAC | **12.6518** | **2.537438** | **32.58%** | **67.31%** | **0.814403** |
| Corrected v2 graph-on | 12.9776 | 2.562773 | 31.76% | 66.82% | 0.821651 |
| Corrected v2 graph-off | 12.9734 | 2.562462 | 31.77% | 66.82% | 0.821620 |

| Seed | Original PPL | v2 graph-on PPL | v2 graph-off PPL |
|---|---:|---:|---:|
| 1701 | 13.2542 | 13.5594 | 13.5508 |
| 1702 | 12.4977 | 12.8797 | 12.8739 |
| 1703 | 12.4123 | 12.6534 | 12.6511 |
| 1704 | 12.2979 | 12.5149 | 12.5156 |
| 1705 | 12.7969 | 13.2804 | 13.2758 |

The corrected graph-on model is **~2.57% worse** in perplexity than the original and loses 5/5 paired seeds; it does not beat the separately trained graph-off control. Frozen **same-weights** graph lesion raises graph-on NLL by +0.000331, so the repaired graph is functional but delivers only a small learned contribution in this test. State-reset gains (reset NLL minus full-history NLL at final 32 real chars of 256-char passages) are negative: original -0.003715, v2 graph-on -0.004547, graph-off -0.004476. No long-history benefit demonstrated.

## Additional real-text historical decoder training

Five new v2 graph-on checkpoints independently continued for 120 additional AdamW updates, batch4 and context256, paired by seed and real FIT-window SHA. Historical arm (503,514 parameters) optimizes next CE + 0.3 average lag64/lag192 historical CE; next-only arm (499,290 parameters) optimizes next CE. The 0.846% parameter gap is within 1%, but objectives differ, so this is not an equal-objective architecture-superiority comparison.

| Metric | v2 history auxiliary | v2 next-only |
|---|---:|---:|
| Next perplexity ↓ | 10.0297 | **9.9887** |
| Next NLL ↓ | 2.305424 | **2.301331** |
| Next top1 ↑ | 37.69% | **37.75%** |

| Historical source target | Corrected history NLL ↓ | FIT unigram prior ↓ | Wrong-ordinal NLL | Zero-read NLL |
|---|---:|---:|---:|---:|
| Lag64 | 4.397644 | **3.532250** | 4.767752 | 4.770394 |
| Lag192 | 4.916371 | **3.633239** | 4.744881 | 4.776011 |

Corrected historical retrieval **fails** both lags versus FIT-only prior. At lag192, the reset-at128 old-source erasure produces the same reported NLL to numerical precision; wrong ordinal is actually better than right ordinal. Training historical reconstruction slightly degrades next-character perplexity.

## Implementation-specific CPU latency

Uncompiled PyTorch 2.10 CPU; two CPU threads; 128-token forward; frozen seed1701; warmup5 + measured32; NOT cached single-token streaming or product deployment latency.

| Arm | Batch1 p50 ms | Batch8 p50 ms |
|---|---:|---:|
| Original | 9.514 | 55.661 |
| v2 graph-on | 10.607 | 61.420 |
| v2 graph-off | 10.307 | 62.348 |

## Reproduction and status

**25/25 local frozen no-refit checkpoint/score replays PASS** across the 15 base and 10 historical fits, including SHA-256 checkpoint/model/optimizer/RNG/sampler records and 36/36 source-byte SHA checks. The separate downloadable research archive contains all 25 original binary checkpoint bytes, exact model source, source protocol, scripts, metric JSON and replay verifier, but **binary checkpoints are NOT yet in Git or LFS**. Therefore `REPO_REPLAY=BLOCKED` until those original bytes are repository-backed and independently restored/scored in clean-checkout CI. This report is a results/limitations record, not an attempt to bypass that compulsory requirement.

Open: independent sealed TEST; true source-identified ≥512-token memory task; real owner isolation; matching 500K SSM/GRU/attention with comparable training-compute accounting; full PyTorch/IEEE/AD and Lean refinement; production inference benchmarks. Preserve both draft PRs and all original negative results. Do not merge/promote the corrected language model on these data.
