# DEV-009 — evaluated first-index gradient correction (DEVELOPMENT ONLY)

**Status: REPLAY_VALIDATED_DEVELOPMENT_ONLY; NO SOTA OR GENERAL CAPABILITY CLAIM.**

GitHub Actions: https://github.com/FPC-effortless/TAC-OSM/actions/runs/38348084111 . All 11 focused integration/index gradient tests passed. Full checkpoint evaluation replay: **6 checkpoints; 576 pre-repair decision episodes and 1,152 independent index-recall diagnostic cases**. Full raw artifact: `DEV-009-RAW-NOT-CONFIRMATORY`. Source files and each checkpoint are verified by SHA-256 hashes; the validator reconstructs every saved evaluation cell and index recall. This is **not retraining replay** or an independently untouched benchmark.

## DEV-009 training change

Compared with DEV-008, both shared CED and sparse CED now receive supervised first-stage index cross entropy on identical synthetic training examples and byte-identical initial weights. Sparse only receives final task CE on steps where the gold item survives hard top-4 selection. This avoids DEV-008's constant -1e6 masked-gold task CE and provides a direct gradient to the indexer. Training loss is therefore **different from DEV-008**, which means direct old/new accuracy comparison is not a controlled architectural comparison. Sparse and shared also have different effective final-task gradient schedules; this factor remains confounded.

Three seeds 43,59,71; 120 AdamW steps each; train on N=8 candidates. Evaluation uses actual hardened loop, deterministic PLM (not learned), two candidate counts N=8/32 and delays D=0/4, 24 episodes per cell/seed/arm. Always evaluate **before repair**. The task is synthetic unique equality and is trivially solvable by exact signature matching using public bits; a nonlearned matching control would be 100% in this setup. No claim of world-model prediction, sublinear addressing or general learned persistent memory.

## Held-out pre-repair exact accuracy

| Seed | Shared + indexCE | Sparse + indexCE |
|---|---:|---:|
| 43 | 20.83% | 34.38% |
| 59 | 20.83% | 63.54% |
| 71 | 66.67% | 63.54% |
| **Mean** | **36.11%** | **53.82%** |

Sparse-minus-shared gaps: +13.54, +42.71, -3.13 percentage points. Seed-level mean +17.71 points, exploratory df=2 95% interval **[-39.92,+75.34] points**. This is inconclusive as a general comparison, with only three seeds and differing conditional training loss terms. All arms substantially underperform the classical exact-match control.

## The most informative diagnostic: first-stage recall@4

| Model | N=8 | N=32 |
|---|---:|---:|
| Shared + indexCE | 98.96% | 86.81% |
| Sparse + indexCE | 100.00% | 99.65% |

These are means across three seed-specific 96-trial tests per size and mode (576 diagnostic cases per mode). The initial indexer **can learn high first-stage recall under direct supervision**, including when evaluated on four times as many candidates as training.

But high index recall **does not translate into high final decision accuracy**. The remaining gap implicates the second-stage attention readout, final selection head, candidate ordering/representation, or insufficient optimization—not first-stage retrieval alone. Pinpointing which is causal requires a new ablation, not a conclusion by elimination.

The sparse model excluded gold during 11, 18, and 8 of 120 development training steps for seeds 43, 59, 71, respectively. Its last reported supervised first-stage CE decreased to 0.00552, 0.00065 and 0.00019, eliminating the catastrophic excluded-gold loss reported in DEV-008. This confirms the original *training-pathology mechanism* was addressed in this small controlled setup; it does **not** establish a general sparse-attention advantage.

## Decision and remaining threats

- **Supported (development-level):** gradients reach the first indexer even when gold is omitted from top-K; first-stage learned index recall can approach 100% in this simple benchmark; checkpoints reproduce 576 pre-repair task cases and 1,152 index-recall diagnostics.
- **Not established:** meaningful improvement relative to exact-match or a fairly parameter/FLOP-matched standard Transformer; learned PLM retention under competing writes; causal environmental future prediction; efficient semantic addressing; real-world applicability.
- **Critical warning:** changing the loss after inspecting a failing previous run is adaptive development, not a preregistered untouched comparison. Three training seeds have large variance. Strong benchmark construction bias remains, especially the simple two-bit equality rule.
- **Next decisive experiment:** fix the candidate-encoder ordering confound and isolate the second-stage scoring head with a direct supervised final-candidate classification control. Then connect the reader to the *learned bounded 768-scalar memory* and noisy-address/competing-write benchmark. Keep a classical exact-address comparator with full cost charged.

Sources: previous `docs/CED-HARDENED-DEV-008-RESULTS.md`; source scripts `run_ced_index_gradient_dev.py`, `validate_ced_index_gradient_dev.py`; DeepSeek-V4.1-Flash https://arxiv.org/abs/2609.19969 .
