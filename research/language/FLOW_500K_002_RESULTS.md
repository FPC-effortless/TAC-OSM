# TAC-LANG-FLOW-500K-002 — Five-seed exploratory model and carried-state streaming

**2026-10-28 · Actual local benchmark completed; NOT scientific promotion.** Source branch is a draft. 500K rule prospective; older 100K studies remain unchanged.

## Genuine-text protocol
- Authentic Vim 9.1 human-authored manual, original 36 chapter **byte SHA-256 manifests verified**. Exactly 30 entire FIT and 6 previously opened DEV chapters.
- 5 paired random seeds 1701–1705, each **120 AdamW optimizer updates**, batch8, 128-character context, 128-symbol ASCII vocabulary. Identical FIT source, document/time windows, SHA-256-verified sample schedules (each seed's schedule matches across all four architectures), next-character categorical cross-entropy, lr0.001, AdamW default weight decay 0.01 and clip norm1.
- Six **not independent and already exposed** DEV documents. Eight chosen windows per DEV document, total **6,144 next-character targets per checkpoint**. Overlapping windows/source repeats mean 30,720 seed-labelled predictions do NOT equal 30,720 independent documents.
- Every architecture in [475K,525K] actual learned count, max relative parameter gap **0.244%**. The prospective 1% model-count parity check passes. **Full training FLOPs, total hardware compute and architecture-tuning budgets were NOT matched or certified; there is no architecture-superiority claim.** Tiny Transformer has only 2 freshly trained layers, not a modern pretrained LLM.
- Model checkpoints were trained at exposed DEV (not independent TEST), and did not perform genuine 512+ specific-fact retrieval or subword benchmarking.

## Five-seed results

| Model | Live parameters | Mean DEV character perplexity ↓ | Top-1 ↑ | Top-5 ↑ | Brier ↓ | ECE15 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Custom diagonal selective SSM | 498,784 | **10.94189** | **36.1621%** | **69.8307%** | **0.77553** | 0.05182 |
| TAC wave/CED 8 wave modes/6 persistent concepts | 499,290 | 12.65183 | 32.5814% | 67.3079% | 0.81440 | 0.05976 |
| GRU (2-layer) | 499,196 | 15.29589 | 32.1126% | 61.6113% | 0.82715 | **0.04810** |
| Tiny causal Transformer (2 layers, 4 heads) | 500,000 | 18.83281 | 23.4961% | 58.5059% | 0.88622 | 0.05138 |
| FIT-only Laplace-smoothed bigram | non-neural | 15.47169 | not audited here | — | — | — |

Per-seed TAC perplexity: 1701 **13.25424**, 1702 **12.49774**, 1703 **12.41234**, 1704 **12.29790**, 1705 **12.79693**. TAC beats GRU/Transformer every seed on NLL and loses to SSM every seed. On six documents, TAC beats FIT bigram on all five seeds for five docs; table-of-contents doc only 3/5 seeds. SSM vs TAC mean NLL 2.39256 vs 2.53744 nats/character; SSM remains strongest among these particular controls.

**Hard science warning:** equal parameter counts, identical observed text and equal optimizer steps do not guarantee equal optimization compute. Synthetic constructed tests only verify implementation mechanics. The results are developmental and cannot qualify real-world long-memory skill or an independent architectural win.

## Reproducibility
- Five TAC and fifteen comparator original Torch .pt state/AdamW optimizer/Torch RNG/NumPy RNG/Python RNG and sampling RNG checkpoints saved LOCALLY in ChatGPT conversation work container.
- **20/20 original SHA-256 checkpoint-byte checks, 20/20 original training-schedule SHA recomputations and 20/20 no-refit numeric score audits PASS.** Non-bitwise float score tolerance up to 1e-6 applied to cross-model scored NLL because batch reductions have small floating differences. All original control causal-prefix tests passed.
- The complete original **binary checkpoints are NOT yet in this GitHub PR**, so GitHub cold checkpoint replay remains OPEN. No merged, production or independent test claim.
- Current original 499,290-parameter training source: `research/language/flow_500k/tac_wave_ced_500k.py`; protocol and source identity in `research/language/FLOW_500K_001_RESEARCH.md`. Exact source SHA256 **acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c**.
- Portable local test/source and original checkpoints are included in the originating ChatGPT session's experiment archive. Corpus text is not redistributed.

## Streaming architecture correction — experimental source only
The originally trained `WaveCED500K.forward(tokens,start=offset)` was **NOT a true carried-state streaming API**; no previous basis/memory/waves/concepts could be supplied. The new additive source `research/language/flow_500k/tac_wave_ced_500k_stream.py` preserves all 499,290 learned coordinates and adds `initial_state`, `return_state`, safe nonzero offset refusal without state, and explicit carry of 88 basis sums, native 8×8, 2×16×8 aligned memory, 8×2 waves, and 6×16 concept seeds = **520 floating values / 2,080 FP32 nominal bytes** plus integer count.

Numerically exercised on the **untrained source and unmodified original seed1701 weights**, not retrained. Local **10/10 focused tests PASS**: legacy no-state inference agrees, multiple chunk partitions through 128 and 256 context agree with full-window forward within declared ~2e-3 conservative tolerance, masked writes, future-token poisoning invariance, stale-state rejection, and representative gradient correspondence within 2e-4. GitHub-source tests do not require original checkpoint bytes; independent GitHub CI for carried state must still run.

**Important distinction**: the model can now preserve and resume a state through 256 tokens, but this test does not establish that it retrieves a fact from 256 tokens back or can work beyond its periodic ordinal alias. Nonlinear decoder, LayerNorm, 6×6 concept solve and learned interference remain.

## PNDS exact-real mathematics
[PNDS PR #53](https://github.com/FPC-effortless/pnds-lean/pull/53) retains 500K state dimensions and adds exact-real 2D rotation squared-norm invariance, bounded damped-wave norm law, pairwise symmetric-concept sum conservation and nonincreasing squared disagreement for constrained diffusion. **These are scoped theorems, not full Torch IEEE, AdamW, six-concept learned-graph or categorical Bayes validation.** New Lean language axiom audit updated for 86 explicit declarations and must be GREEN before reporting the new proof set as checked.

## Required next capability and comparability gates
1. Train candidate and separately trained memory-blind 500K TAC on **actual 512+ token authentic source-identified histories**, with independent fresh whole-document splits, label-availability and old-fact specificity; do not use constructed needle experiments as real performance.
2. Match training *FLOPs* and architecture tuning in addition to 1% actual model parameters, identical factual input and proper task target. Use realistic 500K GRU/SSM/Transformer and FIT-only bigram/conditional ngram.
3. Complete typed PNDS full wave/energy-coupled neural graph proof obligations, finite precision + true Torch JVP/VJP + normalized categorical output law and source-to-source correspondence; periodic 155701 ordinal address alias remains.
4. Commit original model/optimizer/RNG checkpoint binaries and source hashes and verify independent cold GitHub checkout, or explicitly maintain UNQUALIFIED.
5. Keep new work inside six existing TAC families. No seventh flagship.

## Status
Completed: 5 TAC and 15 comparator FIT trainings; frozen-development score replay; new state-carrying inference & gradients; partial mathematical Lean verification. **Not completed**: matched training FLOPs, independent TEST, long-range fact-specific retrieval, model/optimizer Git archive, whole graph Lean–Torch IEEE/AD refinement, deployment validation.
