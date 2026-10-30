# HYBGEOM091 — trainable hybrid/contextual geometry versus compact native TAC

**Prospective, 30 October 2026; preregistered before any optimizer step.** Stacked on negative frozen-weight no-fit HYBGEOM090 (PR #638); do not modify original LANGMEM086/087 checkpoints, PR #638 results or protected TEST.

## Research question

Does allowing TAC-OSM's native 32D upper state to evolve under a locally stable two-mode continuous-time flow **between observed events**, and allowing contextual Euclidean / learned rank-one positive-definite metrics on byte embeddings to reshape next-byte logits, recover the previously measured negative frozen-weight behavior when fitted end-to-end? Answer only for next-byte modeling on publisher-authenticated training and **exposed** DEV documents. A rank-one SPD context metric is not a learned Riemannian geodesic; byte labels are not human semantic judgments. An ordinal is not a real wall-clock time measurement. No claim of improved adaptation, reasoning, causal intelligence or whole-model Lyapunov stability.

## Frozen protocol

Six independent architecture implementations all use unchanged original compact 100,521 parameter native TAC graph with the 8x8 fast/slow matrices, aligned rectangular memory, wave, concepts, recurrent upper state and decoder:

1. Original TAC, 100,521 trainable/deployed weights.
2. Fixed Euclidean distance-decoding strength, 100,522.
3. Context-adapted Euclidean with learned gamma and strength, 100,523.
4. Context-adapted rank-one SPD metric with learned gamma, alpha and strength, 100,524.
5. Two-mode hybrid flow with positive learned per-mode decay rates and bounded learned reset, 100,524.
6. Hybrid plus contextual SPD, 100,527.

Extra weights are real prediction-path scalars, not unused padding. Parameter spread <0.006%. All six use *exactly the same* sampled observed token data, seeds 1701–1705, optimizer AdamW(lr0.001, wd0.01), global gradient clip1, 96 gradient updates, batch4, 64 next-byte targets each sample (=24,576 per arm per seed), and 0 tuning trials. All are trained **from scratch**; previous frozen checkpoints are not initialization. FIT source: original 14 CPython 3.13 publisher documents + UTF-8 supplement, source digest `911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4`. Original source validation checks each file's publisher blob, SHA-256 and size. Frozen deterministic schedule uses NumPy default_rng(seed), selecting owner and offset uniformly from eligible original FIT documents. Identical seed means same original sample order/bytes; the result aggregate checks this per seed across arms.

Exposed DEV source: original pinned CPython publisher `Doc/tutorial/controlflow.rst`, `Doc/tutorial/venv.rst`, 224 targets per document at ordinal 32..255, predicted before opening byte t+1. Valid independent outcomes are genuine published next bytes, not real actions. Both DEV source files were previously exposed in this research program; this is **not unseen generalization**. Two documents are reused across five seeds; statistical confidence intervals use five seed-level differences and do not count 10 document/seed rows as independently sourced documents. No TEST and no publisher-source retuning based on the result.

## Mathematical implementation

For each observed byte, `mode_next` uses LF(10)->1 and period(46)->0; otherwise carry mode. Before processing that byte, the previous upper state flows exactly over one **ordinal** interval via `upper^- = upper*exp(-softplus(rate_q)*1)`. On switching, apply `upper^+ = sigmoid(reset)*upper^-`; both are norm-nonexpansive locally. Then the completely original TAC state update and output head process the observed byte. Other original native state components stay unchanged. This is a bounded local flow/reset property only; the original full TAC recurrent graph is not formally stable by this construction.

Context c is the causally updated normalized 32D upper state. Original candidate byte embeddings e_a become z_a(c)=e_a+sigmoid(gamma) <e_a,c> c (or fixed when gamma=0). Rank-one SPD metric is G(c)=I+softplus(alpha)c c^T, eigenvalues >=1. Prediction logits receive `-0.02*sigmoid(strength)*(distance_squared - mean_distance_squared)` where reference is the *current observed byte*, never the next target. No trainable geometry or rate parameter is concealed from parameter or checkpoint accounting. Call this a context-dependent logit correction, **not an inferred shortest-path geodesic**.

Prediction-time input boundary: observed byte, its causal native recurrent state, source owner and optional preobserved dt only. Language study `dt=1` (no real timestamp). Exposed DEV targets are read after producing logits. No externally selected historical key, future target, global owner cache, or evaluator outcome enters the forward method. Run actual-source future-suffix poison, complete-vs-incremental parity, per-owner batch isolation, finite gradients, SPD eigen and reset guards. For new architecture proof obligations, source-specific research adapter and full PNDS Lean/Python reference equivalence remain open; isolated mathematical identities don't qualify them.

## Decision and evidence

Primary diagnostic: five-seed **paired** exposed-DEV mean next-byte NLL differences versus original TAC, with per-seed differences, seed-level 95% t interval, accuracy, training CPU time, incremental CPU decoding time and parameter count. Do not select a winner if interval contains zero, or when there is unmatched compute. `study_type=single_model_training`, `no_architecture_superiority_claim=true`: train FLOPs differ because attention-like geometry and hybrid state flow add real operators. This is explicitly **equal-exposure exploratory comparison**, NOT the strict architecture-comparison FLOPs-parity gate. A later rigorous model-family win requires a separate matched-FLOPs preflight, longer contexts, stronger controls (GRU/SSM/Transformer, exact history) and untouched publisher groups.

All 30 full original model+optimizer+RNG checkpoints, sampler hashes, honest per-target predictions and metrics must be committed to the *unmerged branch* and cold-replay checked. GitHub artifacts with temporary retention alone do not qualify. A failed job, missing checkpoint, mismatched source or uncommitted archive means `REPO_REPLAY=BLOCKED`; preserve all failures. Do not merge, consume Kaggle GPU minutes, access protected TEST or promote a checkpoint. Run preflight before any fit and produce final counts/archival receipts afterward; execute internal research without independent reward/safety certification.

## Distinct unrun primary claims

Genuine real-time adaptation with timestamps/action outcomes; held-out context-annotated human semantic judgments; causal policy utility; multiagent macro-scale closure; full original state+reset ISS/Lyapunov property; cross-repository Lean4/executable-Python complete graph equivalence. All **OPEN** and not inferred from CPU language-score changes.
