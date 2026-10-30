# LANG10M001 — tokenized 10M TAC prefit research candidate

**Status:** no-fit implementation; exact source-grounded parameter-count and
causal/gradient CI checks must pass before any measured-engineering claim.
No tokenizer FIT, neural training, GPU allocation, new predictions on authentic
data, checkpoint, protected TEST, or qualified language gain from this branch.

## Scope and separation

- This variant is independent of **CODEVALUE004** (PR #657), which is running
  its frozen 500K corpus×LR factorial protocol. Neither its source graph nor its
  24-model Kaggle run is changed or resubmitted here.
- Parent is the frozen CODEVALUE003 implementation (PR #656); existing scores,
  negative findings and checkpoints remain unmodified.
- A draft implementation is **not** permission to deploy or promote a model.
- The global ~100K budget is NOT silently changed. A separately registered
  owner-scoped exception will be needed for each future 10M **training** study,
  with match-within-1% controls, original authentic FIT/DEV sources, replay
  checkpoints, causal anti-cheat, PNDS scope and the phone compute planner.

## Exact proposed architecture

Source: model.py, variant ID **LANG10M001**, 10,001,188 trainable/deployed
parameters (verify on actual constructor, do not trust arithmetic alone).

- Token IDs 0..4095, raw source mapped with a *separately FIT-trained*
  4,096-piece byte-fallback BPE SentencePiece tokenizer.
- Inherits CONTEXT026 first-stage 48D embedding, 8×8 native memory,
  two 16×8 source-aligned banks, eight 2D wave modes, six 16D concept slots,
  88D fast/slow transforms, and source-conditioned readout.
- Keeps first-stage memory geometry **fixed** deliberately; changes vocabulary,
  depth and FFN capacity. This is not a balanced all-axis scaling claim.
- Adds four causal RecurrentFFNBlocks, each with 280D input/output,
  128D recurrence, and 1,024D residual SwiGLU FFN.
- Final gated vocabulary head FFN width 1,214, untied 4,096-token output.
- The first-stage concept Laplacian is retained; upper blocks are graph-free.
- The **whole model resets each window**. No persistent whole-model streaming
  API, O(1) incremental decode guarantee, or long-context advantage exists
  merely because upper blocks use recurrence.

The resulting output projection is expensive. This prototype is a controlled
direct descendant, not a finished optimized design. Future candidate
architectures should compare tied/adaptive output heads, embed/core width,
and memory-state scaling, but require separate registrations and ablations.

## Exact parameter arithmetic

Let the 500K parent be 499,623p with 256-byte vocabulary and one 48D
upper recurrence. Subtract its 37,129p upper block, replace it with the
original first-stage 280D layernorm, expand embedding/pooled-concept/output
heads to 4,096 tokens, and add four (128D recurrent + 1024D FFN) blocks.

For output FFN width W, predicted count = 4,346,376 + 4,658 W,
including all embedding and decoder parameters. With W = 1,214,
the result is exactly 10,001,188 parameters. Confirm on the real model.

No parameters are frozen, padded, or excluded from the deployed total.

## FIT-only tokenizer plan

- Allowed data: original, license-checked, SHA256-verified, locally present
  UTF-8 text/code FIT sources. Do not train on DEV, TEST, target suffixes,
  publisher IDs, or a future cohort. An explicit manifest lists every file
  and FIT-only partition before tokenizer fitting.
- Tokenizer: SentencePiece BPE, vocab 4096, byte fallback, identity
  normalization, no dummy prefix or whitespace stripping; exact code/text
  roundtrip checks. Hard vocabulary limit, fixed special token IDs.
- Run from the repository root after preparing real FIT manifest:
  python research/language-scaling/lang10m/tokenizer_fit.py \
    --fit-manifest /path/to/authorized-FIT-manifest.json \
    --model-prefix /path/to/output/lang10m-bpe4096
- Never silently overwrite output. Archive tokenizer model, vocabulary,
  fit manifest/source hashes, precise recipe, and output hashes.
- Evaluate tokenizer compression on mixed authentic text and code, unicode,
  whitespace and Python indentation; compare a 256-byte control using
  byte-normalized NLL/BPB, not incomparable token-level perplexity.

## Kaggle preflight and full training (future, NOT started)

1. Run CPU tests: parameter count, causal prefix poison, gradients, FIT-only
   source boundary and tokenizer roundtrip.
2. On a **separately authorized** private offline dual-T4 Kaggle allocation,
   verify two physically distinct Tesla T4 UUIDs and CUDA devices before
   a bounded no-update throughput profile. Do not infer total 32GB per replica.
3. Measure actual B×T activation VRAM, backward peak, training examples/sec
   and tokens/sec for sequence lengths 128, 256, 512, 1,024 and safe
   microbatches. T4 uses FP16 (not native BF16); validate AMP parity and
   numerical stability before relying on it.
4. Pair TAC with fair ~10M GRU, simple SSM and/or attention controls using
   same tokenizer, authentic data/splits, exposures, learning-rate selection
   access, complete checkpoints and matched evaluation. No architecture
   superiority claim without adequate compute treatment; report FLOPs.
5. Only if real throughput, reproducibility and budget support it, schedule
   pretraining. An illustrative 200M-token target is a planning heuristic,
   not an authorized run or guaranteed adequate data.
6. Measure genuine held-out quality, long-context statefulness, byte-normalized
   language loss, exact source provenance, decode batch1 p50/p95, RSS/VRAM,
   and eventually Android efficiency. No simulated token fixtures count as
   predictive capability.

CODEVALUE004 can inform data selection and LR policy once its original
evaluation is actually complete; do not edit its preregistered protocol.

## CI and mathematical status

    python -m pytest -q research/language-scaling/lang10m/test_lang10m.py

This is **constructed no-fit engineering validation only**. PNDS shared-kernel
math is inherited, but exact full source-equivalent Lean/Python math,
floating-point correspondence, gradients and composed qualification are OPEN.
A reciprocal PNDS variant record/PR and manifest linkage are required before
claiming a complete new mathematical model specification.

The anti-cheat diagnostics registered here cover only implemented CPU source
tests. Every future trained-data pipeline must separately audit raw
publisher bytes, partitioning, tokenizer FIT boundary, causal masks,
sampling, optimizer, checkpoint replay and training data provenance.
