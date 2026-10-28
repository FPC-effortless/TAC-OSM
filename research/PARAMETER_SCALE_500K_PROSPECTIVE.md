# Prospective TAC-OSM learned-parameter scale — 500K (2026-10-28)

**Owner directive: the compulsory active/deployed trained-model scale is now 500,000 parameters for NEW experiments.** This is a prospective replacement of the prior approximately-100K scale. Do not alter old 100K source, benchmark claims, validator outcomes, archived pre-registrations, checkpoint hashes, comparisons or tests.

## Strict prospective model gate
- Target: **500,000** trainable and deployed learned parameters per model.
- Band: **475,000–525,000** (±5% of target). Every active learned tensor counts exactly once if shared; no inert/dummy parameters, unconnected padding, or frozen tensors counted as active.
- Within a comparative study, every architecture must be within **1% of every other architecture's active parameter count** (relative gap <= 0.01). Also ensure all models lie in the band; choose near 500,000 when practical.
- Check real model instances and separately report learned, trainable, active, frozen, optimizer, recurrent FP32/metadata, source cache and inference working memory. Static distributions or lookup tables are not learned parameters and must be separately costed.
- True architectural superiority also requires equivalent factual available-at-time inputs, authentic FIT data, independent never-opened TEST, equal seeds, optimizer/update schedule, training compute/FLOPs and tuning budget, and identical objectives or explicitly classified non-comparable arms. CPU wall-clock equality is not required.
- Must preserve original model/optimizer/RNG/data source SHA and independent checkpoint and score replay in Git. Synthetic diagnostics are never primary real-world success evidence.
- Incomplete formal equivalence, AdamW, finite precision or Lean proof remains OPEN, not PASS.

## Compatibility

The older **95,000–105,000** gate and **1%-matched** comparisons remain the binding HISTORICAL controls for existing studies registered up to and including 2026-10-27. Validators for those versioned studies and old recordings must not be retroactively changed. New-only budget validation code shall key off a versioned study date/scale contract and reject new studies requesting the retired target, while allowing independent replay of archived originals.

## Language + wave/CED adaptation

TAC-LANG-FLOW-006 remains an *untrained* mathematical proposal within the six-family architecture, not a seventh model family. For the 500K candidate keep the exact source-authorized read-before-write boundaries, right-affine chronological scan of input-only gates, bounded wave modes and concept-seed state, optional energy-decreasing postscan CED and a normalized categorical law. Do not increase model size by simply adding unused layers or duplicate attention scores. The first 500K model must show end-to-end actual active parameter counting, a causality/batch-stream/gradient numerical receipt and a full mathematical source contract in PNDS before performance promotion.

For the latest 500K research progress and its open gates see TAC-OSM research language issue #525 and PNDS issue #52; this policy is prospective even if old master-plan sections below cite ~100K.
