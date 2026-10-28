# LANGCOMPARE029: exact-source trained benchmark execution contract

**Status 2026-10-28: SOURCE-PINNED, NO FIT, NO DEV SCORING.**

This draft implements three separable layers that must not be conflated.

## 1. Current verified CPU-only engineering

- `scripts/langcompare029_preflight.py`: exact original model types, active/deployed counts, source ownership, future-prefix probes. The Mamba-3 CUDA class is *not* instantiated on CPU. Uses the same immutable current `CONTEXT026` source.
- `scripts/langcompare029_sampling.py`: genuine PSF CPython source-owner selection, five original seeds, 320 × 8 × 128 FIT next-byte targets each, all five source sample-order hashes equal to those preserved in original LANGTRAIN025 checkpoints.
- `scripts/langcompare029_mamba_ledger.py`: source-derived 501,884 Mamba-3 SISO analytical count (per-block mixer 45,754 + FFN 30,720 + layer norms 320; six blocks plus embeddings/head/norm). **Not a CUDA-verified count.**
- `scripts/langcompare029_checkpoint.py`: safe serialized model/optimizer/torch/CUDA/Python/NumPy RNG bytes; exact hash and FIT identity verification; zero-update diagnostic cold load and identical logits. **Not a trained replay.**
- `scripts/langcompare029_source_pins.py`: official original source Git blob identity for Pathway BDH and upstream Mamba-3.

## 2. Future real FIT launcher — currently fail closed

`python scripts/langcompare029_train.py --mode dry-run --arm tac_context026 --seed 1701`

Returns **BLOCKED_NO_FIT** while incomplete. The draft GitHub CPU workflow does *only* dry-run, never an optimizer.

`--mode fit` exists for a future independently reviewed execution:
- all four model implementations and actual parameters known and within 1% at 500K; no frozen/dummy padding;
- machine-validated `FAIR_COMPARISON.json` with source, five registered seeds, identical original sample order, equal exposure and fair independently profiled FLOP/optimizer/tuning budgets;
- every necessary immutable actual CUDA/hardware/venue receipt and operator-protected `PHONE_LAB_GPU_EXECUTION_ENABLED=true`; exactly 2 BF16-capable SM80+ devices, each seed assigned to GPU index derived deterministically from its seed;
- unique checkpoint output (no overwrites) containing original optimizer/RNG and sampled FIT provenance.

The current Kaggle fixed two-T4 SM75 route is **not compatible** with the official Mamba-3 BF16 kernel. Approval of another class of GPU is a separate operator/research-plan action, **not** something source control may silently grant.

The future launcher trains **one arm and one seed only** and never opens DEV or TEST; trained checkpoint status still requires a later independent clean Git replay, complete five-seed four-arm cohort and postfit matched compute receipts.

## 3. Scientific interpretation

Historical older LANGTRAIN025 native NLL **2.099349221**, prior P3 **1.877306112** remains a 0/5-seed loss. Those weights do *not* represent CONTEXT026. A no-fit checkpoint and source/shape/casuality/gradient PASS do not imply language skill.

The ultimate comparison must report authentic original-source pooled next-byte NLL, bits per byte, Brier, top-1, top-5, per-document NLL, memory, latency, actual training FLOPs and every failed run. The four DEV documents have **already been exposed**; they are internal development evaluation, not an independent generalization test.

Do not calculate or publish a four-way winner from partial, unmatched or synthetic runs. Do not unlock protected TEST, promote, merge or deploy from this draft.
