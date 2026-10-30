# LANGPARALLEL096 — primary layered parallel language TAC vs non-layered parallel control

Registered before any new scoring, 30 October 2026. **NO NEW TRAINING**. This uses existing **actual parallel-scan language TAC** implementations and original trained model artifacts, not the sequential compact LANGFAST095 model.

## Architectures (historical 500K research only)

**Non-layered:** CONTEXT026, 500,122 learned weights, publisher training original LANGLAYER035 action run 40785939365, 5 seeds 1701..1705. Input-only parallel Hillis–Steele affine prefix scan for 8×8 native associative, two 16×8 aligned heads, eight 2D waves, six 16D base concept seeds, and a shared byte decoder. Reset windows, no streaming carry.

**Layered parallel:** LANGLAYER038/039, 499,901 learned weights, original 40795790808 with the *same* publisher FIT sampling per seed 320 AdamW updates, B=8,T=128. Preserves all native first-stage memories and six base concepts, then applies an independent **59D recurrent scan layer** after its LayerNorm, before the narrower shared decoder. Additional upper CED/concept graph is intentionally **absent** after existing 036–040 evidence. Depth=2, not just two copies of the original 8×8 native TAC block.

The prospective 95k–105k model size remains binding. These existing 500K model artifacts were created under specific earlier exceptions; **no model fitting or size-policy exception** is introduced by this diagnostic. Matched training **FLOPs are unverified**. Source, seeds, optimizer steps, training data and test conditions are matched; CPU process time is a separate efficiency outcome.

## Original real-world publisher inputs

Training source is the immutable PSF CPython3.13 original 14-owner FIT+UTF8 supplement; same 320 update publisher windows, checkpoint hashes, sample-order digest, RNG/optimizer/model states are verified across both arms. No original checkpoints are overwritten or reconstructed from metrics.

Evaluation is three **previously exposed** authentic MIT/Apache-2.0 Rust Book documents, official immutable commit `2771df165f092a4fbb9f46b5b2e7450f19010adb`, publisher Git blobs:
- installation `a9e03f534aec9791f6b8200068376fa5678a576c`
- guessing-game `f118eb5455f20aa14128056733a0182869d816d8`
- variables `ed15afc0fd46a011578cdf41b3e4f03a5e3257fc`

Freeze 8 target ordinals/document (256,384,...,1152) and exactly same next-byte labels, owner identity, 64- and 128-byte causal contexts, no future labels, no cross-document mixing. Each original trained model gets 24 unique real targets per length; 120 target predictions/arm across five seeds per length. This is narrow developer-source engineering evidence, **not blind TEST, not independent transfer qualification**.

## Performance and integrity

Warm both historical models; time five full 4×128 reset-window batches on same Github hosted CPU, 2 threads and same genuine source bytes. Report per-seed paired full-window batch processing CPU ms/observed byte. This **must not** be relabeled incremental streaming/decode speed, on-device latency or equal compute training.

Integrity checks: original archived model/optimizer/RNG (sha256) and actual sampler parity per seed; exact immutable Git publisher bytes, source hash, all models same target offsets; preflight causal future poison, rejection of illegal carry and actual parallel scan vs serial mathematical unit test; all ten score result rows required; no incomplete aggregate. State/context changes at different T do not prove persistent memory. No protected TEST, GPU use, paid services, merges, checkpoint promotion, or qualified architecture winner.

**PNDS:** Existing LANGLAYER034–040 scoped Lean/Python math records are inherited as references, not new full graph proof. New mathematical operator or model variant is **not** created here. Full actual source graph, finite-precision, gradients, masks and training-FLOP equalization remain OPEN. The previous compact sequential LANGFAST095 branch and negative tied decoder evidence remain unaltered.
