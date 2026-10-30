# LANGFAST095 — no-fit original checkpoint performance surgery

**Prospectively registered 2026-10-30 before timing or numerical results.** Stacked on LANGHEAD094 draft PR #648; no earlier original checkpoint/source/result is mutated. Novel model-training FLOP matching remains OPEN; this is *solely a no-fit implementation optimization*.

## Isolated code changes

Retain the original CompactDualClippedTAC086 native forward computational graph and all trainable weights: fast and slow 8×8 matrices, aligned associative memory, wave, basis, four concepts, upper persistent state, FFN, byte embedding and original attention/finite-state controls. Derive optimized095.py directly from frozen original source Git blob `1e072193e6ba89a46d8bd33e65dd207843558402` and rename the class for isolated testing, without editing original source.

1. Original validate() allocated a fresh nine-tensor zero-filled start state every forward call merely to obtain expected tensor shapes and owner checks. Optimized validate() checks those same literal original expected shapes and identical finite checks without dummy tensors, explicitly retaining owner uniqueness/clock/byte validation.
2. Reuse **nonpersistent** 4×4 and 8×8 exact identity buffers instead of allocating them on each forward call. These buffers are not trainable, remain excluded from the state_dict and original checkpoint, and follow PyTorch device/dtype conversion rules.

No fuzzy state compression, simplified recurrence, approximation, new data keys, source oracle or selective fast memory bypass is permitted. No model weights or optimizer steps change.

## Falsification

Load three full publisher-source-attested 384-update **original** TAC seed checkpoints from LANGHEAD094 replay bundle. For each, original Rust Book source document `src/ch01-01-installation.md`, pinned Git blob `a9e03f534aec9791f6b8200068376fa5678a576c`. Verify original original model Git blob, exact source/checkpoint bytes, names and original weight hashes. Compare 256 genuine observed bytes sequentially with **bitwise identity** of each 256-way logit and all nine persistent state tensors. Require identity also for full multi-byte calls, all five registered lesion conditions, two distinct independent owner streams, and copied original model gradients under same input/loss. Reject changed error paths including missing state keys, nonfinite states, duplicate/empty/invalid owner identities. If any numerical or gradient check fails, discard optimized performance claims and preserve failure.

## Measurement

Use exactly one-thread CPU FP32 torch==2.10.0 on GitHub-hosted runner, 5 paired original/optimized repetitions of 192 real observed bytes ×3 original seeds, randomized order per seed. Warm both models before each paired block; timing via process_time for CPU, report seed medians, median-of-seed-medians and paired differences. Record CPU profiler top 22 operators for both original and optimized at seed1701. **Never** report GPU, mobile runtime, energy, wall-clock guarantees or claim semantic/perplexity improvement. This CPU measurement is not matched-train-FLOP qualification. Both variants MUST have 100,521 unchanged actual learned parameters.

Persist the generated original benchmark JSON source-controlled, independently repeat source/checkpoint bitwise replay on fresh checkout, preserve negative speed results. No protected TEST, no Kaggle quota, no model training, no merges or promotion. PNDS mathematical proof scope unchanged because recurrence equations are the same; exact original full graph math and numerical gradients remain open beyond tested checks.
