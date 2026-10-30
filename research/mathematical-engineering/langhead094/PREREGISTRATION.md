# LANGHEAD094 — longer training with real GRU/SSM recurrent controls and a second publisher

**Preregistered on 2026-10-30 before new optimizer steps or any newly selected Rust source outcome reads.** Stacked on draft LANGHEAD093 PR #646 and all earlier original checkpoints unchanged. No merging, model promotion, or independent qualification is authorized.

## Capability question and preregistered falsifier
LANGHEAD093 observed a small −0.00543 NLL (=0.20%) tied-head improvement on three new CPython files after 192 AdamW updates. That signal might be a transient low-budget training effect, or publisher-specific. Does it remain after **384 updates** on a different natural-language/code documentation publisher? How does trained native TAC compare descriptively with an equally sized two-layer GRU and a selective diagonal SSM, once state/input access are made consistent?

This is NOT a qualified matched-compute architecture ranking. FLOPs differ; a later experiment requires **complete prefit and postfit FLOP parity** and possibly optimizer-budget adjustment. GPU costs are zero.

## Four actual independent neural models
- `original`: existing original compact TAC, 100521 learned parameters, native fast/slow 8×8 matrices plus aligned association, waves, concepts, upper recurrence, gated FFN.
- `tied_dot`: identical TAC core plus one trainable output similarity gain, 100522 parameters. Scores observe current real byte embedding x and dot against tied byte-candidate vectors, no future label.
- `gru2`: PyTorch 2-layer GRU of hidden width 83, 32-wide byte embedding, learned vocabulary decoder, 100661 parameters.
- `selective_ssm`: actual diagonal gated recurrent state width128, input-conditioned contractive decay, learned candidate plus gated width154 FFN vocabulary decoder, 100404 parameters. Explicitly NOT a Mamba-3 or Gated DeltaNet equivalent.

The largest/smallest deployed and active parameter ratio is below 1%. No unused parameter padding, frozen pretrained encoder, or privileged oracle inputs.

All 4 use the same observed byte sequence, owner identity and causal recurrent states, train with **identical genuine training target exposure**: original externally published CPython 3.13 fourteen FIT documents plus verified UTF-8 supplement, aggregate source SHA256 `911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4`. Five seeds 1701–1705, frozen NumPy sampler, 384 optimizer updates, batch4, context64, **98,304 real next-byte FIT targets per model**, AdamW LR0.001, WD0.01, clip1, zero tuning trials.

## Publisher-held-out evaluation ownership
**Before model fitting**, the evaluator registered three specific fresh Rust Book documents, independent of the previous CPython-source evaluation owner cohort, and different from the two Rust Book files evaluated historically under LANGMEM086. Rust Book publisher `https://github.com/rust-lang/book`, commit `2771df165f092a4fbb9f46b5b2e7450f19010adb`, copyright MIT/Apache-2.0, Git blob SHA-1:
- `src/ch01-01-installation.md`: `a9e03f534aec9791f6b8200068376fa5678a576c`
- `src/ch02-00-guessing-game-tutorial.md`: `f118eb5455f20aa14128056733a0182869d816d8`
- `src/ch03-01-variables-and-mutability.md`: `ed15afc0fd46a011578cdf41b3e4f03a5e3257fc`

Explicit frozen source-owner manifest fingerprint `3d9edc0302554ca1b1e0cd7339cd613193832504896ba17cd23952580b359134`. These documents are **held out by publisher from training**, but it is **only one different publisher and three developer-grade files**, not protected TEST, independently qualified novelty, or arbitrary generalization.

Each full model+optimizer+RNG+sampler checkpoint must be stored and cold reloaded **before** the Rust document bytes are downloaded and verified by exact Git blob. Each run does a 128-byte causal state burn-in and next-byte predictions for indices128..639, total512 genuine chronological targets/document/model/seed. Across five seeds and three owners this makes 7,680 real next-byte target predictions per arm. The future target is accessed for scoring only after model predictions. No post-evaluation fitting, threshold tuning, model selection, or protected TEST access.

## Independent evidence and honesty gates
Before fitting: reviewed prospective parity manifest, phone-lab placement, true live size, fitted-source hash, dependency/probe boundaries and architecture-agnostic causality, owner isolation, complete-vs-streamed prefix, gradient finiteness. After fitting: preserve **all 20** immutable actual model/optimizer/RNG checkpoints and publisher file bytes in Git; re-check per-seed cross-arm sampler digest and every raw new-publisher next-byte prediction independently on fresh checkout. Report five seed-paired mean NLL and 95% t intervals, accuracy and per-document failure/success rows, actual CPU train/decode time with hardware qualification, parameter counts, state payload and checkpoint digests; do not pseudoreplicate the 3 documents five times.

The workflow must abort/mark BLOCKED on any source, preflight, missing-arm, checkpoint, optimizer or original prediction replay failure, preserving all negative results and failed CI runs. Original pinned artifacts from LANGHEAD093 and earlier must not change.

**OPEN:** entire actual trainable/backward FLOP equality, stronger contemporary Mamba-family architecture parity, PNDS source-pinned full graph Lean/Python proof and reciprocal variant index, human semantic judgments, actual elapsed time, action costs/outcomes, long horizon safety and any model promotion. Prior PNDS #91 covers only the tied-output algebra.
