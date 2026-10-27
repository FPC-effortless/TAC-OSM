# MLAG004 — The 128-position boundary in frozen language TAC memory

Date: 2026-10-28. **Authentic Vim 9.1 human-authored documentation, previously exposed DEV only.** This is a frozen checkpoint stress test, **not trained 256-character recall**, independent TEST, or a language architecture win.

## Protocol and provenance

Replayed 10 original source-checkpoint models (64-only historical objective) and 10 MLAG001 dual-horizon models (64+96), each 100,680 learned parameters, using unchanged source-code SHA256 `835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457`. All 36 original Vim manual chapter hashes, original source training protocol, and **20 checkpoint bytes** were checked. No training and no optimizer updates. The earlier train used only 128-character windows. For this diagnostic, context320 windows are selected from the same **six previously seen whole-document DEV chapters**, 4 deterministic windows per document. All source oracle targets were read from earlier positions in the same real text. Historical lags 64,96,128,256 have 256,224,192,64 target positions *per window*, respectively, so results across lags have unequal evaluation counts.

## Frozen context320 results

Mean historical NLL (nats/character; lower is better) across ten checkpoints per family:

| Requested lag | Old lag64-only fit | New 64+96 fit | FIT-only smoothed unigram |
|---:|---:|---:|---:|
| 64 | 3.569666 | 3.664138 | 3.551061 |
| 96 | 5.525508 | 4.647630 | 3.564737 |
| 128 | 3.732451 | 3.770317 | 3.584431 |
| 256 | 4.433266 | **6.047623** | **3.639902** |

None of these 20 models was trained to make predictions after position 127. The old and dual models both lose to FIT-only unigram at lag256. A **trained 128-context -> frozen 320-context** out-of-distribution transition is *not* a fair head-to-head against a newly 320-context-trained model. The lag96 result on context320 does not cancel the previously verified lag96 improvement on native context128; both observations are true under different conditions.

### Position-stratified analysis isolates the failure boundary

Same 10 dual-horizon checkpoints, frozen 320-window real source. Historical lag is held **exactly at 96**. Split predictions by their *current ordinal position* within the 320-long input:

| Current position in window | Correct-address NLL | FIT-only prior NLL | Aligned-memory zero NLL |
|---|---:|---:|---:|
| 96–127 (within trained range) | **3.328660** | 3.670140 | 4.852030 |
| 128–191 (first unseen positional block) | **4.811247** | 3.677899 | 4.852030 |
| 192–255 | **5.028577** | 3.505290 | 4.852030 |
| 256–319 | **4.762551** | 3.458320 | 4.852030 |

The same lag96 query works relatively well at trained positions 96–127 but deteriorates **immediately beyond 127**, well before the full Fourier address would alias at 155,701. This strongly suggests **training-position regime dependence and/or ongoing-memory interference**, not a simplistic '96 events is too far' explanation. The diagnostic **cannot independently distinguish** learned time-position extrapolation, accumulation drift, and progressive associative interference. Run matched intervention tests rather than asserting which mechanism dominates.

The source operator itself is callable at context320, and a constructed CI test verifies **causal prefix identity** between 320-token forward and 128-token prefix: no future-token leakage; the model returns a finite `[B,64,128]` 256-lag categorical read for a context320 input. **Well-typed computation is not accurate memory.**

## Proposed source-specific fixes and distinguishing experiments

1. **Expose training to positions beyond 127.** Retrain fresh matched groups with context320, not just longer historical labels attached to a context128 dataset.
2. **Separate horizon supervision from context exposure.** Match the 320-context control (`0.15 L64+0.15 L96`) against the 320-context test (`0.10 L64+0.10 L128+0.10 L256`). Both retain the same main-language objective, 100,680 learned parameters and total historical coefficient 0.30. Use the *same* genuine source FIT/DEV documents, seeds, actual 320-example sampling order, optimizer settings and update count. Each newly trained arm must be stored as hash-verified checkpoints with AdamW and RNG replay. Document per-lag target counts and compute; two vs three history heads incur unequal FLOPs, so do not claim FLOP parity.
3. **Locate why the cliff occurs.** Within each fixed-lag evaluation, measure position-stratified retrieval NLL and true-key vs wrong-key/zero-memory on identical target coordinates; collect source key norms, retention/gate dynamics and source-memory interference, and causally compare local-window resets against continuous nonreset streams. Resets change task definition and cannot count as uninterrupted memory.
4. **Protect generation.** Track next-character likelihood, top1 and true-label probability tails; do not trade a long-history retrieval number against worsening language generation. Compare FIT unigram and bigram, parameter-matched strong baselines, and explicit external-memory storage costs.
5. **Seal independent TEST.** Old six DEV chapters have been repeatedly examined. Genuine multi-session/temporal reasoning and NoLiMa/LongMemEval-style evaluation require appropriate model scale and authorized source data; a character-level 100k model cannot credibly claim those capabilities yet.

No parameters or checkpoints were changed in MLAG004. The source-position cycle limit (155,701) and FP32 key sensitivity remain separate, longer-range math risks; they do not explain a cliff at position128 by themselves.
