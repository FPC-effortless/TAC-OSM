# TAC-LANG-FLOW-500K — five-seed language model, independent controls and streaming repair

**Date:** 2026-10-28. **Evidence status:** completed genuine-text, previously exposed development study; NOT an independent architecture-superiority or genuine long-memory result.

## Architecture / prospective policy

The prospective new-model budget is **500,000 active learned coordinates**, within 475,000–525,000; comparator counts must differ by no more than 1%, and historical 100K experiment registrations remain unchanged. The wave/CED language source contains **499,290 trainable parameters**: an input-only causal 88-dimensional basis prefix, one native 8×8 associative memory, two source-aligned 16×8 memory heads, eight damped two-coordinate wave modes, six persistent 16-dimensional concept seeds, a postscan six-concept positive-definite graph energy solve, and a local gated SwiGLU-style language head. Recurrent floating state: **520 FP32 numbers**, nominally **2,080 bytes** plus metadata; source/computation cache is separate.

The architecture is not a proof of active inference, automated reasoning, or long-history factual retrieval. It is a mathematical and software candidate under the existing six research families.

## Authentic dataset and frozen experiment

Vim 9.1 user manual, 30 genuine human-authored FIT whole chapters and 6 previously exposed DEV whole chapters. Original 36 publisher-file byte SHA-256 hashes reverified. Eight DEV windows per document, 128 character positions each; exactly **6,144 previously exposed DEV next-character targets per model and seed**. The five seeds are independent initializations and sampler schedules, not independent institutions/documents; within-document windows overlap.

Five original seeds 1701–1705, 120 AdamW updates, batch 8, context 128, next-character cross entropy, learning rate .001, optimizer default decay .01, clip norm1, equal exact sampled FIT documents and positions within each paired seed, no sealed TEST. Matching parameters: 498,784 selective-diagonal custom SSM; 499,290 TAC wave/CED; 499,196 GRU; 500,000 scratch small causal Transformer. Parameter gap max 0.244%. **Complete training FLOPs/optimizer compute and architecture tuning equalization NOT established.** This is an exploratory controlled comparison and **cannot justify a fully fair architectural winner**.

## Actual held-out probability scores

| Metric | Custom SSM | Wave/CED TAC | GRU | Tiny Transformer |
|---|---:|---:|---:|---:|
| Mean character perplexity ↓ | **10.9419** | 12.6518 | 15.2959 | 18.8328 |
| Mean NLL, nats/character ↓ | **2.39256** | 2.53744 | 2.72739 | 2.93553 |
| Bits/character ↓ | **3.45174** | 3.66075 | 3.93480 | 4.23508 |
| Top-one accuracy ↑ | **36.16%** | 32.58% | 32.11% | 23.50% |
| Top-five accuracy ↑ | **69.83%** | 67.31% | 61.61% | 58.51% |
| Categorical Brier ↓ | **0.77553** | 0.81440 | 0.82715 | 0.88622 |
| 15-bin ECE ↓ | 0.05182 | 0.05976 | **0.04810** | 0.05138 |

FIT-only nonneural smoothed character bigram: **15.47169** perplexity. TAC's seed 1701–1705 perplexities: **13.25424, 12.49774, 12.41234, 12.29790, 12.79693**; every TAC seed beats GRU/Transformer and loses to custom SSM in this limited schedule. In five of six DEV documents, all five TAC checkpoints have better NLL than the FIT-only bigram; in the table-of-contents document only three do. None of this establishes a generalized language skill.

**Exact original checkpoint provenance:** 20/20 original checkpoint-byte SHA-256, model/optimizer/RNG loads, original training-sample SHA-256 reconstruction and frozen no-refit heldout scores verified. Some float NLL scores differ by ~2e-7 due to reduction rounding; cross-architecture replay tolerances explicitly **1e-6**. Control causal-prefix tests pass. Original 20 checkpoint bytes and code are in the attached full local ZIP; **not yet committed to GitHub cold CI**. Copyrighted source documents are excluded.

## Genuine carried-state streaming correction

The initial trained PyTorch code accepted a positional start offset without carrying the previous state. A separately named source-compatible revision (same 499,290 parameters, old checkpoint weights unmodified) now carries native 8×8, aligned 2×16×8, all wave modes, previous 88-basis accumulator, six 16-dimensional concept seeds and count between chunks. It uses chronological affine prefix composition with nonzero initial states and exact masked writes. All tests compare to complete full-window forward and reject state/count mismatches.

**10 local numerical tests passed**, including old no-state inference preservation, 256-character split/unsplit correspondence, multiple 128-token partitions, masked write equivalence, future-token poisoning, and representative gradient correspondence (unoptimized FP32 tolerances). **Nine source-controlled tests passed in TAC-OSM GitHub CI.** This does not teach the model historical recall; it only fixes legal continuation mechanics. There is no explicit stable owner ID in the raw Vim tokens, and source boundaries must be enforced by the caller.

### Frozen real-data 256-character state-carry value diagnostic

For each DEV document, select eight genuine 257-character passages, score the same last 32 actual targets after (i) all preceding 256 characters or (ii) only last 128 while retaining correct absolute ordinal 128–255. This is a *frozen-model diagnostic*, not the independently preregistered 512+ historical fact task. Across five seeds:

| Same-target condition | Mean loss, nats/character ↓ |
|---|---:|
| Full 256-character carried state | 2.729155 |
| State reset after 128 characters, ordinal preserved | **2.725440** |

Mean gain (reset loss − full loss) **−0.003715 nats/character**, only **two of five** seeds benefit from earlier retained state. **Negative long-history-utility evidence** for this setting. It does not prove older facts cannot be learned or retrieved under a different loss/data scale.

## CPU performance (implementation-specific)

PyTorch 2.10 eager CPU, 2 threads, full-window inference at 128 characters, no cached single-token generation, one frozen seed each, 32 measurements after warmup:

| Model | Batch1 p50 ms ↓ | Batch8 p50 ms ↓ |
|---|---:|---:|
| Tiny Transformer | **1.315** | **10.960** |
| TAC wave/CED | 6.279 | 35.601 |
| GRU | 7.581 | 20.184 |
| Custom SSM | 12.102 | 23.148 |

TAC's 6×6 solve and multiple associative scans incur higher batching costs here. No FLOP parity or compiled/fused operator comparison.

## Mathematical formalization

Source-affine right-composition and 520-scalar typed state are in PNDS #53. Exact-real Lean statements added for two-coordinate rotation norm preservation, damped wave nonexpansiveness under 0≤rho≤1, pairwise symmetric diffusion preserving sum and reducing squared disagreement under 0≤eta≤1. These conditional *operator-level* claims do not certify the entire nonlinear six-concept energy network, PyTorch IEEE math, optimizer, calibrated softmax probabilities, full gradients or useful long memory.

The dedicated language axiom audit originally failed from stale 83-theorem receipt after three new declarations were added; it was repaired to expect 86. **Report the latest source-pinned CI verdict separately.**

## Key remaining scientific work

1. Train long-context *source-specific* historical reconstruction on authentic verified texts with meaningful available-at-time references at ≥512 positions, wrong-key and false-positive controls and isolated next-character benefit.
2. Give every 500K comparator the same supervised targets, history access, loss mixture, exact parameter band and genuinely comparable full forward/backward/optimizer FLOP budget. Document actual hardware, batch size and throughput separately.
3. Prove wave modes, memory operator, concept equilibrium, autograd and finite-precision bounds end-to-end in PNDS, including state owner/stream boundaries.
4. Commit source and **actual original** 20 binary checkpoint/optimizer/RNG artifacts into TAC's replay archive; independent cold GitHub action replays score on immutable lawful source.
5. Evaluate on independently held-out unseen authentic documents before promotion. Maintain six-family rule and retain the 100K historical results.