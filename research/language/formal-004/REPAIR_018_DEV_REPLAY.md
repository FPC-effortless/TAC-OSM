# TAC-LANG-REPAIR-018 — Real-source historical retrieval reliability

**Status:** exploratory, frozen-checkpoint real-document DEV replay. This is NOT independent TEST, a new trained model, or proof of long-history retrieval improvement.

## What motivated the repair

TAC-LANG-GATE-003 trained the historical reconstruction objective **only at exactly 64 characters**. The source code permits querying other delays, but being mathematically queryable does not establish that they were learned. Previously exposed DEV results showed severe degradation at lag 96.

The first suggested policy trusted all delays ≤64; that was too broad. The committed fail-closed policy now **trusts only lag=64** and marks ALL other delays (including 1–63 and >64) unqualified until separately trained and evaluated.

## Actual source data and immutable replay

- Authentic Vim 9.1 manual: **30 FIT chapters, six whole-document DEV chapters**, with all **36 byte SHA256 digests verified** against the frozen original experiment `PROTOCOL.json`. Not a general language benchmark.
- Original `parallel_tac_language_gate.py` SHA256: `835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457`.
- Original `PROTOCOL.json` SHA256: `181936efb2bd7513597180455079590289ccbf9150381bea5625f489faecb2f1`.
- Original `RESULTS.json` SHA256: `40759a5c8c4c12ea39546d1af9051a7f8c00a330fc01295a91c2aad6445aa196`.
- All ten original frozen checkpoint SHA256 digests verified from `RESULTS.json`. Five SiLU and five raw-linear (experiment-labeled `tanh`) gate modes. Each architecture has **100,680 learned parameters** and the same 408-float persistent neural core.
- Exact existing DEV sampler reconstructed: eight predetermined context-128 windows from each of six DEV chapters, batched identically. Each checkpoint is evaluated on **3,072 historical labels at lag64** and **1,536 at lag96**. Recomputed source-history negative log-loss matched archived per-checkpoint measurements within 1e-5.
- Only original `FIT` chapter token frequencies are used to build the fallback: 128 positive probabilities, 0.5 pseudocount per category, 512 bytes when stored as FP32. Its 128 static categorical values are **extra nonlearned metadata, not free state**. No new optimizer or training, no DEV targets or logits enter the routing policy.

## Exploratory real DEV results

Mean cross-entropy in **nats per historical target character**; lower is better.

| Lag | Original frozen TAC (10-checkpoint mean) | FIT-only unigram prior | Selective policy | Decision |
|---|---:|---:|---:|---|
| 64 | 3.460926 | 3.531907 | 3.460926 | Original TAC history |
| 96 | 5.219261 | 3.542343 | 3.542343 | Abstain, categorical fallback |

Per-checkpoint lag96 outcomes:

| Original gate | Seed | Original lag64 NLL | Original lag96 NLL | FIT prior lag96 NLL | Source minus FIT at 96 |
|---|---:|---:|---:|---:|---:|
| raw-linear (`tanh` label) | 1701 | 3.4994 | 5.3101 | 3.5423 | +1.7677 |
| SiLU | 1701 | 3.4925 | 5.2963 | 3.5423 | +1.7539 |
| raw-linear | 1702 | 3.3189 | 4.4112 | 3.5423 | +0.8689 |
| SiLU | 1702 | 3.2989 | 4.3449 | 3.5423 | +0.8026 |
| raw-linear | 1703 | 3.5844 | 5.8235 | 3.5423 | +2.2812 |
| SiLU | 1703 | 3.5851 | 5.8328 | 3.5423 | +2.2905 |
| raw-linear | 1704 | 3.4060 | 5.3954 | 3.5423 | +1.8530 |
| SiLU | 1704 | 3.3852 | 5.2953 | 3.5423 | +1.7530 |
| raw-linear | 1705 | 3.5246 | 5.2765 | 3.5423 | +1.7342 |
| SiLU | 1705 | 3.5142 | 5.2066 | 3.5423 | +1.6642 |

**All ten** source models had higher lag96 NLL than the FIT-only unigram. The mean reduction is **1.676918 nats/character** at 96. The fallback did not reconstruct the correct historical event; it simply emitted the known distribution of characters in FIT text. It is a *safe distributional alternative after abstention*, not a genuine 96-character retrieval capability.

At lag64, original TAC is better than the FIT-only unigram on the ten-seed average; two individual checkpoints did not beat the FIT prior. The fixed policy's exact trained-lag trust is source-preregistered, **not a per-seed oracle**.

## Safety and probability contract

Code in `selective_history.py`:

1. A query is permitted only for a valid positive integer delay and source context length.
2. The **only** trusted historical delay is 64, exactly the original training objective.
3. Every other delay returns `abstained=True` and a strictly positive FIT-only 128-character fallback. The model history head is **not consulted** for these delays. A downstream application can instead refuse the request.
4. The selected result includes the trust/abstention label. It does not silently present fallback as reconstructed original text.
5. The route is a pure function of the requested lag and the fixed 64 training specification. It cannot consult a historical target, future character or DEV loss at runtime.

Companion PNDS `LanguageSelectiveRetrieval.lean` proves exact trust/fallback behavior and positive, normalized fallback, as well as baseline probability floors for optional fixed convex mixtures; targeted proof audit increased from **67 to 73**, pending/confirmed via latest CI. It is an exact-real scoped proof, not a model accuracy or calibration theorem.

`audit_selective_history_dev.py` is a no-fit, `weights_only=True`, source/hash-gated replay script requiring the original local archive and Vim9.1 documents; the private binary checkpoints remain **unavailable to cold GitHub CI**. CI checks the adapter's shape, normalization, label-blind routing and fail-closed behavior using constructed unit fixtures.

## Critical scientific limitations

- The development set had already been examined in prior experiments; choosing a fallback because lag96 was bad is a DEV-derived research decision, not independent confirmation. There is no new untouched TEST claim.
- This is source-backed human technical manual text only; broad language generalization and safety remain unknown.
- A static FIT unigram cannot recover entity identity, event timing or a specific past character. Lower log-loss only reflects safer categorical uncertainty on this corpus.
- Extra static prior storage (128 FP32 values = 512 bytes) is explicitly accounted for, as is any source archive storage.
- A full selective classifier should be qualified on more than one historical lag and real contexts before extending the trusted set. The correct policy today is **exact lag64-only trust**, not blind trust of all shorter lags.

## Next research gate

Preregister a real-document train/evaluate protocol comparing original 100,680-parameter TAC with smooth-key, conditioned-rotation, era-aware source identity and online-mean candidates. Match per-arm active learned parameter counts within 1%, same FIT/DEV splits, sample order, learning budget and input history access. Original 10 checkpoints remain frozen references. Train variants separately; store all checkpoints, sample-order hashes, per-seed results and machine replay. Include strong FIT-only unigram/bigram and matched-parameter GRU controls.

Primary claim must depend on **genuine historical retrieval improvement at registered lags and source identities** (not merely the abstention fallback), measured alongside next-character NLL, wrong-key controls, future-leakage controls, memory lesions, calibration, latency and any external-memory storage/I/O. Untouched TEST must remain protected.
