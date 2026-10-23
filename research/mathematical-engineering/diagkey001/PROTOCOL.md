# DIAGKEY001 — Owner-requested keyed associative recall stress test

**Scientific classification: SYNTHETIC MECHANISM DIAGNOSTIC ONLY. This cannot qualify real-world TAC-OSM capability, intelligence, architecture superiority, safe behavior, or deployment.** The owner requested a tightly bounded synthetic diagnostic first after ME074 found almost no immediate sensitivity to the 8×8 associative matrix. This is a narrow unit/mechanism study, not a rescission of the compulsory real-world evidence rule. Original ME073 weights, original BPI2012 files, original TEST and the reserved untouched robot task are NOT used or changed.

## Prior-art and what the mechanism test can and cannot reveal

This is not a novel benchmark. Read [Huang et al., ICML 2025](https://proceedings.mlr.press/v267/huang25ab.html) and [Wang, Shi & Fox, JMLR 2026](https://www.jmlr.org/beta/papers/v27/25-0903.html): Mamba/SSM, attention and delta-rule recurrent memory mechanisms can be analysed by associative recall. An 8×8 recurrent matrix has a different inductive bias/capacity than a full Transformer. We test whether the native TAC *implementation* learns a useful association under a fixed synthetic probe, not whether all possible TAC networks have or lack useful memory.

ME074's original age6 fast-state zero increased real next-event/EOR NLL from 0.246678 to 1.163264; association-zero increased it only 0.246678→0.247267. At age16 the association effect was +0.006387. These are *out-of-training-distribution frozen readout lesions*. Such small effects do not prove associative capacity globally, and the fast state was 64D in actual ME073, not 32D.

## Immutable diagnostic generator

Each input sequence consists of four WRITE(key,value) events with unique keys from a 16-key target namespace and unique randomly assigned values in 32 categories, followed by 16/64/256 distractor WRITE events with keys from a **disjoint** 48-key namespace, then a QUERY(target key, value=missing). The query key is uniformly selected from the four original writes, and its value is never supplied at the query step. Distractor values can coincide with target values but never carry the target key. New mapping and distractors generated for *every* sequence by deterministic SHA256-derived random seed; fixed training/test split namespaces and seed ranges. Stored target chance rate **1/32=3.125%**; do not accidentally leak the answer via the query value or future labels.

This probes four competing writes and interference; it is **not** many-key 32/64/96-pair capacity. A different experiment is needed to examine memory slot limits. The length is number of distractors, *not* total input sequence (4 writes + D distractors +1 query).

## Four fitted diagnostic architectures

1. Native TAC-style fast/slow gated nonlinear rank-one associative recurrence, 8×8 key-value matrix, normalized/delta error update; diagnostic width **88** (not the unchanged real ME073 64D). **101,554** actively trained/deployed parameters.
2. Re-architected TAC without associative write/read pathway, reallocating capacity to useful readout, no disconnected dummy matrix weights, width88. **101,560** trainables.
3. GRUCell with same event embeddings/query sequence and a useful readout, width128. **101,500** trainables.
4. Two-layer causal Transformer, width64, 4 heads, FF144, same event input; causal attention over complete available prefix. **101,616** trainables.

All share 32-value output, 3 event types, identical target/exposure, same optimizer, objective, seed order, and no external oracle. Absolute 95k–105k and relative 1% trainable and deployed parameters are required **before any fit**, checked from live model implementations. Maximum current count discrepancy 116/101616≈0.115%. Transformer has full causal access to all prefix tokens, which is an explicit memory-budget structural advantage, not matched inference-state footprint.

## Fixed original training protocol

Five seeds 75001..75005. For EACH arm/seed train a fresh model for 640 AdamW steps, batch16, weight decay 1e-4, lr0.001, gradient clipping1.0, cross entropy at query. Training schedules rotate delays 16/64/256 identically at consecutive steps with exactly the same stateless SHA-derived FIT sequence batch for every arm at matching seed/step. This tests in-distribution long-delay performance, *not* generalization from short trained delays to unseen long ones.

For each trained arm/seed evaluate 512 new deterministic TEST mappings for each delay, **never used for gradient updates**, report exact-answer accuracy, cross-entropy, random-chance oracle1/32, all five seed directions. Original weight/optimizer/RNG state checkpoint and complete corresponding numerical scores must be Git-owned, SHA-identified, and clean-checkout reloaded; no retroactive seed replacement or hyperparameter tuning on results.

**Compute mismatch qualification:** exact supported-operator PyTorch forward+backward FLOPs are profiled at batch2 for each sequence length, but Torch profiler may omit fused kernels, softmax and some CPU operations. Report each arm's supported operator FLOPs separately; if full training FLOP coverage, actual computation and paired FLOP parity (within the existing required 1%) are not demonstrated, **NO FAIR ARCHITECTURE SUPERIORITY** follows regardless of accuracy. Equal optimizer steps and examples are not sufficient to assert equal compute.

**Nonqualification/kill criteria:** If at delays64 and256 the native model does not beat both GRU and retrained native-noassoc by a meaningful predeclared >=5 percentage points, on all five paired seeds with positive paired 95% binomial/case-cluster CI and valid parameter, compute and data parity, **associative-advantage claim fails on this diagnostic**. It does NOT logically refute every associative network or real-world advantage. If all models remain near chance, mark the training budget/test underpowered and DO NOT infer absence of memory. No synthetic success is ever a real research promotion.

## Actual priority after diagnosis

The next qualified training study remains a matched ~100k TAC/GRU/Transformer/direct-horizon competition on a new authentic source with verified 64/256-event histories and an independently unopened holdout. On original BPI2012 source there are zero actual >=256-history future-label cases and only four original FIT cases with >=128: synthetic recall cannot repair missing ecological validity. Real source and original Git checkpoint gates remain compulsory.

Original diagnostic code: `scripts/diagkey001_associative_recall.py`; scores/checkpoints: `research/replay-bundles/DIAGKEY001/`. This protocol registers the owner-requested narrow diagnostic without allowing it to be mistaken for authentic-data training.
