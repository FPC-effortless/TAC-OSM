# DEV-007B/C — Small CED/CSA2-inspired retrieval pilot (development only)

> **Status: DEVELOPMENT-ONLY, NOT CONFIRMATORY.** This report analyzes an independent CPU PyTorch research reference, **not** a successful TAC-OSM integrated world model, a replication of DeepSeek's complete CED, or a proof of faster end-to-end inference. No existing DEV-003–006 evidence is modified.

## Architecture and task

Four arms: (A) 4-layer causal Transformer, (B) 2-layer causal encoder + 2-layer CED decoder with independent encoder-projected global KV, (C) same with shared global KV across decoder layers, (D) shared KV with first-layer top-k full scan and second-layer index reuse. All use 16-channel observations, 2 attention heads, four total blocks; CED includes width-4 decoder local attention. Reference score-pair ledger counts executed logical matrix entries, **not** measured FLOPs or actual KV allocation for every arm.

Task: procedurally generate H keyed observations and a query. Each matched pair has identical queries, keys and distractors but switches the target historical value (0 vs 1). The query is a separately corrupted view of its target key. There are no labeled oracle indices in model input. The cosine nearest-neighbor control solves 100% of *this easy artificial generator* at all tested H. This tests learned retrieval, **not online long-term memory, planning, action-conditioned world dynamics, generalization to real scenes, or TAC-OSM's end-to-end selective-computation thesis**.

Training: same observed batches in all arms, AdamW 0.002, 600 updates, 8 opposite-past pairs/step, train H=8 with hard-distractor fraction 0.20. Eval held-out seeded pairs at H=8/16/32, hard fractions 0 and 0.5, 96 pairs per cell per seed; five independent training seeds 43,59,71,83,97. Every arm sees 576? Correction: each seed/arm receives 6 cells * 96 pairs * 2 = **1,152 evaluated episodes**; across 4 arms and 5 seeds = **23,040** model episode predictions. Equal training data, but **not parameter-, training-FLOP-, or optimizer-initialization-matched across different architectures**. In DEV-007C the CED shared and sparse arms start with exactly matching weights, verified by tensor-hash equality for all five seeds. The longer training schedule and seed set were chosen during development after looking at the 120-step pilot; they are **not prospectively frozen independent confirmatory evidence**.

## Results: DEV-007C matched CED shared vs sparse initial weights

| Arm | Trainable parameters | Mean task accuracy | H=8 | H=16 | H=32 | Mean gap vs Transformer |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 8,674 | 93.96% | 98.28% | 94.64% | 88.96% | — |
| CED dense | 10,722 | 94.91% | 98.75% | 95.21% | 90.78% | +0.95 pp |
| CED shared | 10,210 | 94.05% | 98.02% | 95.00% | 89.11% | +0.09 pp |
| CED sparse | 10,210 | 93.85% | 97.92% | 94.48% | 89.17% | -0.10 pp |

Exploratory 95% Student-t intervals over **five seed-level paired differences**, not episode-level pseudoreplication:
- Dense CED minus Transformer: +0.95 pp, approx [-2.14,+4.05].
- Shared CED minus Transformer: +0.09 pp, approx [-1.35,+1.52].
- Sparse CED minus Transformer: -0.10 pp, approx [-1.47,+1.26].

No robust model-quality advantage shown. Baseline parameters are smaller and cross-architecture comparisons are not matched. Even a narrow descriptive confidence interval would not solve this design flaw.

## Logic count vs measured CPU speed (H=32, 192 observations per evaluation batch)

| Arm | Reference attention score pairs (encoder + global decoder + local window) | Decoder global KV scalars |
|---|---:|---:|
| Transformer | 1,672,704 | N/A in this CED-specific ledger |
| Dense CED | 1,774,080 | 405,504 |
| Shared CED | 1,774,080 | 202,752 |
| Sparse CED | 1,406,592 | 202,752 |

The sparse CED needs **20.7% fewer reference attention score pairs** than dense CED, but the first top-k selection computes the full H×H index scores. Its complexity is not sublinear. Shared and sparse modes halve the *CED decoder-global KV producer's scalar count* relative to dense CED; this is **not** a comparison of full model memory or peak process memory.

Unoptimized eager CPU PyTorch latency probe (H=32, batch 32, 2 threads, 40 repeats, after 10 warmups; *single-run diagnostic, not production benchmark*): Transformer median 2.268 ms, dense CED 3.487 ms, shared CED 4.306 ms, sparse CED 5.239 ms. The sparse reference is **slower** despite lower logical score-pair counts. Kernel/gather overhead, Python and CPU implementation matter; do not project these numbers onto GPU/flash-attention kernels.

## Verification and provenance

- 15 small unit tests passed (12 reference mechanism cases + 3 data/benchmark tests).
- Saved the 20 trained PyTorch checkpoints, per-evaluation-cell raw targets, predictions, retrieved positions, input SHA-256 hashes, source SHA-256 and checkpoint SHA-256 in the user's exported research archive.
- The replay validator reloaded each checkpoint, checked source and checkpoint hashes and exactly reproduced **all 23,040** evaluation predictions and associated raw evaluation cells. This **does not reproduce the training process**, prove hidden-test isolation, or confer scientific independence.
- Distinct train and eval RNG streams. Handcrafted synthetic generator, no external domains tested.
- Prior 120-step/three-seed and 600-step/five-seed non-matched-initialization pilots were exploratory; the latter average values were 93.96/93.28/93.54/93.66% for Transformer/dense/shared/sparse. Those do not replace the DEV-007C matched-CED pilot.
- **Artifact limitation:** raw checkpoint ZIP and runnable script remain a local exported conversation research artifact; this branch document alone cannot independently reproduce the run. Do not merge as verified capability evidence until code and raw archive are reviewed and preserved in appropriate immutable CI artifacts.

## Why this does *not* reproduce DeepSeek-V4.1-Flash

The prototype processes the full historical sequence in decoder layers; DeepSeek's CED efficiency also relies on avoiding most decoder prompt prefill, using bounded replay and other system optimizations. No realistic query-only decoding path, FP4 cache, MoE, learned hierarchical sparse index, or fused kernels are implemented here. Therefore **CED prefill reductions claimed by DeepSeek are not measured by this pilot**.

## Next steps (priority order)

1. Preserve and peer-review source/weights/raw evidence in CI while retaining negative findings.
2. Implement a *query-only* CED decoder and explicit encoder-prefill vs decoder-decode accounting. Validate cached vs recomputed equivalence and invalidate on memory writes.
3. Build parameter-matched and FLOP-matched baselines; also measure GPU kernel efficiency where supported. No expected GPU speedup is claimed.
4. Evaluate same architectures on the TAC-OSM DEV-006 finite-memory interference and delayed-retention grid **without changing the existing benchmark or silently removing its 768-scalar state constraint**.
5. Add a truly indexed retrieval mechanism only with build/update/query/storage cost accounting, then test whether total cost scales with the relevant subset instead of complete history.
6. Test future-latent prediction only after retention is improved; DEV-006's task transfer and MSE remain negative/uncertain.

Sources: DeepSeek V4.1-Flash https://arxiv.org/abs/2609.19969 ; YOCO https://arxiv.org/abs/2405.05254 ; IndexCache https://arxiv.org/abs/2603.12201.
