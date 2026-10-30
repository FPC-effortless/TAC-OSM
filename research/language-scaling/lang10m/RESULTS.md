# LANG10M001 verified pretraining gates

**Tokenizer pilot complete. Capacity profile complete. Neural training not started.**

The actual candidate has10001188 trainable/deployed model parameters. The fitted4096-piece byte-fallback tokenizer uses219 authenticated FIT sources,3630713 UTF-8 bytes and1170938 token instances in one corpus pass. All original source strings roundtrip exactly. This is a small tokenizer/capacity pilot, not sufficient unique data for full10M pretraining. Source/recipe/model/vocabulary hashes and original notices are archived in tokenizer/; a supplementary CPython libmpdec notice is recorded without changing any fitted bytes.

## Completed GPU evidence

Both runs used exactly two physically observed distinct TeslaT4s with15360MiB each. The initial profile passed all10 FP32 shapes but every AMP case failed the Float-H/Half-RHS concept solve. Those negatives remain unchanged in capacity/attempt1.

The repair promotes only the10M evidence projection output to FP32 under AMP. It reuses original parameter objects/state keys and consumes no extra initialization RNG. Actual count remains10001188. Constructed FP32 output and all-parameter gradients match the legacy projection exactly; CPU-BF16 autocast also passes. Original500K source and experiments were not changed. PNDS source linkage points to the repaired source; the upper partial mathematics is unchanged, while full composition/IEEE/AD certification remains OPEN.

The second matrix passed its registered forward/loss-precision, finite-gradient, same-weight and capacity gates in all20 cases. All20 initial weight hashes match the original baseline and every final hash equals its initial hash. Original FP32 diagnostic losses also match exactly. There were zero optimizer updates and zero held-out inputs. Verification independently reconstructs every sampled FIT owner/offset/hash without model inference.

| Batch × context | Precision | Median forward+backward tokens/s | Peak allocated GiB | Zero-gradient parameter tensors |
| --- | --- | ---: | ---: | ---: |
| 1 × 128 | amp_fp16 | 2422 | 0.125 | 0 |
| 1 × 128 | fp32 | 2808 | 0.125 | 0 |
| 2 × 128 | amp_fp16 | 4642 | 0.125 | 0 |
| 2 × 128 | fp32 | 5330 | 0.132 | 0 |
| 4 × 128 | amp_fp16 | 9139 | 0.151 | 2 |
| 4 × 128 | fp32 | 10869 | 0.189 | 0 |
| 8 × 128 | amp_fp16 | 18031 | 0.231 | 2 |
| 8 × 128 | fp32 | 21250 | 0.327 | 0 |
| 2 × 256 | amp_fp16 | 9220 | 0.153 | 1 |
| 2 × 256 | fp32 | 10467 | 0.195 | 0 |
| 4 × 256 | amp_fp16 | 18060 | 0.235 | 2 |
| 4 × 256 | fp32 | 20273 | 0.336 | 0 |
| 2 × 512 | amp_fp16 | 16097 | 0.240 | 2 |
| 2 × 512 | fp32 | 18881 | 0.346 | 0 |
| 4 × 512 | amp_fp16 | 28509 | 0.406 | 2 |
| 4 × 512 | fp32 | 27188 | 0.609 | 0 |
| 1 × 1024 | amp_fp16 | 15888 | 0.243 | 2 |
| 1 × 1024 | fp32 | 18610 | 0.356 | 0 |
| 2 × 1024 | amp_fp16 | 27542 | 0.414 | 2 |
| 2 × 1024 | fp32 | 26622 | 0.627 | 0 |

These rates are single-GPU, synchronized forward+backward measurements with2 warmups/5 measured repetitions. They exclude optimizer steps, data loading and distributed communication. Allocated VRAM is not total process/device memory; reserved peaks are preserved in raw receipts. AdamW moments were not allocated; their additional80009504 bytes are an estimate only. No maximum-batch search or persistent-state decode claim was made.

## Important remaining precision gate

Unscaled AMP has1–2 entirely zero gradient tensors in8/10 shapes, while FP32 has zero such tensors in every shape. This may indicate mixed-precision loss/underflow. Tensor names and full FP32/AMP gradient parity were not measured, so the cause is not established. Passing finite-gradient and forward-tolerance gates is not proof of equivalent backward updates or safe FP16 training. GradScaler training stability remains OPEN. FP32 is the conservative validated starting point; AMP offered little benefit at the largest tested shape and was slower on several small shapes.

## What CODEVALUE004 contributes

CODEVALUE004 completed/verified separately: all24 fits,120 original checkpoints,24 selections,280 original score files and143360 saved suggestions checked. Its primary TAC broader-corpus+cosine comparison passed: NLL2.3380→2.0279, gain0.3101 with paired95% CI0.2048–0.4231, positive in all5 projects/all3 seeds. The corpus main effect was0.3024; cosine effect0.0078 had CI−0.0014–0.0177. Broader data is the clear priority; a positive isolated TAC cosine effect is uncertain. GRU stayed stronger and all TAC utility screens failed. Its protocol and CODEVALUE003 reference remain unchanged.

## Next research gate

Memory fits the tested10M shapes comfortably, but hardware feasibility is not a trained language result. Before optimizer training, expand/authenticate enough unique mixed FIT material, freeze new DEV/transfer boundaries, register matched10M controls with the same tokenizer/exposure/selection/compute treatment and checkpoint/replay, and scope the owner-directed10M training contract. The global100K rule is not silently relaxed. No pretrained capability, English quality, developer benefit, Android performance, revenue, promotion or independent qualification is claimed.

Both profile matrices, logs, source/archive/bootstrap hashes, tokenizer artifacts and CAPACITY_VERIFICATION.json are preserved. The timed coordinator work across both runs totals about1.10 GPU-minutes excluding platform startup, within the registered combined12-GPU-minute allowance. No further allocation or automatic retry was made.
