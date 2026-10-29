# LANGRECOMP053 — original trained TAC full-graph training-mode backward parity; VRAM NOT measured

**Original source CPU research actually measured 29 October 2026.** [Draft PR #587](https://github.com/FPC-effortless/TAC-OSM/pull/587), [original successful GitHub CPU Actions #40830969085](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40830969085), [original metrics artifact #12878771812](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40830969085/artifacts/12878771812). Five constructed forward/gradient/source-integrity tests PASS; original publisher source-license and no-fit CPU planning PASS. **No optimizer step, new fit, checkpoint update, GPU, protected TEST or model release.** Original 499,623p `LANGLAYER040 state48_no_graph` seed1701 trained on legally published CPython FIT320, original strict checkpoint SHA256 `0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb`, sampled FIT schedule original SHA256 `45985ef86e7df644a3acfeee73fe781bbd8d2cb290ac6465dfe670579e6f702f`; original optimizer/RNG and weights verified read-only before diagnostics.

## What was actually tested

Real original source first-window batch1×32 and batch2×64 byte next-character targets (true causal x[t+1]), original source-owner/offset SHA256 recorded in raw evidence. Actual full trained model in train() mode: native full original `model(x)` graph vs PyTorch 2.10 **nonreentrant full-forward `checkpoint(lambda tokens:model(tokens),x,use_reentrant=False,preserve_rng_state=True)`**. Entire 499,623-parameter original graph includes native/aligned associative memories, wave state, concept states and Laplacian read, upper 48D recurrence and SwiGLU; no memory path was bypassed. Equal original token tensors, original frozen parameter values, RNG and original next-byte NLL, no optimizer update. Model input integer tokens don't require gradients, so nonreentrant path is essential.

| Original FIT minibatch | Output logits/NLL difference | Full original parameter gradient elements checked | Maximum gradient abs difference | Native logical saved tensor requests | Checkpoint logical saved tensor requests | Reduction in logical requested bytes |
|---|---|---:|---:|---:|---:|---:|
| B1×T32 | **exact 0** |499,623 (55 named gradient tensors)|**0**|239 |6 | 4,026,908B → 66,052B (**98.36%**) |
| B2×T64 | **exact 0** |499,623 (55 named gradient tensors)|**0**|255 |6 | 11,120,540B → 264,196B (**97.62%**) |

All 55 named trainable parameter gradients present, finite and elementwise equal under original FP32 CPU eager execution (max diff0), equal original NLLs 1.27308226 / 1.79289210, same original input and label source, unchanged original parameter tensors, identical post-backward CPU RNG. Tolerances predeclared atol3e-5,rtol1e-4; actual errors zero. This is **empirical CPU autograd equivalence for two small original source minibatches**, not a Lean/IEEE/finite precision formal proof across all contexts/optimizer steps.

## Computation cost — exploratory single-shot, not benchmark latency

| Batch | Native forward ms | Native backward ms | Recompute forward ms | Recompute backward ms |
|---|---:|---:|---:|---:|
| B1×32 |7.61|7.42|**701.13**|16.37|
| B2×64 |9.40|11.77|10.80|22.80|

First recomputation call shows extreme initialization/warm-up overhead (701ms forward versus ~11ms on the larger subsequent call). **Do not interpret the first row as stable computational penalty; runs are single-shot and not warmup-balanced.** The second row shows additional recomputation backward cost (22.80ms vs11.77ms) on this host. A separately frozen repeated warmup/CPU gradient profile is required for stable latency or throughput claims. All CPU metrics refer to one GitHub Ubuntu host, batch1/2 short genuine source windows.

## Important measurement boundary: this is NOT 98% peak memory or original GRAPH011

These 98.36% and 97.62% reductions are sums of **logical saved-tensor hook request bytes** collected during forward. They include potential aliases/repeat saves, omit allocator/RSS and physical CUDA residency, count short-lived requests differently across native/checkpoint, and do not account for model parameter tensors (499,623 FP32 parameters ≈1,998,492B), gradients, optimizer states, transient rematerialization tensors, Python heap, runtime/framework, CUDA kernels, or workspace. Recomputing full model during backward may allocate many temporary tensors; whole-process peak memory can be dramatically different. A claim that total VRAM shrank 98% or 99% would be **false or unmeasured** from this diagnostic. Original provided GRAPH011 figures **5.51MB → 0.0014MB** still have no original source/measurement receipt in this repo lineage, and original GRAPH011 code has NOT been integrated. The present path is a distinct, transparent PyTorch checkpoint adaptation.

**Verdict:** Whole-training-graph nonreentrant checkpointing is a viable parity-preserving mathematical implementation candidate for original trained TAC; on this short authentic FIT setup it successfully reconstructs all parameter gradients and avoids most logical autograd saved-tensor requests. Native inference streaming remains superior to full-prefix per-token replay, and the prior LANGMEMORY051 finding about source-context identity loss is separate.

**Next verification gate:** Before any true graph-memory speed/peak claim, preregister warmed paired original source batches and compare source/gradient parity with checkpointing on/off; use actual peak CUDA allocated/reserved measurement only on separately reviewed owner-approved genuine **dual T4** GPU environment (both physical GPUs used, source and checkpoint unchanged, no protected TEST), with source-level PNDS Lean/Python graph and AD/finite-precision obligations OPEN. Do not merge, spend GPU quota, or promote model. Preserve original JSON evidence, job logs and SHA manifest in repo and user's persistent Library.
